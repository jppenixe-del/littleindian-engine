// 🦅 napk9_train_li11.rs — "LI11" (Little Indian 11): nova arquitetura NapK9, L1=512,
//    inspirada no que aprendemos do Stockfish atual (master, lido a sério desta sessão —
//    ver tools/sf_nnue_interpret.cpp, validado byte a byte contra um binário SF real) mas
//    NÃO uma cópia: usa os NOSSOS full-threats (9216 dims, fórmula aligeirada pelo Reckless
//    a partir da ideia original do SF — ver CREDITS.md), não o FullThreats novo do SF
//    (60720 dims, feito para L1=1024, dimensionado para hardware que não temos).
//
//    O QUE SE ADOTA DO SF (ideias, não código nem pesos — GPL-3.0, lido só conceptualmente):
//      1. Ativação em PARES no acumulador: clip(acc,0,1) depois multiplica metade×metade
//         (bullet já tem isto pronto: `.crelu().pairwise_mul()`, confirmado nesta sessão —
//         não precisa de operação nova no treino). Dá termos de 2ª ordem "de graça" ao stack
//         denso sem aumentar os parâmetros dele.
//      2. Ligação de salto no FC0: 32 neurónios reais + 1 neurónio extra linear que entra
//         direto na soma final, sem passar por mais nenhuma não-linearidade — o mesmo
//         princípio do nosso PSQT (termo linear a saltar a parte densa), uma camada mais
//         tarde no pipeline.
//      3. Os 32 neurónios reais do FC0 levam DUAS ativações em paralelo (ao quadrado-cortada
//         E cortada simples), concatenadas — dá ao FC1 acesso tanto ao termo linear quanto ao
//         quadrático da mesma camada.
//
//    O QUE NÃO SE ADOTA: o FullThreats novo do SF (60720 dims) — ver nota acima. Cabeça
//    única desde o início (bullet/small confirmados mortos em jogo real nesta sessão,
//    ver evaluateAllHeads/evalheads) — sem saídas fantasma, sem risco do "Cycle found"
//    documentado no LEIA-ME.md antigo (esse bug era especificamente sobre 2+ saídas a
//    partilhar o mesmo acumulador; aqui há só 1 saída real).
//
//    QUANTIZAÇÃO: contrato NOSSO, não do SF (o deles tem shifts/divisores calibrados à
//    escala interna deles, copiá-los literalmente seria o MESMO erro do clipping ±0.99 —
//    ver project_v11_eval_bug). accw/psqtw mantêm QA=255 (igual ao histórico). fc0/fc1
//    usam ×64 (mesmo papel que QB=64 tinha nas l1/l2 antigas). fc2 usa ×(64*64) (a cadeia
//    acumula QB duas vezes até à última camada, mesma lógica que o l3 antigo tinha com
//    255*64 — aqui não há QA no meio da cadeia densa, só nos dois extremos: acumulador e
//    psqt). O skip (fc0_out[32]) viaja com a MESMA escala do fc0 (×64) — combina-se com o
//    fc2_out (já em ×64*64) só depois da desquantização em float no lado C++, nunca em
//    inteiro bruto (ver nota no engine).
//
//    VALIDAR ANTES DE QUALQUER TREINO LONGO: treino curto (10-30 superbatches) + checar
//    `eval` numa posição simétrica (deve ficar perto de 0, não saturar em ±3000) — foi
//    exatamente isto que apanhou o bug do clipping em project_v11_eval_bug; repetir aqui
//    como primeira validação, antes de gastar GPU a sério.

use bullet_lib::{
    nn::optimiser::AdamW,
    napk9_v10::{NapkInputV10, NapkV2MaterialBuckets},
    napk9_v10_binpack_loader::NapkBinpackLoader,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{loader, loader::sfbinpack::{MoveType, PieceType, TrainingDataEntry}, ValueTrainerBuilder},
};

const SCALE: f32 = 400.0;   // motor agora usa OUTPUT_SCALE_CP=400 (ver commit do mesmo dia)

// FC0: 32 neurónios reais (preserva a nossa constante histórica BIG_DL1=32, decisão do
// utilizador OD-1 — não o 31 literal do SF) + 1 de salto = 33 saídas totais.
const FC0_REAL: usize = 32;
const FC0_TOTAL: usize = FC0_REAL + 1;
const FC1_OUT: usize = 32;

fn save_ids_li11() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("accw").round().quantise::<i16>(255),
        SavedFormat::id("accb").round().quantise::<i16>(255),
        SavedFormat::id("psqtw").round().quantise::<i16>(255),
        SavedFormat::id("psqtb").round().quantise::<i16>(255),
        SavedFormat::id("fc0w").round().quantise::<i16>(64),
        SavedFormat::id("fc0b").round().quantise::<i16>(64),
        SavedFormat::id("fc1w").round().quantise::<i16>(64),
        SavedFormat::id("fc1b").round().quantise::<i16>(64),
        SavedFormat::id("fc2w").round().quantise::<i16>(64 * 64),
        SavedFormat::id("fc2b").round().quantise::<i16>(64 * 64),
    ]
}

fn main() {
    let l1: usize = std::env::var("NAPK_L1").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let threats_mode: u8 = match std::env::var("NAPK_THREATS").unwrap_or_default().as_str() {
        "none" => 0, "640" => 1, _ => 2,
    };
    let n_in: usize = 1 + match threats_mode { 0 => 22528, 1 => 23168, _ => 31744 };
    println!("🦅 napk9_train_li11 — L1={l1}, scale={SCALE}, FC0={FC0_REAL}+1, FC1={FC1_OUT}");
    println!("   NAPK_THREATS: {}", match threats_mode { 0 => "NONE", 1 => "640 clássico", _ => "FULL 9216 (nosso, Reckless-lightened)" });

    // pairwise_mul() corta o acumulador a meio e multiplica — l1 tem de ser par.
    assert!(l1 % 2 == 0, "NAPK_L1 tem de ser par (pairwise_mul corta a meio)");
    let dense_in = l1; // l1/2 (paired, por perspetiva) × 2 perspetivas = l1 outra vez

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(NapkInputV10 { mode: threats_mode })
        .output_buckets(NapkV2MaterialBuckets::<8>)
        .save_format(&save_ids_li11())
        .build_custom(move |builder, (stm, ntm, buckets), targets| {
            let acc_layer  = builder.new_affine("acc",  n_in, l1);
            let psqt_layer = builder.new_affine("psqt", n_in, 8);

            let fc0 = builder.new_affine("fc0", dense_in, FC0_TOTAL);
            let fc1 = builder.new_affine("fc1", FC0_REAL * 2, FC1_OUT);
            let fc2 = builder.new_affine("fc2", FC1_OUT, 1 * 8);

            // 1) ativação em pares (clip 0..1, depois multiplica metade × metade)
            let acc_us_paired   = acc_layer.forward(stm).crelu().pairwise_mul();
            let acc_them_paired = acc_layer.forward(ntm).crelu().pairwise_mul();
            let x = acc_us_paired.concat(acc_them_paired);  // largura = l1

            let psqt_b = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);

            // 2) FC0 -> 32 reais + 1 salto
            let fc0_out  = fc0.forward(x);
            let skip     = fc0_out.slice_rows(FC0_REAL, FC0_TOTAL);       // 1 largura, sem ativação
            let real_out = fc0_out.slice_rows(0, FC0_REAL);               // 32 largura

            // 3) dupla ativação nos 32 reais: ao quadrado-cortada + cortada simples, concatenadas
            let sqr_path = real_out.sqrrelu();
            let lin_path = real_out.crelu();
            let concat64 = sqr_path.concat(lin_path);  // largura 64

            let fc1_out = fc1.forward(concat64).crelu();
            let fc2_out = fc2.forward(fc1_out).select(buckets);  // 1 saída, escolhida pelo bucket

            // soma final: FC2 + salto (cru, sem ativação) + psqt (linear, sem ativação)
            let out = fc2_out + skip + psqt_b;

            let loss = out.sigmoid().squared_error(targets);
            (out, loss)
        });

    let tag = std::env::var("NAPK_TAG").unwrap_or_default();
    let net_id = if tag.is_empty() { format!("NAPKa0s_li11_{l1}") } else { format!("NAPKa0s_li11_{l1}_{tag}") };
    println!("🦅 NET_ID: {net_id}");

    let lr_start: f32 = std::env::var("NAPK_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let batch_size: usize = std::env::var("NAPK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let bps: usize = std::env::var("NAPK_BPS").ok().and_then(|s| s.parse().ok()).unwrap_or(6_104);
    let end_sb: usize = std::env::var("NAPK_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(800);
    let final_lr = lr_start / 100.0;
    let wdl_weight: f32 = std::env::var("NAPK_WDL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let save_rate: usize = std::env::var("NAPK_SAVERATE").ok().and_then(|s| s.parse().ok())
        .filter(|&r: &usize| r > 0).unwrap_or(10).min(end_sb);
    println!("🦅 batch={batch_size} bps={bps} end_sb={end_sb} wdl={wdl_weight} save_rate={save_rate}");

    let schedule = TrainingSchedule {
        net_id, eval_scale: SCALE,
        steps: TrainingSteps { batch_size, batches_per_superbatch: bps, start_superbatch: 1, end_superbatch: end_sb },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_weight },
        lr_scheduler: lr::CosineDecayLR { initial_lr: lr_start, final_lr, final_superbatch: end_sb },
        save_rate,
    };
    let settings = LocalSettings {
        threads: 10, test_set: None, output_directory: "checkpoints_napk", batch_queue_size: 64,
    };

    if let Ok(resume_dir) = std::env::var("NAPK_RESUME") {
        if !resume_dir.is_empty() { trainer.load_from_checkpoint(&resume_dir); }
    }

    let binpack_path = std::env::var("NAPK_BINPACK")
        .unwrap_or_else(|_| "/mnt/d/Nap2Siriux/training/sf_test80.binpack".to_string());
    let paths: Vec<&str> = binpack_path.split(':').filter(|s| !s.is_empty()).collect();
    let buffer_mb: usize = std::env::var("NAPK_BINPACK_BUFFER_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let binpack_threads: usize = std::env::var("NAPK_BINPACK_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let min_ply: u16 = std::env::var("NAPK_MIN_PLY").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    println!("🦅 NAPK_BINPACK ({} ficheiro(s)), buffer={buffer_mb}MB/{binpack_threads}t, min_ply={min_ply}", paths.len());
    for p in &paths { println!("   - {p}"); }

    let filter = move |entry: &TrainingDataEntry| {
        let in_check = entry.pos.is_checked(entry.pos.side_to_move());
        let normal = entry.mv.mtype() == MoveType::Normal;
        let dest_empty = entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
        entry.ply >= min_ply && !in_check && entry.score.unsigned_abs() <= 10000 && normal && dest_empty
    };

    // 🦅 NAPK_DATA tem prioridade sobre NAPK_BINPACK: .data2 já passou pelo binpack_to_data.rs
    // (com --dist possível, quota por bucket de material) -- usado para testar se o
    // desbalanceamento de buckets (não suportado pelo loader on-the-fly) explica o
    // enviesamento persistente do startpos. Ver project_li11_design / memória do dia.
    if let Ok(data_path) = std::env::var("NAPK_DATA") {
        let dpaths: Vec<&str> = data_path.split(':').filter(|s| !s.is_empty()).collect();
        println!("🦅 NAPK_DATA ({} ficheiro(s), .data2 -- ignora NAPK_BINPACK)", dpaths.len());
        for p in &dpaths { println!("   - {p}"); }
        let data_loader = loader::DirectSequentialDataLoader::new(&dpaths);
        trainer.run(&schedule, &settings, &data_loader);
    } else {
        let data_loader = NapkBinpackLoader::new_concat_multiple(&paths, buffer_mb, binpack_threads, filter);
        trainer.run(&schedule, &settings, &data_loader);
    }

    println!("🦅 FIM. checkpoints_napk/{}/", std::env::var("NAPK_TAG").unwrap_or_default());
}
