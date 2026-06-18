// 🦅 napk9_train_v10_binpack.rs — treino V10 100% on-the-fly: lê o binpack do Stockfish a
//    direito (NapkBinpackLoader, napk9_v10_binpack_loader.rs), SEM binpack_to_data.rs nem
//    .data2 nenhum no meio. Threats continuam on-the-fly no NapkInputV10::map_features, como
//    sempre — isto só elimina o passo de conversão para disco. Arquitetura igual ao
//    napk9_train_v10.rs (acc/psqt partilhados + bullet/small/big), config alinhada à do Coda
//    (ver napk9_train_v10_coda.rs e a nota de proveniência abaixo).
//
//    PROVENIÊNCIA (conceptual só, sem copiar código — Coda não tem licença declarada, ver
//    CREDITS.md): valores confirmados lendo github.com/adamtwiss/coda/training/configs/
//    v7_1024h16x32s.rs (a config dele mais próxima da nossa rede L1=1024) a 2026-06-18:
//    eval_scale=400, filtro ply≥16+!check+|score|≤10000+Normal+destino-vazio (IGUAL ao nosso,
//    não só ply/check/score), batch=16384, bps=6104, end_superbatch=800, buffer=256MB/4
//    threads, wdl=0.0 (puro cp, sem mistura de resultado). Mantemos o nosso wdl=0.3 por
//    defeito (decisão já tomada e documentada no napk9_train_v10.rs — resultado real ajuda
//    com finais Syzygy) mas é trivial pôr NAPK_WDL=0 para replicar o Coda exatamente.
//
//    INSTALAÇÃO: copiar napk9_v10.rs + napk9_v10_features.rs + napk9_v10_binpack_loader.rs
//    p/ o src/ do bullet (lado a lado), registar SÓ `mod napk9_v10;` e
//    `mod napk9_v10_binpack_loader;` no lib.rs (nunca os 2 ficheiros do mesmo módulo — é o
//    bug "NapkInputV10 has no field mode" do s29 cont.17, o include! já traz o features),
//    este ficheiro p/ examples/.
//
//    USO: NAPK_BINPACK="/caminho/test80.binpack:/caminho/outro.binpack" NAPK_L1=768 \
//           cargo run --release -p bullet_lib --example napk9_train_v10_binpack --features cuda
//    (sem NAPK_DATA, sem NAPK_SAVERATE-por-épocas — aqui o progresso mede-se em superbatches,
//    não há .data2 prévio p/ contar posições e calcular épocas automaticamente.)
//
//    L1=768 (default): o motor lê L1 da rede dinamicamente (MAX_L1=1536, nnue_net.cpp) — não
//    há nada cravado a 1024 no lado do C++. 768 é mais leve/rápido a treinar e a avaliar do
//    que 1024 (acumulador mais pequeno); ajusta com NAPK_L1 se quiseres comparar.

use bullet_lib::{
    nn::optimiser::{AdamW, AdamWParams},
    napk9_v10::{NapkInputV10, NapkV2MaterialBuckets},
    napk9_v10_binpack_loader::NapkBinpackLoader,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{loader::sfbinpack::{MoveType, PieceType, TrainingDataEntry}, ValueTrainerBuilder},
};

const SCALE: f32 = 400.0;   // CODA v7_1024h16x32s confirmado

const BIG_L2: usize = 32;
const BIG_L3: usize = 32;

fn save_ids_all() -> Vec<SavedFormat> {
    let mut v = vec![
        SavedFormat::id("accw").round().quantise::<i16>(255),
        SavedFormat::id("accb").round().quantise::<i16>(255),
        SavedFormat::id("psqtw").round().quantise::<i16>(255),
        SavedFormat::id("psqtb").round().quantise::<i16>(255),
    ];
    for prefix in ["bullet", "small", "g"] {
        v.push(SavedFormat::id(&format!("{prefix}_l1w")).round().quantise::<i16>(64));
        v.push(SavedFormat::id(&format!("{prefix}_l1b")).round().quantise::<i16>(64));
        v.push(SavedFormat::id(&format!("{prefix}_l2w")).round().quantise::<i16>(64));
        v.push(SavedFormat::id(&format!("{prefix}_l2b")).round().quantise::<i16>(64));
        v.push(SavedFormat::id(&format!("{prefix}_l3w")).round().quantise::<i16>(255 * 64));
        v.push(SavedFormat::id(&format!("{prefix}_l3b")).round().quantise::<i16>(255 * 64));
    }
    v
}

fn main() {
    let l1: usize = std::env::var("NAPK_L1").ok().and_then(|s| s.parse().ok()).unwrap_or(768);
    let threats_mode: u8 = match std::env::var("NAPK_THREATS").unwrap_or_default().as_str() {
        "none" => 0, "640" => 1, _ => 2,
    };
    let n_in: usize = 1 + match threats_mode { 0 => 22528, 1 => 23168, _ => 31744 };
    println!("🦅 napk9_train_v10_binpack — treino direto do binpack (sem .data2), scale={SCALE}, L1={l1}");
    println!("   NAPK_THREATS: {}", match threats_mode { 0 => "NONE", 1 => "640 clássico", _ => "FULL 9216 (V10)" });

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(NapkInputV10 { mode: threats_mode })
        .output_buckets(NapkV2MaterialBuckets::<8>)
        .save_format(&save_ids_all())
        .build_custom(move |builder, (stm, ntm, buckets), targets| {
            let acc_layer  = builder.new_affine("acc",  n_in, l1);
            let psqt_layer = builder.new_affine("psqt", n_in, 8);

            let bullet_l1 = builder.new_affine("bullet_l1", l1 * 2, 16);
            let bullet_l2 = builder.new_affine("bullet_l2", 16, 32);
            let bullet_l3 = builder.new_affine("bullet_l3", 32, 1 * 8);
            let small_l1 = builder.new_affine("small_l1", l1 * 2, 32);
            let small_l2 = builder.new_affine("small_l2", 32, 32);
            let small_l3 = builder.new_affine("small_l3", 32, 1 * 8);
            let big_l1 = builder.new_affine("g_l1", l1 * 2, BIG_L2);
            let big_l2 = builder.new_affine("g_l2", BIG_L2, BIG_L3);
            let big_l3 = builder.new_affine("g_l3", BIG_L3, 1 * 8);

            let acc_us = acc_layer.forward(stm).screlu();
            let acc_them = acc_layer.forward(ntm).screlu();
            let x = acc_us.concat(acc_them);

            let psqt_b = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);
            let psqt_s = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);
            let psqt_g = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);

            let b1 = bullet_l1.forward(x).screlu();
            let b2 = bullet_l2.forward(b1).screlu();
            let out_bullet = bullet_l3.forward(b2).select(buckets) + psqt_b;

            let s1 = small_l1.forward(x).screlu();
            let s2 = small_l2.forward(s1).screlu();
            let out_small = small_l3.forward(s2).select(buckets) + psqt_s;

            let g1 = big_l1.forward(x).screlu();
            let g2 = big_l2.forward(g1).screlu();
            let out_big = big_l3.forward(g2).select(buckets) + psqt_g;

            let loss = out_big.sigmoid().squared_error(targets)
                     + out_small.sigmoid().squared_error(targets)
                     + out_bullet.sigmoid().squared_error(targets);
            (out_big, loss)
        });

    // CODA: clipping mais apertado nas camadas esparsas de entrada — ver napk9_train_v10_coda.rs.
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("accw", stricter_clipping);
    trainer.optimiser.set_params_for_weight("psqtw", stricter_clipping);

    let tag = std::env::var("NAPK_TAG").unwrap_or_default();
    let net_id = if tag.is_empty() { format!("NAPKa0s_v10bp_{l1}") } else { format!("NAPKa0s_v10bp_{l1}_{tag}") };
    println!("🦅 NET_ID: {net_id}");

    let lr_start: f32 = std::env::var("NAPK_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let batch_size: usize = std::env::var("NAPK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let bps: usize = std::env::var("NAPK_BPS").ok().and_then(|s| s.parse().ok()).unwrap_or(6_104);
    // CODA v7_1024h16x32s: 800 superbatches (a config dele mais próxima da nossa L1=1024).
    let end_sb: usize = std::env::var("NAPK_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(800);
    let final_lr = lr_start / 100.0;
    // CODA v7_1024h16x32s usa wdl=0.0 (puro cp); o nosso histórico (napk9_train_v10.rs) usa
    // 0.3 (decisão documentada: resultado real ajuda com finais Syzygy). Default fica 0.3 —
    // põe NAPK_WDL=0 p/ replicar o Coda exatamente.
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

    // ── fonte: binpack(s) do Stockfish direto, sem .data2 ──
    let binpack_path = std::env::var("NAPK_BINPACK")
        .unwrap_or_else(|_| "/mnt/d/Nap2Siriux/training/sf_test80.binpack".to_string());
    let paths: Vec<&str> = binpack_path.split(':').filter(|s| !s.is_empty()).collect();
    // CODA v7_1024h16x32s: buffer 256MB, 4 threads.
    let buffer_mb: usize = std::env::var("NAPK_BINPACK_BUFFER_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let binpack_threads: usize = std::env::var("NAPK_BINPACK_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    // CODA: ply>=16. Histórico nosso (s29, ver bullet_integration/binpack_to_data.rs): com o
    // loader ANTIGO (sequencial, sem shuffle) isto deixava a rede sem aberturas; agora o
    // shuffle do NapkBinpackLoader já existe por construção (buffer cheio → Fisher-Yates),
    // por isso este filtro deixou de ter o efeito catastrófico que tinha — mas NUNCA foi
    // testado a sério (ver memória project_coda_training_alignment). Default 0 (mantém a
    // decisão já validada em produção); põe NAPK_MIN_PLY=16 p/ testar a versão do Coda.
    let min_ply: u16 = std::env::var("NAPK_MIN_PLY").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    println!("🦅 NAPK_BINPACK ({} ficheiro(s)), buffer={buffer_mb}MB/{binpack_threads}t, min_ply={min_ply}", paths.len());
    for p in &paths { println!("   - {p}"); }
    println!("   ⚠️ sem --dist/quota por bucket aqui (só no binpack_to_data.rs offline) — distribuição segue a do binpack tal como vem.");

    let filter = move |entry: &TrainingDataEntry| {
        let in_check = entry.pos.is_checked(entry.pos.side_to_move());
        let normal = entry.mv.mtype() == MoveType::Normal;
        let dest_empty = entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
        entry.ply >= min_ply && !in_check && entry.score.unsigned_abs() <= 10000 && normal && dest_empty
    };

    let data_loader = NapkBinpackLoader::new_concat_multiple(&paths, buffer_mb, binpack_threads, filter);
    trainer.run(&schedule, &settings, &data_loader);

    println!("🦅 FIM. checkpoints_napk/{}/", std::env::var("NAPK_TAG").unwrap_or_default());
}
