// 🦅 napk9_train_v10_coda.rs — trainer V10, configuração alinhada ao processo do Coda
//    (github.com/adamtwiss/coda, referência conceptual — ver bullet_integration/refs/sykora_main_ref.rs).
//
//    Derivado do napk9_train_v10.rs (mesma arquitetura NapK9: acc/psqt partilhados + 3 cabeças
//    bullet/small/big estilo SFNNv13). O que muda em relação ao napk9_train_v10.rs (procura "CODA:"):
//      1. eval_scale: 408.0 → 400.0 (igual ao schedule do Coda)
//      2. clipping mais apertado em accw/psqtw (Coda faz o mesmo em l0w/l0f — as suas camadas
//         esparsas de entrada, equivalentes às nossas acc/psqt)
//      3. default de superbatches: 800 (em vez do auto-por-epochs=10) — é o que o Coda corre;
//         continua a aceitar NAPK_SB/NAPK_EPOCHS para escolher manualmente
//
//    NÃO migrámos para o SfBinpackLoader do Coda (lê o binpack a direito, sem passar por um
//    ficheiro intermédio): o NapkInputV10 tem RequiredDataType=NapkRecordV2 (o nosso .data2),
//    não TrainingDataEntry — para ler o binpack diretamente seria preciso escrever um loader
//    novo. Em vez disso, o shuffle (que o Coda obtém de um buffer interno do SfBinpackLoader)
//    fica a cargo do training/shuffle_data2.py — corre-se ANTES do treino sobre o .data2.
//
//    INSTALAÇÃO: igual ao napk9_train_v10.rs (cp napk9_v10.rs + napk9_v10_features.rs p/ o src/
//    do bullet, registar `mod napk9_v10;`, este ficheiro p/ examples/).
//    USO: NAPK_DATA=napk_sf_v2_shuffled.data2 NAPK_L1=256 NAPK_SB=800 \
//           cargo run --release -p bullet_lib --example napk9_train_v10_coda --features cuda
use bullet_lib::{
    nn::optimiser::{AdamW, AdamWParams},
    napk9_v10::{NapkInputV10, NapkV2MaterialBuckets},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const SCALE: f32 = 400.0;   // CODA: era 408.0 (OUTPUT_SCALE_CP do motor) — ⚠️ ver nota no fim.

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
    let l1: usize = std::env::var("NAPK_L1").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let threats_mode: u8 = match std::env::var("NAPK_THREATS").unwrap_or_default().as_str() {
        "none" => 0, "640" => 1, _ => 2,
    };
    let n_in: usize = 1 + match threats_mode { 0 => 22528, 1 => 23168, _ => 31744 };
    println!("🦅 NAPK_THREATS: {}", match threats_mode { 0 => "NONE", 1 => "640 clássico", _ => "FULL 9216 (V10)" });
    println!("🦅 napk9_train_v10_coda — config alinhada ao Coda (scale={SCALE}), estrutura NapK9, L1={l1}");

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

    // CODA: clipping mais apertado nas camadas esparsas de entrada (acc/psqt), tal como o
    // Coda aperta l0w (peso principal) e l0f (factoriser) — mitiga blowup ao longo de
    // centenas de superbatches. As nossas acc/psqt fazem o mesmo papel (entrada esparsa →
    // densa, partilhada pelas 3 cabeças), por isso ganham a mesma proteção.
    let stricter_clipping = AdamWParams {
        max_weight: 0.99,
        min_weight: -0.99,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("accw", stricter_clipping);
    trainer.optimiser.set_params_for_weight("psqtw", stricter_clipping);

    let tag = std::env::var("NAPK_TAG").unwrap_or_default();
    let net_id = if tag.is_empty() {
        format!("NAPKa0s_v10coda_{l1}")
    } else {
        format!("NAPKa0s_v10coda_{l1}_{tag}")
    };
    println!("🦅 NET_ID: {net_id}  (checkpoint → checkpoints_napk/{net_id}-<sb>/)");
    let lr_start: f32 = std::env::var("NAPK_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let batch_size: usize = std::env::var("NAPK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let bps: usize = std::env::var("NAPK_BPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);

    let data_path = std::env::var("NAPK_DATA")
        .unwrap_or_else(|_| "/mnt/d/Nap2Siriux/training/napk_sf_v2_shuffled.data2".to_string());
    let paths: Vec<&str> = data_path.split(':').filter(|s| !s.is_empty()).collect();

    const NAPK_RECORD: u64 = 112;
    let total_bytes: u64 = paths.iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    let n_positions = total_bytes / NAPK_RECORD;

    // CODA: o Coda corre ~800 superbatches fixos (não um nº de épocas alvo). Mantemos o
    // cálculo automático por épocas como rede de segurança contra overfit (ver
    // napk9_train_v10.rs), mas o DEFAULT desta variante é 800 superbatches diretos —
    // NAPK_SB continua a sobrepor-se a tudo, NAPK_EPOCHS continua disponível se preferires
    // pensar em épocas em vez de superbatches.
    let pos_per_sb = (batch_size * bps) as u64;
    let end_sb: usize = std::env::var("NAPK_SB").ok().and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            if let Ok(ep) = std::env::var("NAPK_EPOCHS") {
                let target_epochs: f64 = ep.parse().unwrap_or(10.0);
                (((n_positions as f64) * target_epochs) / (pos_per_sb as f64)).ceil().max(1.0) as usize
            } else {
                800   // CODA: default direto, igual ao processo de referência
            }
        });

    let real_epochs = (end_sb as f64 * pos_per_sb as f64) / (n_positions.max(1) as f64);
    println!("🦅 DADOS ({} ficheiro(s)), {} posições ({:.2} GB):", paths.len(), n_positions, total_bytes as f64/1e9);
    for p in &paths { println!("   - {p}"); }
    println!("🦅 SUPERBATCHES: {end_sb} (≈{real_epochs:.1} épocas; {pos_per_sb} pos/sb)");
    if real_epochs > 20.0 {
        println!("   ⚠️ {real_epochs:.0} épocas sobre ESTE dataset é muito — confirma que NAPK_DATA \
                   aponta para um corpus grande (binpack SF), não para o selfplay pequeno.");
    }

    let final_lr = lr_start / 100.0;
    let wdl_weight: f32 = std::env::var("NAPK_WDL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let save_rate_env: usize = std::env::var("NAPK_SAVERATE").ok().and_then(|s| s.parse().ok())
        .filter(|&r: &usize| r > 0).unwrap_or(10).min(end_sb);
    println!("   wdl_weight = {wdl_weight} | save_rate = {save_rate_env}");

    let schedule = TrainingSchedule {
        net_id, eval_scale: SCALE,
        steps: TrainingSteps { batch_size, batches_per_superbatch: bps,
            start_superbatch: 1, end_superbatch: end_sb },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_weight },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: lr_start,
            final_lr,
            final_superbatch: end_sb,
        },
        save_rate: save_rate_env,
    };
    let settings = LocalSettings {
        threads: 10, test_set: None, output_directory: "checkpoints_napk", batch_queue_size: 64,
    };

    if let Ok(resume_dir) = std::env::var("NAPK_RESUME") {
        if !resume_dir.is_empty() { trainer.load_from_checkpoint(&resume_dir); }
    }

    println!("🦅 Dados pré-baralhados? Confirma que NAPK_DATA passou por training/shuffle_data2.py \
              — este loader (DirectSequentialDataLoader) NÃO baralha (lê o ficheiro em ordem).");

    let data_loader = loader::DirectSequentialDataLoader::new(&paths);
    trainer.run(&schedule, &settings, &data_loader);

    println!("🦅 FIM. checkpoints_napk/{}/", net_id);
}

// ⚠️ NOTA — scale 400 vs OUTPUT_SCALE_CP do motor (408, ver docs/CLAUDE.md e src/search.cpp):
//   esta rede vai sair calibrada a 400, não a 408. Antes de embeber e jogar com ela, ou:
//   (a) atualiza OUTPUT_SCALE_CP=400 no motor e as margens "× 408/400" em search.cpp passam
//       a "× 400/400" (i.e. removem-se), ou
//   (b) re-escala os pesos de saída ×408/400 na conversão p/ .napk9.
//   Sem isto, os scores em cp ficam ~2% desviados (não catastrófico, mas errado) — não
//   prometer ao motor um scale que a rede não tem.
