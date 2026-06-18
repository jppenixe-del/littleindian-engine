use bullet_lib::{
    nn::optimiser::AdamW,
    napk9::{NapkInput23168, NapkMaterialBuckets},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

// ════════════════════════════════════════════════════════════════════════════
// 🦅 napk9_train_juntas.rs — as 3 cabecas (bullet/small/big) na MESMA passagem, MESMOS dados.
//   = ideia do Stockfish (big+small na mesma rede, treino conjunto, coerentes).
//
// AUTOSSUFICIENTE: constroi as 3 cabecas INLINE (nao usa build_graph, que no napk9.rs atual so
//   devolve 1 cabeca). acc/psqt UMA vez (partilhados) + bullet/small/big a partir do MESMO x.
//   loss = loss(big)+loss(small)+loss(bullet). ids batem com o save → auto_converter_3files junta.
//   psqt forward SEPARADO por cabeca (anti "Cycle found").
//
//   USO: NAPK_DATA="finais.data:sf_mid.data" NAPK_L1=256 \
//          cargo run --release -p bullet_lib --example napk9_train_juntas --features cuda
// ════════════════════════════════════════════════════════════════════════════
const SCALE: f32 = 408.0;

// 🦅 SFNNv13 (commit a6d055d): a big NÃO tem camada gorda L1→L1. Vai do acc GRANDE (L1×2) p/ uma
//   densa PEQUENA fixa, DIRETO. O SF usa L2Big=31, L3Big=32; usamos 32/32 (redondo p/ AVX2/512).
//   Isto mata o gargalo: a 1ª densa passa de 2048→1024 (2M pesos) p/ 2048→32 (65k). ~30× menos.
//   (a inteligência está no acumulador esparso + threats; as densas são só compressor SIMD.)
const BIG_L2: usize = 32;   // 1ª densa da big: L1×2 → 32  (era L1→L1 gordo). = L2Big do SFNNv13.
const BIG_L3: usize = 32;   // 2ª densa: 32 → 32. = L3Big do SFNNv13.

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
    println!("🦅 napk9_train_juntas — 3 cabecas JUNTAS (1 treino, mesmos dados), L1={l1}");
    println!("   big estilo SFNNv13: acc({l1}) → {BIG_L2} → {BIG_L3} → 1×8 (afunil, SEM camada L1→L1 gorda)");
    println!("   estilo Stockfish: acc partilhado + bullet/small/big, loss combinada, coerentes");

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(NapkInput23168)
        .output_buckets(NapkMaterialBuckets::<8>)
        .save_format(&save_ids_all())
        .build_custom(move |builder, (stm, ntm, buckets), targets| {
            // ── acc + psqt UMA vez (partilhados pelas 3 cabecas) ──
            let acc_layer  = builder.new_affine("acc",  23169, l1);
            let psqt_layer = builder.new_affine("psqt", 23169, 8);

            // ── as 3 cabecas (mesmas dimensoes do napk9.rs: bullet 16→32, small 32→32, big L1→bigL2) ──
            let bullet_l1 = builder.new_affine("bullet_l1", l1 * 2, 16);
            let bullet_l2 = builder.new_affine("bullet_l2", 16, 32);
            let bullet_l3 = builder.new_affine("bullet_l3", 32, 1 * 8);
            let small_l1 = builder.new_affine("small_l1", l1 * 2, 32);
            let small_l2 = builder.new_affine("small_l2", 32, 32);
            let small_l3 = builder.new_affine("small_l3", 32, 1 * 8);
            let big_l1 = builder.new_affine("g_l1", l1 * 2, BIG_L2);   // 🦅 SFNNv13: acc GRANDE → 32 DIRETO (afunil)
            let big_l2 = builder.new_affine("g_l2", BIG_L2, BIG_L3);   //   32 → 32  (era l1→bigL2 gordo)
            let big_l3 = builder.new_affine("g_l3", BIG_L3, 1 * 8);    //   32 → 1×8 buckets

            // ── forward do accumulator (partilhado) ──
            let acc_us = acc_layer.forward(stm).screlu();
            let acc_them = acc_layer.forward(ntm).screlu();
            let x = acc_us.concat(acc_them);

            // psqt forward SEPARADO por cabeca (anti Cycle found)
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

            // 🦅 LOSS COMBINADA: as 3 cabecas treinam contra o MESMO alvo, na MESMA passagem.
            let loss = out_big.sigmoid().squared_error(targets)
                     + out_small.sigmoid().squared_error(targets)
                     + out_bullet.sigmoid().squared_error(targets);
            (out_big, loss)
        });

    // 🦅 TAG por dataset (Maréchal): o net_id incluía só o L1 → 128/20M e 128/160M colidiam
    //   (ambos NAPKa0s_juntas_128-N) → trocas de rede nos testes. NAPK_TAG distingue-os.
    //   Ex: NAPK_TAG=20M → NAPKa0s_juntas_128_20M ; NAPK_TAG=160M → NAPKa0s_juntas_128_160M.
    let tag = std::env::var("NAPK_TAG").unwrap_or_default();
    let net_id = if tag.is_empty() {
        format!("NAPKa0s_juntas_{l1}")
    } else {
        format!("NAPKa0s_juntas_{l1}_{tag}")
    };
    println!("🦅 NET_ID: {net_id}  (checkpoint → checkpoints_napk/{net_id}-<sb>/)");
    let lr_start: f32 = std::env::var("NAPK_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let batch_size: usize = std::env::var("NAPK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let bps: usize = std::env::var("NAPK_BPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);

    // ── dados primeiro (preciso do tamanho p/ calcular os superbatches) ──
    let data_path = std::env::var("NAPK_DATA")
        .unwrap_or_else(|_| "/mnt/d/Nap2Siriux/training/napk_unified.data".to_string());
    let paths: Vec<&str> = data_path.split(':').filter(|s| !s.is_empty()).collect();

    // 🦅 nº de posições = soma dos tamanhos / 268 (NapkRecord). p/ calcular épocas/superbatches.
    const NAPK_RECORD: u64 = 268;
    let total_bytes: u64 = paths.iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    let n_positions = total_bytes / NAPK_RECORD;

    // 🦅 SUPERBATCHES AUTOMÁTICOS em função da AMOSTRA (Maréchal): em vez de NAPK_SB fixo (que com
    //   datasets pequenos dava 68 épocas → overfit), calcula os sb p/ atingir NAPK_EPOCHS épocas
    //   (default 10, saudável). end_sb = (n_pos × épocas) / (batch × bps). NAPK_SB força manual.
    let target_epochs: f64 = std::env::var("NAPK_EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let pos_per_sb = (batch_size * bps) as u64;
    let auto_sb = (((n_positions as f64) * target_epochs) / (pos_per_sb as f64)).ceil() as usize;
    let auto_sb = auto_sb.max(1);
    let end_sb: usize = std::env::var("NAPK_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(auto_sb);

    let real_epochs = (end_sb as f64 * pos_per_sb as f64) / (n_positions.max(1) as f64);
    println!("🦅 DADOS ({} ficheiro(s)), {} posições ({:.2} GB):", paths.len(), n_positions, total_bytes as f64/1e9);
    for p in &paths { println!("   - {p}"); }
    println!("🦅 SUPERBATCHES: {} (alvo {:.0} épocas → vê cada posição ~{:.1}× ; {} pos/sb)",
             end_sb, target_epochs, real_epochs, pos_per_sb);
    if real_epochs > 20.0 {
        println!("   ⚠️ {:.0} épocas é MUITO (risco de overfit). Baixa NAPK_SB ou NAPK_EPOCHS.", real_epochs);
    }

    // 🦅 LR: CosineDecayLR (como o sykora 3240 CCRL e o Stockfish) — decai SUAVE de lr_start até
    //   final_lr ao longo de TODO o treino, em vez do StepLR (degraus abruptos). O decaimento suave
    //   treina melhor no fim (menos "saltos" na otimização). final_lr = lr_start/100 (standard).
    let final_lr = lr_start / 100.0;

    // 🦅 wdl_weight: peso do RESULTADO do jogo no target (resto = sigmoid(cp/scale)).
    //   ERA 0.75 (BUG: 75% resultado → com finais Syzygy decididos, a rede aprendia "tudo é
    //   vitória" → scores inflados, startpos dava ~0.95 prob em vez de ~0.5). Baixado p/ 0.3:
    //   30% resultado + 70% cp fino → scores calibrados. Ajustável por NAPK_WDL p/ experimentar.
    let wdl_weight: f32 = std::env::var("NAPK_WDL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    println!("   wdl_weight = {wdl_weight} (era 0.75; baixo = confia no cp, alto = no resultado)");

    // 🦅 save_rate: a cada quantos superbatches salva um checkpoint. ERA end_sb.min(100) (= só no
    //   FIM, 1 checkpoint). Para o GATING (gate_superbatches.py escolhe o superbatch + forte por
    //   battle), precisamos de checkpoints INTERMÉDIOS. NAPK_SAVERATE (default 10) → salva a cada 10
    //   superbatches (NAPKa0s_juntas_<L1>_<tag>-10, -20, ...). Põe um nº grande p/ só no fim.
    let save_rate_env: usize = std::env::var("NAPK_SAVERATE").ok().and_then(|s| s.parse().ok())
        .filter(|&r: &usize| r > 0).unwrap_or(10).min(end_sb);
    println!("   save_rate = {save_rate_env} (checkpoints intermédios p/ o gating)");

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
        save_rate: save_rate_env,  // 🦅 intermédios p/ gating (NAPK_SAVERATE, default 10)
    };
    let settings = LocalSettings {
        threads: 10, test_set: None, output_directory: "checkpoints_napk", batch_queue_size: 64,
    };

    if let Ok(resume_dir) = std::env::var("NAPK_RESUME") {
        if !resume_dir.is_empty() { trainer.load_from_checkpoint(&resume_dir); }
    }

    let data_loader = loader::DirectSequentialDataLoader::new(&paths);
    trainer.run(&schedule, &settings, &data_loader);

    println!("🦅 FIM. 3 cabecas em checkpoints_napk/NAPKa0s_juntas_{l1}/ (1 checkpoint, coerente).");
}
