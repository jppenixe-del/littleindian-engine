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
// 🦅 trainsiriux_rust.rs — UMA rede única coerente, treino bullet (Rust, GPU, ~280k pos/seg).
//
// PORQUÊ: o "monstro" treinava 3 cabeças em 3 PROCESSOS separados → cada uma via os dados noutra
//   ordem/momento → divergiam → perdia até ao 256 do trainsiriux. E o trainvsiriux (PyTorch) é
//   COERENTE mas LENTO (dataloader Python esfomeia a GPU). Este: o MELHOR dos dois — UMA rede
//   coerente (1 treino, 1 passagem, sem heads) E a velocidade do bullet (dataloader Rust nativo).
//
// AUTOSSUFICIENTE: define a rede AQUI (acc -> l1 -> l2 -> 1x8 + psqt), nao depende de NapKa0sV9 nem
//   do build_graph (que entre versoes mudou de assinatura). So usa NapkInput23168 +
//   NapkMaterialBuckets (tipos estaveis) e o ValueTrainerBuilder. Compila por si.
//
//   USO: NAPK_DATA="finais.data:sf_mid.data" NAPK_L1=256 \
//          cargo run --release -p bullet_lib --example trainsiriux_rust --features cuda
//   L1 por env (default 256). Salva NAPK9LEB (ids "g_*" = a big que o motor le).
// ════════════════════════════════════════════════════════════════════════════
const SCALE: f32 = 408.0;

fn detect_big_l2(l1: usize) -> usize {
    if l1 <= 128 { 16 }
    else if l1 <= 512 { 32 }
    else if l1 <= 768 { 48 }
    else if l1 <= 1024 { 64 }
    else { 96 }
}

fn save_ids() -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("accw").round().quantise::<i16>(255),
        SavedFormat::id("accb").round().quantise::<i16>(255),
        SavedFormat::id("psqtw").round().quantise::<i16>(255),
        SavedFormat::id("psqtb").round().quantise::<i16>(255),
        SavedFormat::id("g_l1w").round().quantise::<i16>(64),
        SavedFormat::id("g_l1b").round().quantise::<i16>(64),
        SavedFormat::id("g_l2w").round().quantise::<i16>(64),
        SavedFormat::id("g_l2b").round().quantise::<i16>(64),
        SavedFormat::id("g_l3w").round().quantise::<i16>(255 * 64),
        SavedFormat::id("g_l3b").round().quantise::<i16>(255 * 64),
    ]
}

fn main() {
    let l1: usize = std::env::var("NAPK_L1").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let big_l2 = detect_big_l2(l1);

    println!("🦅 trainsiriux_rust — UMA rede coerente, L1={l1} (bigL2={big_l2}), treino bullet rapido");

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(NapkInput23168)
        .output_buckets(NapkMaterialBuckets::<8>)
        .save_format(&save_ids())
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // 🦅 REDE UNICA inline (= arquitetura do trainsiriux: acc -> g_l1 -> g_l2 -> g_l3 + psqt)
            let acc_layer  = builder.new_affine("acc",  23169, l1);
            let psqt_layer = builder.new_affine("psqt", 23169, 8);
            let g_l1 = builder.new_affine("g_l1", l1 * 2, l1);
            let g_l2 = builder.new_affine("g_l2", l1, big_l2);
            let g_l3 = builder.new_affine("g_l3", big_l2, 1 * 8);

            let acc_us   = acc_layer.forward(stm).screlu();
            let acc_them = acc_layer.forward(ntm).screlu();
            let x = acc_us.concat(acc_them);

            let psqt_bias = psqt_layer.forward(stm).select(buckets)
                          - psqt_layer.forward(ntm).select(buckets);

            let h1 = g_l1.forward(x).screlu();
            let h2 = g_l2.forward(h1).screlu();
            let out = g_l3.forward(h2).select(buckets) + psqt_bias;

            let loss = out.sigmoid().squared_error(targets);
            (out, loss)
        });

    let net_id = format!("siriux_rust_{l1}");
    let lr_start: f32 = std::env::var("NAPK_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let end_sb: usize = std::env::var("NAPK_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let batch_size: usize = std::env::var("NAPK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16_384);
    let bps: usize = std::env::var("NAPK_BPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: bps,
            start_superbatch: 1,
            end_superbatch: end_sb,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.1, step: 40 },
        save_rate: 100,
    };
    let settings = LocalSettings {
        threads: 10, test_set: None, output_directory: "checkpoints_napk", batch_queue_size: 64,
    };

    if let Ok(resume_dir) = std::env::var("NAPK_RESUME") {
        if !resume_dir.is_empty() {
            println!("🦅 WARM START: pesos de '{resume_dir}' (fine-tuning)");
            trainer.load_from_checkpoint(&resume_dir);
        }
    }

    let data_path = std::env::var("NAPK_DATA")
        .unwrap_or_else(|_| "/mnt/d/Nap2Siriux/training/napk_unified.data".to_string());
    let paths: Vec<&str> = data_path.split(':').filter(|s| !s.is_empty()).collect();
    println!("🦅 DADOS ({} ficheiro(s)):", paths.len());
    for p in &paths { println!("   - {p}"); }
    let data_loader = loader::DirectSequentialDataLoader::new(&paths);
    trainer.run(&schedule, &settings, &data_loader);

    println!("🦅 FIM. Rede unica em checkpoints_napk/siriux_rust_{l1}/. Converter com auto_converter.");
}
