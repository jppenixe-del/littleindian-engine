use bullet_lib::{
    nn::optimiser::AdamW,
    napk9::{NapKa0sV9, NapkInput23168, NapkMaterialBuckets},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

// 🦅 TREINO sem chaos — 1 saída (eval). Resolve o "Cycle found". Cabeça por env NAPK_HEAD.
const L1_SIZE: usize = 128;
const SCALE: f32 = 408.0;

fn save_ids_for(head: &str) -> Vec<SavedFormat> {
    let prefix = match head { "bullet" => "bullet", "small" => "small", _ => "g" };
    vec![
        SavedFormat::id("accw").round().quantise::<i16>(255),
        SavedFormat::id("accb").round().quantise::<i16>(255),
        SavedFormat::id("psqtw").round().quantise::<i16>(255),
        SavedFormat::id("psqtb").round().quantise::<i16>(255),
        SavedFormat::id(&format!("{}_l1w", prefix)).round().quantise::<i16>(64),
        SavedFormat::id(&format!("{}_l1b", prefix)).round().quantise::<i16>(64),
        SavedFormat::id(&format!("{}_l2w", prefix)).round().quantise::<i16>(64),
        SavedFormat::id(&format!("{}_l2b", prefix)).round().quantise::<i16>(64),
        SavedFormat::id(&format!("{}_l3w", prefix)).round().quantise::<i16>(255 * 64),
        SavedFormat::id(&format!("{}_l3b", prefix)).round().quantise::<i16>(255 * 64),
    ]
}

fn main() {
    let head: &'static str = match std::env::var("NAPK_HEAD").as_deref() {
        Ok("bullet") => "bullet",
        Ok("small") => "small",
        _ => "big",
    };
    println!("🦅 A treinar a cabeça: {} | L1={} (SEM chaos → sem ciclo)", head, L1_SIZE);

    let napk9_net = NapKa0sV9 { l1_size: L1_SIZE, head };

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(NapkInput23168)
        .output_buckets(NapkMaterialBuckets::<8>)
        .save_format(&save_ids_for(head))
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            // 🦅 1 SAÍDA só (sem chaos) → o otimizador não cria ciclo.
            let out_eval = napk9_net.build_graph(builder, stm, ntm, buckets);
            let loss = out_eval.sigmoid().squared_error(targets);
            (out_eval, loss)
        });

    let net_id = format!("NAPKa0s_V9_{}", head);
    // 🦅 LR e superbatches controláveis por env (p/ a FASE 2 / fine-tuning usar LR mais baixo).
    //   Defeitos = treino normal do zero (fase 1). Fine-tuning: NAPK_FT_LR=0.0001 NAPK_SB=60.
    let ft_lr: f32 = std::env::var("NAPK_FT_LR").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let end_sb: usize = std::env::var("NAPK_SB").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(100);
    // 🦅 batch_size: a 5060 Ti come 65k em ~10s → 65536 por defeito (4× o antigo 16384).
    //   batches_per_superbatch ajustado p/ 250 → superbatch = 65536×250 = 16.384.000 (IGUAL ao
    //   antigo 16384×1000), mantendo a granularidade dos checkpoints. Ambos override por env.
    let batch_size: usize = std::env::var("NAPK_BATCH").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(65_536);
    let bps: usize = std::env::var("NAPK_BPS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(250);
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
        lr_scheduler: lr::StepLR { start: ft_lr, gamma: 0.1, step: 40 },
        save_rate: 100,
    };
    let settings = LocalSettings {
        // 🦅 threads = workers de CARREGAMENTO de dados (CPU→GPU), NÃO afeta o cálculo na GPU.
        //   Mais threads = alimenta a GPU mais depressa (menos tempo à espera de dados). 10 é
        //   bom p/ a 5060 Ti; se o teu CPU tiver menos de 10 núcleos, baixa.
        threads: 10, test_set: None, output_directory: "checkpoints_napk", batch_queue_size: 64,
    };
    // 🦅 WARM START / FINE-TUNING (a técnica do SF: não treinar do zero, partir de uma rede já
    //   boa). Se a env NAPK_RESUME apontar para uma pasta de checkpoint, carrega esses pesos
    //   ANTES de treinar → a rede NOVA herda o que a anterior aprendeu, e os dados desta fase
    //   só AFINAM (em vez de re-descobrir tudo do zero).
    //   ⚠️ Para fine-tuning a sério, baixa o LR desta fase (NAPK_FT_LR) — senão "esquece".
    if let Ok(resume_dir) = std::env::var("NAPK_RESUME") {
        if !resume_dir.is_empty() {
            println!("🦅 WARM START: a carregar pesos de '{}' (fine-tuning, não do zero)", resume_dir);
            trainer.load_from_checkpoint(&resume_dir);
            println!("   ✅ pesos carregados — o treino parte desta rede, não de zero.");
        }
    }

    // 🦅 FONTE DOS DADOS — o NapkInput23168 lê os features (peças + 640 threats) de us_feats/
    //   them_feats do NapkRecord, que SÓ o teu .data preenche (build-threats → merge_to_bullet).
    //   Por isso NÃO se usa o SfBinpackLoader aqui (entrega TrainingDataEntry, não NapkRecord —
    //   tipos incompatíveis). Para usar dados do Stockfish (binpack), converte-os ANTES para o
    //   teu .plain → .bin → build-threats → .data (ver binpack_to_plain.py). Assim o binpack
    //   passa por todo o teu pipeline e os threats são calculados como sempre.
    let data_path = std::env::var("NAPK_DATA")
        .unwrap_or_else(|_| "/mnt/d/Nap2Siriux/training/napk_unified.data".to_string());
    // 🦅 NAPK_DATA aceita VÁRIOS ficheiros separados por ':' (estilo PATH) — assim treina lendo
    //   os dois (ex: finais.data:sf_mid.data) SEM precisar de cat/juntar 45GB. O loader do bullet
    //   já aceita um array de caminhos (DirectSequentialDataLoader::new(&[...])).
    let paths: Vec<&str> = data_path.split(':').filter(|s| !s.is_empty()).collect();
    println!("🦅 DADOS ({} ficheiro(s)):", paths.len());
    for p in &paths { println!("   - {p}"); }
    let data_loader = loader::DirectSequentialDataLoader::new(&paths);
    trainer.run(&schedule, &settings, &data_loader);
}
