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

// 🦅 OPÇÃO B — treina UMA cabeça (sem ciclo). Escolhe a cabeça por env NAPK_HEAD:
//    NAPK_HEAD=bullet | small | big  (default big). Corre 3× e junta no auto_converter.
const L1_SIZE: usize = 128;
const SCALE: f32 = 408.0;

fn save_ids_for(head: &str) -> Vec<SavedFormat> {
    // ids dos pesos da cabeça escolhida + acc/psqt/chaos (sempre).
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
        SavedFormat::id("chaos_l1w").round().quantise::<i16>(64),
        SavedFormat::id("chaos_l1b").round().quantise::<i16>(64),
        SavedFormat::id("chaos_l2w").round().quantise::<i16>(64),
        SavedFormat::id("chaos_l2b").round().quantise::<i16>(64),
        SavedFormat::id("chaos_l3w").round().quantise::<i16>(255 * 64),
        SavedFormat::id("chaos_l3b").round().quantise::<i16>(255 * 64),
    ]
}

fn main() {
    let head: &'static str = match std::env::var("NAPK_HEAD").as_deref() {
        Ok("bullet") => "bullet",
        Ok("small") => "small",
        _ => "big",
    };
    println!("🦅 A treinar a cabeça: {}", head);

    let napk9_net = NapKa0sV9 { l1_size: L1_SIZE, head };

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(NapkInput23168)
        .output_buckets(NapkMaterialBuckets::<8>)
        .save_format(&save_ids_for(head))
        .build_custom(|builder, (stm, ntm, buckets), targets| {
            let (out_eval, out_chaos) = napk9_net.build_graph(builder, stm, ntm, buckets);
            let loss = out_eval.sigmoid().squared_error(targets)
                     + out_chaos.sigmoid().squared_error(targets) * 0.5;
            (out_eval, loss)
        });

    let net_id = format!("NAPKa0s_V9_{}", head);
    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1000,
            start_superbatch: 1,
            end_superbatch: 100,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 40 },
        save_rate: 100,
    };
    let settings = LocalSettings {
        threads: 4, test_set: None, output_directory: "checkpoints_napk", batch_queue_size: 64,
    };
    let data_loader = loader::DirectSequentialDataLoader::new(
        &["/mnt/d/Nap2Siriux/training/napk_unified.data"]);
    trainer.run(&schedule, &settings, &data_loader);
}
