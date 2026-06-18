// 🦅 napk9_v10_binpack_loader.rs — loader que lê o binpack do Stockfish a direito (sem
//    passar por binpack_to_data.rs / .data2 nenhum) e entrega NapkRecordV2 já filtrado e
//    baralhado ao trainer. Os THREATS continuam on-the-fly no NapkInputV10::map_features
//    (napk9_v10.rs) — isto só substitui a ORIGEM dos dados, nada mais muda.
//
//    Estrutura COPIADA da SfBinpackLoader real do bullet (crates/bullet_lib/src/value/
//    loader/sfbinpack.rs, jw1912/bullet @ main, confirmada via GitHub a 2026-06-18) — mesmo
//    pipeline de 3 estágios (reader → conversor paralelo → shuffle), só troca:
//      ChessBoard → NapkRecordV2, convert_to_bulletformat → convert_to_napk.
//    Isto significa: MESMO mecanismo de shuffle que o "Coda" (buffer cheio → Fisher-Yates →
//    emite o buffer inteiro como 1 chunk), e o mesmo loop infinito sobre os file_paths (não
//    para no fim do binpack — relê do início; suporta tantos superbatches quantos quiseres
//    sem se preocupar com o tamanho do dataset).
//
//    ⚠️ NÃO suporta --dist (quota por material bucket) — isso só existe no binpack_to_data.rs
//    offline. Se precisares de balancear buckets, usa esse caminho (já tem shuffle embutido,
//    --shuffle-mb). Este loader é para quando NÃO precisas de balancear, só de treinar direto.
//
//    INSTALAÇÃO: copiar para o src/ do bullet (lado a lado do napk9_v10.rs), registar no lib.rs
//    com `pub mod napk9_v10_binpack_loader;` (TEM de ser pub — os examples/ são crates
//    separadas que importam isto como `bullet_lib::napk9_v10_binpack_loader::NapkBinpackLoader`,
//    não `crate::...`; é o mesmo padrão do `pub mod napk9_v10;` já usado pelo NapkInputV10).
//
//    USO no trainer: NapkBinpackLoader::new_concat_multiple(&paths, buffer_mb, threads, filter)
//    onde filter: impl Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static.
//
//    ⚠️ NOTA DE VERSÃO (confirmado lendo o checkout real em /mnt/c/data/env/bullet a 2026-06-18):
//    o `main` do GitHub jw1912/bullet usa `bullet_trainer::reader::DataReader::read_chunks`;
//    ESTE checkout local (mais antigo/diferente) usa `crate::value::loader::DataLoader` com
//    `data_file_paths`/`count_positions`/`map_chunks` — mesma lógica interna, trait diferente.
//    Implementado para a versão LOCAL (a que compila aqui), não para o `main` do GitHub.

use std::{fs::File, sync::mpsc, thread};

use sfbinpack::CompressedTrainingDataEntryReader;
pub use sfbinpack::{
    TrainingDataEntry,
    chess::{color::Color, piecetype::PieceType},
};

use crate::napk9_v10::NapkRecordV2;
use crate::value::loader::DataLoader;

// ── PRNG sem dependências externas (mesmo xorshift64 que o bullet usa internamente em
//    value/loader/rng.rs — não dá para importar de lá, é módulo privado, por isso copia-se). ──
struct SimpleRand(u64);
impl SimpleRand {
    fn with_seed() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).expect("Guaranteed increasing.").as_micros() as u64
            & 0xFFFF_FFFF;
        Self(seed | 1)
    }
    fn rng(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
fn shuffle(data: &mut [NapkRecordV2]) {
    let mut rng = SimpleRand::with_seed();
    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}

/// entry.score e entry.result vêm do binpack já em POV side-to-move (confirmado: a
/// SfBinpackLoader real só os passa a POV-branco para alimentar o ChessBoard, que é
/// absoluto; o NapkRecordV2 quer POV-stm, tal como o NapkRecord clássico — por isso aqui
/// NÃO se inverte nada, ao contrário do convert_to_bulletformat do bullet).
fn convert_to_napk(entry: &TrainingDataEntry) -> NapkRecordV2 {
    let mut bb = [[0u64; 6]; 2];
    let pc_bb = |color: Color, pt: PieceType| entry.pos.pieces_bb_color(color, pt).bits();
    for (c_idx, color) in [Color::White, Color::Black].into_iter().enumerate() {
        bb[c_idx][0] = pc_bb(color, PieceType::Pawn);
        bb[c_idx][1] = pc_bb(color, PieceType::Knight);
        bb[c_idx][2] = pc_bb(color, PieceType::Bishop);
        bb[c_idx][3] = pc_bb(color, PieceType::Rook);
        bb[c_idx][4] = pc_bb(color, PieceType::Queen);
        bb[c_idx][5] = pc_bb(color, PieceType::King);
    }
    let npieces: i32 = bb.iter().flatten().map(|b| b.count_ones() as i32).sum();
    let mb = (((npieces - 1) / 4).clamp(0, 7)) as u8;
    let stm: u8 = if entry.pos.side_to_move() == Color::White { 0 } else { 1 };

    let cp_stm = (entry.score as i32).clamp(-2500, 2500) as f32;
    // wdl: usa o RESULTADO REAL do jogo (POV-stm, sem inverter) quando o binpack o tem;
    // cai p/ sigmoid(cp/400) só quando vem a 0 — caso dos binpacks "min-v2.v6" (ver nota em
    // binpack_to_plain.rs). O --v2 do binpack_to_data.rs NUNCA olhava p/ entry.result; aqui
    // corrige-se isso.
    let wdl_stm = match entry.result {
        r if r > 0 => 1.0f32,
        r if r < 0 => 0.0f32,
        _ => (1.0_f64 / (1.0 + (-(cp_stm as f64) / 400.0).exp())) as f32,
    };

    NapkRecordV2 { bb, stm, bucket: mb, pad: [0; 2], score: cp_stm, wdl: wdl_stm, pad2: [0; 4] }
}

#[derive(Clone)]
pub struct NapkBinpackLoader<T: Fn(&TrainingDataEntry) -> bool> {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: T,
}

impl<T: Fn(&TrainingDataEntry) -> bool> NapkBinpackLoader<T> {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(paths: &[&str], buffer_size_mb: usize, threads: usize, filter: T) -> Self {
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<NapkRecordV2>() / 2,
            threads,
            filter,
        }
    }
}

impl<T> DataLoader<NapkRecordV2> for NapkBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[NapkRecordV2]) -> bool>(&self, _: usize, mut f: F) {
        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();

        let reader_buffer_size = 16384 * threads;
        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<TrainingDataEntry>>(4);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut buffer = Vec::with_capacity(reader_buffer_size);

            'dataloading: loop {
                for file in &file_paths {
                    let file = File::open(file).unwrap();
                    let mut reader = CompressedTrainingDataEntryReader::new(file).unwrap();

                    while reader.has_next() {
                        buffer.push(reader.next());

                        if buffer.len() == reader_buffer_size || !reader.has_next() {
                            if reader_msg_receiver.try_recv().unwrap_or(false) || reader_sender.send(buffer).is_err() {
                                break 'dataloading;
                            }

                            buffer = Vec::with_capacity(reader_buffer_size);
                        }
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) = mpsc::sync_channel::<Vec<NapkRecordV2>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok(unfiltered) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                thread::scope(|s| {
                    let chunk_size = unfiltered.len().div_ceil(threads).max(1);
                    let mut handles = Vec::new();

                    for chunk in unfiltered.chunks(chunk_size) {
                        let this_sender = converted_sender.clone();
                        let handle = s.spawn(move || {
                            let mut buffer = Vec::with_capacity(chunk_size);

                            for entry in chunk {
                                if filter(entry) {
                                    buffer.push(convert_to_napk(entry));
                                }
                            }

                            this_sender.send(buffer).is_err()
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        if handle.join().unwrap() {
                            should_break = true;
                        }
                    }
                });

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<NapkRecordV2>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);

            'dataloading: while let Ok(converted) = converted_receiver.recv() {
                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }
    }
}
