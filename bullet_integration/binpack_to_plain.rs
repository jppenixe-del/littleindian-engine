// binpack_to_plain.rs — Napoleon / Nap2Siriux
// =============================================
// 🦅 Converte um binpack do Stockfish → o teu formato .plain (fen | cp_white | wdl_white),
//    para entrar no teu pipeline NORMAL (prepare-plain → build-threats → merge → .data).
//
// PORQUÊ converter em vez de ler o binpack direto no treino:
//   O teu NapkInput23168 LÊ os features (peças + 640 threats) de us_feats/them_feats do
//   NapkRecord — ele NÃO calcula nada do board. O SfBinpackLoader entrega TrainingDataEntry
//   (fen/score/result), não NapkRecord → incompatível. Por isso o binpack tem de passar pelo
//   teu pipeline (que calcula os threats no build-threats). Este conversor é o 1º passo.
//
// Usa o crate sfbinpack 0.6.0 que JÁ tens no bullet (mesmo do examples/simple.rs).
//
// COMPILAR: põe este ficheiro em /mnt/c/data/env/bullet/examples/binpack_to_plain.rs
//   e corre (do diretório do bullet):
//     cargo run --release -p bullet_lib --example binpack_to_plain -- \
//       /caminho/test80.binpack /mnt/d/Nap2Siriux/training/sf_t80.plain
//
// SAÍDA: linhas "fen | cp_white | wdl_white" (cp POV brancas, wdl sigmoide POV brancas).
//   Aplica o filtro SF (ply>=16, sem xeque, |score|<=10000, lance normal, destino vazio).

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

// ⚠️ Mesma API do teu examples/simple.rs (sfbinpack 0.6.0).
use bullet_lib::value::loader::sfbinpack::{
    MoveType, PieceType,
    // O reader de baixo nível. Se o nome diferir na tua 0.6.0, ver nota no fim.
};
// Color (enum branca/preta) e o reader vêm do crate sfbinpack.
// ⚠️ Se este caminho falhar no compile, tenta um dos alternativos (comenta este, descomenta um):
use sfbinpack::chess::color::Color;
// use sfbinpack::chess::Color;
// use sfbinpack::Color;
use sfbinpack::CompressedTrainingDataEntryReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Uso: binpack_to_plain <input.binpack> <output.plain> [max_pos]");
        eprintln!("  max_pos: limite opcional de posições escritas (0 = todas)");
        std::process::exit(1);
    }
    let in_path = &args[1];
    let out_path = &args[2];
    let max_pos: u64 = if args.len() > 3 { args[3].parse().unwrap_or(0) } else { 0 };

    println!("🦅 binpack → .plain");
    println!("   in : {in_path}");
    println!("   out: {out_path}");

    let file = File::open(in_path).expect("não consegui abrir o binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .expect("não consegui criar o reader do binpack");

    let out = File::create(out_path).expect("não consegui criar o .plain");
    let mut w = BufWriter::with_capacity(1 << 20, out);

    let mut seen: u64 = 0;
    let mut kept: u64 = 0;

    while reader.has_next() {
        let entry = reader.next();
        seen += 1;

        // ── Filtro SF (igual ao examples/simple.rs) — posições calmas e posicionais ──
        let in_check = entry.pos.is_checked(entry.pos.side_to_move());
        let normal = entry.mv.mtype() == MoveType::Normal;
        let dest_empty = entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
        if !(entry.ply >= 16 && !in_check && entry.score.unsigned_abs() <= 10000
             && normal && dest_empty) {
            continue;
        }

        // ── score: o binpack dá score POV side-to-move. Converter p/ POV BRANCAS ──
        // entry.pos.side_to_move(): 0=brancas, 1=negras (convenção sfbinpack).
        // entry.pos.side_to_move() devolve Color (enum), não 0/1. entry.score é i16.
        let white_to_move = entry.pos.side_to_move() == Color::White;
        let score_i32 = entry.score as i32;
        let cp_white: i32 = if white_to_move { score_i32 } else { -score_i32 };
        let cp_white = cp_white.clamp(-3000, 3000);

        // ── wdl: estes binpacks "min-v2.v6" do SF têm result≈0 (otimizados p/ o SCORE, não o
        //   resultado da partida). Por isso derivamos o wdl do PRÓPRIO cp via sigmoid (escala
        //   400, igual ao db_para_plain_pv) — um sinal coerente, em vez de 0.5 morto.
        //   Se algum entry tiver result != 0, usamo-lo (mistura: result domina quando existe).
        let wdl_from_cp = 1.0_f64 / (1.0 + (-(cp_white as f64) / 400.0).exp());
        let wdl_white = match entry.result {
            r if r > 0 => if white_to_move { 1.0 } else { 0.0 },
            r if r < 0 => if white_to_move { 0.0 } else { 1.0 },
            _ => wdl_from_cp,   // result==0 (caso destes binpacks) → usa o cp
        };

        // FEN da posição
        let fen = entry.pos.fen().unwrap_or_default();

        // ── escrever no teu formato .plain ──
        // (o teu prepare-plain mistura o wdl do cp com este result; aqui damos o result real)
        writeln!(w, "{} | {} | {:.4}", fen, cp_white, wdl_white).ok();
        kept += 1;

        if kept % 1_000_000 == 0 {
            println!("   ... {kept} escritas / {seen} lidas");
        }
        if max_pos > 0 && kept >= max_pos {
            break;
        }
    }

    w.flush().ok();
    println!("✅ FINI: {kept} posições escritas (de {seen} lidas) em {out_path}");
    println!("   próximo: trainvsiriux prepare-plain → build-threats → merge_to_bullet → napctl");
}

// ────────────────────────────────────────────────────────────────────────────
// NOTA sobre a API (CONFIRMADA pelo cargo do Maréchal, sfbinpack 0.6.0):
//
// ✅ entry.pos.side_to_move() → devolve `Color` (enum), comparar com Color::White.
// ✅ entry.score → i16 (usar `as i32`).
// ✅ entry.ply, entry.mv.mtype()==MoveType::Normal, entry.pos.piece_at(sq).piece_type() — OK.
//
// A confirmar no próximo compile (se der erro, é só ajustar):
//   • entry.result → tipo? (assumi inteiro com sinal {1,0,-1}). Se for f32/enum, ajustar o match.
//   • entry.pos.fen() → assumi que devolve algo .unwrap_or_default()-able (Result/Option<String>).
//     Se devolver String direto, tira o .unwrap_or_default().
//   • caminho do `use ... Color`: se falhar, usa um dos alternativos comentados no topo.
// ────────────────────────────────────────────────────────────────────────────
