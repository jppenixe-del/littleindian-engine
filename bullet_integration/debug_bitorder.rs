// 🦅 debug_bitorder.rs — diagnóstico pontual: confirma se pieces_bb_color() usa a mesma
//    convenção de bit (a1=bit0 .. h8=bit63) que o resto do código (FEN, Board). Lê 1 entry
//    do binpack, mostra o FEN (trusted) e a casa do rei branco segundo pieces_bb_color().
use std::env;
use std::fs::File;
use bullet_lib::value::loader::sfbinpack::{PieceType};
use sfbinpack::chess::color::Color;
use sfbinpack::CompressedTrainingDataEntryReader;

fn material(pos: &sfbinpack::chess::position::Position) -> (i32, i32) {
    let val = |pt: PieceType| -> i32 { match pt {
        PieceType::Pawn => 1, PieceType::Knight => 3, PieceType::Bishop => 3,
        PieceType::Rook => 5, PieceType::Queen => 9, _ => 0 } };
    let mut w = 0; let mut b = 0;
    for pt in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen] {
        w += (pos.pieces_bb_color(Color::White, pt).bits().count_ones() as i32) * val(pt);
        b += (pos.pieces_bb_color(Color::Black, pt).bits().count_ones() as i32) * val(pt);
    }
    (w, b)
}

fn sq_name(sq: u32) -> String {
    let file = (sq % 8) as u8;
    let rank = (sq / 8) as u8;
    format!("{}{}", (b'a' + file) as char, rank + 1)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let file = File::open(path).expect("abrir binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(file).expect("reader");
    let mut shown = 0;
    let mut scanned = 0;
    while reader.has_next() && shown < n {
        let entry = reader.next();
        scanned += 1;
        let (wmat, bmat) = material(&entry.pos);
        if (wmat - bmat).abs() < 4 { continue; }  // só posições com desequilíbrio óbvio
        let fen = entry.pos.fen().unwrap_or_default();
        let stm_white = entry.pos.side_to_move() == Color::White;
        println!("FEN: {}  (material W={} B={}, stm={})", fen, wmat, bmat, if stm_white {"W"} else {"B"});
        println!("  entry.score={}  entry.result={}  entry.ply={}", entry.score, entry.result, entry.ply);
        shown += 1;
    }
    println!("(varridas {} entries p/ achar {} com desequilíbrio >=4)", scanned, shown);
}
