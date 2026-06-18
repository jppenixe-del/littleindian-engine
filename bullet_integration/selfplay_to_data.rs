use std::env;
use std::fs::File;
use std::io::{BufWriter, Write, BufRead, BufReader};


// ── BUCKET_MAP (32 king-buckets, simétrico) — IDÊNTICO ao trainvsiriux.py ──
const BUCKET_MAP: [usize; 64] = [
    0,  1,  2,  3,  3,  2,  1,  0,
    4,  5,  6,  7,  7,  6,  5,  4,
    8,  9, 10, 11, 11, 10,  9,  8,
   12, 13, 14, 15, 15, 14, 13, 12,
   16, 17, 18, 19, 19, 18, 17, 16,
   20, 21, 22, 23, 23, 22, 21, 20,
   24, 25, 26, 27, 27, 26, 25, 24,
   28, 29, 30, 31, 31, 30, 29, 28,
];

// PIECE_IDX: Q=1,R=2,B=3,N=4,P=5 (rei tratado à parte como 0)
fn piece_idx(pt: u8) -> usize {
    // pt: 0=P,1=N,2=B,3=R,4=Q,5=K  (a nossa convenção interna do parser do FEN)
    match pt {
        0 => 5, // Pawn
        1 => 4, // Knight
        2 => 3, // Bishop
        3 => 2, // Rook
        4 => 1, // Queen
        _ => 0, // King (não usado aqui)
    }
}

// ── Board simples parseado do FEN ──
// pieces[sq] = Some((pt, white)) onde pt: 0=P,1=N,2=B,3=R,4=Q,5=K
struct Board {
    pieces: [Option<(u8, bool)>; 64],
    white_to_move: bool,
}

impl Board {
    fn from_fen(fen: &str) -> Option<Board> {
        let mut pieces = [None; 64];
        let mut parts = fen.split_whitespace();
        let placement = parts.next()?;
        let stm = parts.next().unwrap_or("w");
        // FEN: rank 8 primeiro. sq = rank*8 + file, com rank 0 = '1' (A1=0).
        let mut rank: i32 = 7;
        let mut file: i32 = 0;
        for ch in placement.chars() {
            match ch {
                '/' => { rank -= 1; file = 0; }
                '1'..='8' => { file += (ch as u8 - b'0') as i32; }
                _ => {
                    let white = ch.is_ascii_uppercase();
                    let pt = match ch.to_ascii_lowercase() {
                        'p' => 0u8, 'n' => 1, 'b' => 2, 'r' => 3, 'q' => 4, 'k' => 5,
                        _ => return None,
                    };
                    if rank < 0 || rank > 7 || file < 0 || file > 7 { return None; }
                    let sq = (rank * 8 + file) as usize;
                    pieces[sq] = Some((pt, white));
                    file += 1;
                }
            }
        }
        Some(Board { pieces, white_to_move: stm == "w" })
    }

    fn king_sq(&self, white: bool) -> Option<usize> {
        for sq in 0..64 {
            if let Some((5, w)) = self.pieces[sq] { if w == white { return Some(sq); } }
        }
        None
    }

    fn piece_count(&self) -> usize {
        self.pieces.iter().filter(|p| p.is_some()).count()
    }
}

// ── Geração de ataques (bitboards) — para o is_attacked_by dos threats ──
// Ocupação como u64 (bit sq). attacks_to: a casa `target` é atacada por alguma peça de `by_white`?
fn occ_bb(b: &Board) -> u64 {
    let mut o = 0u64;
    for sq in 0..64 { if b.pieces[sq].is_some() { o |= 1u64 << sq; } }
    o
}

const KNIGHT_DELTAS: [(i32, i32); 8] =
    [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)];
const KING_DELTAS: [(i32, i32); 8] =
    [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1,1),(1,-1),(-1,1),(-1,-1)];
const ROOK_DIRS: [(i32, i32); 4] = [(1,0),(-1,0),(0,1),(0,-1)];

#[inline] fn file_of(sq: usize) -> i32 { (sq % 8) as i32 }
#[inline] fn rank_of(sq: usize) -> i32 { (sq / 8) as i32 }

// A casa `target` é atacada por alguma peça da cor `by_white`?
fn is_attacked_by(b: &Board, target: usize, by_white: bool, occ: u64) -> bool {
    let tf = file_of(target);
    let tr = rank_of(target);

    // Peões: um peão branco em (r-1) ataca para cima; preto em (r+1) ataca para baixo.
    // target atacado por peão branco ⇔ existe peão branco em target-7/target-9 (nas diagonais de baixo)
    let pawn_dir: i32 = if by_white { -1 } else { 1 }; // de onde vem o peão atacante (rank)
    for df in [-1i32, 1] {
        let sf = tf + df;
        let sr = tr + pawn_dir;
        if sf >= 0 && sf < 8 && sr >= 0 && sr < 8 {
            let s = (sr * 8 + sf) as usize;
            if let Some((0, w)) = b.pieces[s] { if w == by_white { return true; } }
        }
    }
    // Cavalos
    for (df, dr) in KNIGHT_DELTAS {
        let sf = tf + df; let sr = tr + dr;
        if sf >= 0 && sf < 8 && sr >= 0 && sr < 8 {
            let s = (sr * 8 + sf) as usize;
            if let Some((1, w)) = b.pieces[s] { if w == by_white { return true; } }
        }
    }
    // Rei
    for (df, dr) in KING_DELTAS {
        let sf = tf + df; let sr = tr + dr;
        if sf >= 0 && sf < 8 && sr >= 0 && sr < 8 {
            let s = (sr * 8 + sf) as usize;
            if let Some((5, w)) = b.pieces[s] { if w == by_white { return true; } }
        }
    }
    // Bispos/Damas (diagonais)
    for (df, dr) in BISHOP_DIRS {
        let mut sf = tf + df; let mut sr = tr + dr;
        while sf >= 0 && sf < 8 && sr >= 0 && sr < 8 {
            let s = (sr * 8 + sf) as usize;
            if (occ >> s) & 1 == 1 {
                if let Some((pt, w)) = b.pieces[s] {
                    if w == by_white && (pt == 2 || pt == 4) { return true; }
                }
                break; // bloqueado
            }
            sf += df; sr += dr;
        }
    }
    // Torres/Damas (linhas/colunas)
    for (df, dr) in ROOK_DIRS {
        let mut sf = tf + df; let mut sr = tr + dr;
        while sf >= 0 && sf < 8 && sr >= 0 && sr < 8 {
            let s = (sr * 8 + sf) as usize;
            if (occ >> s) & 1 == 1 {
                if let Some((pt, w)) = b.pieces[s] {
                    if w == by_white && (pt == 3 || pt == 4) { return true; }
                }
                break;
            }
            sf += df; sr += dr;
        }
    }
    false
}

#[inline] fn make_threat(dir: usize, victim: usize, sq: usize) -> u16 {
    ((dir * 5 + victim) * 64 + sq) as u16
}

// gather_threats: persp 0 = brancas, 1 = pretas. Devolve índices [0,640).
// victim order [P,N,B,R,Q] → pt interno 0..4 (P=0,N=1,B=2,R=3,Q=4).
fn gather_threats(b: &Board, persp: usize, occ: u64) -> Vec<u16> {
    let me_white = persp == 0;
    let mut out = Vec::with_capacity(32);
    // victim v=0..4 corresponde a pt interno: v0=P(0), v1=N(1), v2=B(2), v3=R(3), v4=Q(4)
    for v in 0..5usize {
        let pt = v as u8; // P=0,N=1,B=2,R=3,Q=4 — coincide com a ordem [P,N,B,R,Q]
        for sq in 0..64usize {
            if let Some((spt, sw)) = b.pieces[sq] {
                if spt != pt { continue; }
                let s_idx = if persp == 0 { sq } else { sq ^ 56 };
                // dir0: as MINHAS peças atacadas por THEM
                if sw == me_white && is_attacked_by(b, sq, !me_white, occ) {
                    out.push(make_threat(0, v, s_idx));
                }
                // dir1: as peças DELES que EU ataco
                if sw != me_white && is_attacked_by(b, sq, me_white, occ) {
                    out.push(make_threat(1, v, s_idx));
                }
            }
        }
    }
    out
}

// board_to_napkav2: devolve (us_pieces, them_pieces) índices de peças [0,22528).
fn piece_features(b: &Board) -> Option<(Vec<u16>, Vec<u16>)> {
    let kw = b.king_sq(true)?;
    let kb = b.king_sq(false)?;
    let bucket_w = BUCKET_MAP[kw];
    let bucket_b = BUCKET_MAP[kb ^ 56];
    let mut us = Vec::with_capacity(32);
    let mut them = Vec::with_capacity(32);
    for sq in 0..64usize {
        if let Some((pt, white)) = b.pieces[sq] {
            let sq_f = sq ^ 56;
            if pt == 5 {
                // rei
                if !white { us.push((bucket_w * 704 + 0 * 64 + sq) as u16); }      // rei preto → us
                if white  { them.push((bucket_b * 704 + 0 * 64 + sq_f) as u16); }  // rei branco → them
            } else {
                let pidx = piece_idx(pt);
                let idx_w = pidx + (if white { 0 } else { 5 });
                us.push((bucket_w * 704 + idx_w * 64 + sq) as u16);
                let idx_b = pidx + (if !white { 0 } else { 5 });
                them.push((bucket_b * 704 + idx_b * 64 + sq_f) as u16);
            }
        }
    }
    Some((us, them))
}


// ════════════════════════════════════════════════════════════════════════════
// 🦅 selfplay_to_data — converte o txt do `nap2siriux extract` em .data (NapkRecord 268b).
//   Entrada: linhas "FEN | <N>cp | <wdl>" (do datagen/extract do motor).
//     ex:  rnbqkbnr/... w KQkq - 0 1 | 23cp | 0.5
//   Reusa EXATAMENTE board/threats/features/escrita do binpack_to_data (mesma indexação, 640
//   threats, mesmo NapkRecord). Só muda a LEITURA (txt em vez de binpack).
//   O wdl da linha é o RESULTADO REAL do jogo (0/0.5/1, POV brancas) — melhor que o sigmoid do cp.
// USO: selfplay_to_data <in.txt> <out.data> [max_pos]
// ════════════════════════════════════════════════════════════════════════════
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Uso: selfplay_to_data <in.txt> <out.data> [max_pos]");
        std::process::exit(1);
    }
    let in_path = &args[1];
    let out_path = &args[2];
    let max_pos: u64 = if args.len() > 3 { args[3].parse().unwrap_or(0) } else { 0 };

    println!("🦅 selfplay txt → .data (NapkRecord, threats em Rust)");
    println!("   in : {in_path}");
    println!("   out: {out_path}");

    let inf = File::open(in_path).expect("não consegui abrir o txt de entrada");
    let reader = BufReader::with_capacity(1 << 22, inf);
    let out = File::create(out_path).expect("não consegui criar o .data");
    let mut w = BufWriter::with_capacity(1 << 22, out);

    let mut seen: u64 = 0;
    let mut kept: u64 = 0;
    let mut bad: u64 = 0;
    let mut bucket_count: [u64; 8] = [0; 8];
    let mut buf = [0u8; 268];

    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        let line = line.trim();
        if line.is_empty() { continue; }
        seen += 1;

        // formato: "FEN | <N>cp | <wdl>"  (split por '|')
        let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
        if parts.len() < 3 { bad += 1; continue; }
        let fen = parts[0];
        // "<N>cp" → N
        let cp_str = parts[1].trim_end_matches("cp").trim();
        let score_i32: i32 = match cp_str.parse() { Ok(v) => v, Err(_) => { bad += 1; continue; } };
        // wdl "0.0/0.5/1.0" (POV brancas, resultado REAL do jogo)
        let wdl_white: f64 = match parts[2].parse() { Ok(v) => v, Err(_) => { bad += 1; continue; } };

        let board = match Board::from_fen(fen) { Some(b) => b, None => { bad += 1; continue; } };

        // filtro SF: |score|<=10000 (o ply>=16 e !check já vêm filtrados do extract do motor).
        if score_i32.unsigned_abs() > 10000 { continue; }

        // material bucket = (piece_count - 1)/4, clamp [0,7]
        let npieces = board.piece_count() as i32;
        let mb = (((npieces - 1) / 4).clamp(0, 7)) as u8;

        // features de peças
        let (us_p, them_p) = match piece_features(&board) { Some(v) => v, None => continue };
        let occ = occ_bb(&board);
        // threats (persp 0 us, persp 1 them), +22528
        let thr_us: Vec<u16> = gather_threats(&board, 0, occ).iter().map(|t| t + 22528).collect();
        let thr_them: Vec<u16> = gather_threats(&board, 1, occ).iter().map(|t| t + 22528).collect();

        let mut us_comb: Vec<u16> = us_p; us_comb.extend_from_slice(&thr_us);
        let mut them_comb: Vec<u16> = them_p; them_comb.extend_from_slice(&thr_them);
        us_comb.truncate(64);
        them_comb.truncate(64);

        // 🦅 CRÍTICO: o score do `extract` JÁ É POV BRANCAS (o datagen do Sirius faz
        //   `if sideToMove==BLACK score=-score` antes de gravar). NÃO inverter aqui — usar tal-como-vem.
        //   (o wdl da linha TAMBÉM é POV brancas: 1.0=brancas ganham.)
        let cp_white: i32 = score_i32;
        let cp_white = cp_white.clamp(-2500, 2500);

        // FLIP se turn=BLACK: trocar us/them, cp=-cp, wdl=1-wdl (POV brancas → POV side-to-move,
        //   que é o que o NapkRecord/treino espera: us=quem joga).
        let (final_us, final_them, final_cp, final_wdl) = if board.white_to_move {
            (us_comb, them_comb, cp_white as f32, wdl_white as f32)
        } else {
            (them_comb, us_comb, (-cp_white) as f32, (1.0 - wdl_white) as f32)
        };

        // ── escrever NapkRecord (268b, <64H 64H B B B B f f>) ──
        for b in buf.iter_mut() { *b = 0; }
        for (i, &f) in final_us.iter().take(64).enumerate() {
            buf[i*2..i*2+2].copy_from_slice(&f.to_le_bytes());
        }
        for (i, &f) in final_them.iter().take(64).enumerate() {
            buf[128 + i*2..128 + i*2+2].copy_from_slice(&f.to_le_bytes());
        }
        buf[256] = final_us.len().min(64) as u8;
        buf[257] = final_them.len().min(64) as u8;
        buf[258] = mb;
        buf[259] = 0;
        buf[260..264].copy_from_slice(&final_cp.to_le_bytes());
        buf[264..268].copy_from_slice(&final_wdl.to_le_bytes());
        w.write_all(&buf).ok();
        kept += 1;
        bucket_count[mb as usize] += 1;

        if kept % 2_000_000 == 0 { println!("   ... {kept} escritas / {seen} lidas"); }
        if max_pos > 0 && kept >= max_pos { break; }
    }

    w.flush().ok();
    println!("✅ FINI: {kept} posições (.data) de {seen} lidas ({bad} linhas inválidas) → {out_path}");
    print!("   distribuição mb:");
    for i in 0..8 { if bucket_count[i] > 0 { print!(" mb{i}={}", bucket_count[i]); } }
    println!();
}
