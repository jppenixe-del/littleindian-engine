// 🦅 NapK9 V10 — GERADOR DE FULL THREATS (Rust), CONTRATO DE CALIBRAÇÃO com o C++.
//
//   Tem de produzir EXATAMENTE os mesmos índices que o gatherThreatsFull (nnue_net.cpp).
//   O harness (training/calibra_threats.py) compara os dois — se divergirem, o treino ensina
//   features diferentes das que o motor vê = rede inútil. NUNCA mudar um lado sem o outro.
//
//   USO 1 (harness): rustc -O napk9_v10_features.rs -o dump_threats
//                    echo "<FEN>" | ./dump_threats   → "persp0: ..." / "persp1: ..."
//   USO 2 (bullet):  copiar gather_threats_full + make_threat_full + compute_attack_maps_by_type
//                    p/ o módulo napk9 do bullet (PC) e usar no map_features do NapkInputV10
//                    com offset 22528 (features = 22528 peças + 9216 threats = 31744).
//
//   FORMATO: side(2: 0=vítima minha,1=deles) × rel(2: 0=ataque,1=defesa) × atacante(6 P,N,B,R,Q,K)
//            × vítima(6) × casa_da_vítima(64, perspetiva: pretas sq^56) = 9216 features.

pub const THREAT_FEATURES_FULL: usize = 2 * 2 * 6 * 6 * 64; // 9216

#[inline]
pub fn make_threat_full(side: usize, rel: usize, att: usize, vic: usize, sq: usize) -> usize {
    (((side * 2 + rel) * 6 + att) * 6 + vic) * 64 + sq
}

#[derive(Clone, Copy, Default)]
pub struct Pos {
    pub pieces: [[u64; 6]; 2], // [cor][tipo] — cor 0=branca; P=0,N=1,B=2,R=3,Q=4,K=5
}
impl Pos {
    pub fn occ(&self) -> u64 {
        let mut o = 0u64;
        for c in 0..2 { for t in 0..6 { o |= self.pieces[c][t]; } }
        o
    }
}

fn pawn_attacks(c: usize, pawns: u64) -> u64 {
    if c == 0 { ((pawns & !0x0101010101010101) << 7) | ((pawns & !0x8080808080808080) << 9) }
    else      { ((pawns & !0x8080808080808080) >> 7) | ((pawns & !0x0101010101010101) >> 9) }
}
fn knight_attacks(sq: usize) -> u64 {
    let b = 1u64 << sq; let (nf_a, nf_h) = (!0x0101010101010101u64, !0x8080808080808080u64);
    let (nf_ab, nf_gh) = (!0x0303030303030303u64, !0xC0C0C0C0C0C0C0C0u64);
    ((b & nf_h) << 17) | ((b & nf_a) << 15) | ((b & nf_gh) << 10) | ((b & nf_ab) << 6)
        | ((b & nf_gh) >> 6) | ((b & nf_ab) >> 10) | ((b & nf_h) >> 15) | ((b & nf_a) >> 17)
}
fn king_attacks(sq: usize) -> u64 {
    let b = 1u64 << sq; let (nf_a, nf_h) = (!0x0101010101010101u64, !0x8080808080808080u64);
    ((b & nf_h) << 1) | ((b & nf_a) >> 1) | (b << 8) | (b >> 8)
        | ((b & nf_h) << 9) | ((b & nf_a) << 7) | ((b & nf_h) >> 7) | ((b & nf_a) >> 9)
}
fn ray(sq: usize, df: i32, dr: i32, occ: u64) -> u64 {
    let (mut f, mut r) = ((sq % 8) as i32, (sq / 8) as i32);
    let mut a = 0u64;
    loop {
        f += df; r += dr;
        if !(0..8).contains(&f) || !(0..8).contains(&r) { break; }
        let s = (r * 8 + f) as usize;
        a |= 1u64 << s;
        if occ & (1u64 << s) != 0 { break; }
    }
    a
}
fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    ray(sq,1,1,occ) | ray(sq,1,-1,occ) | ray(sq,-1,1,occ) | ray(sq,-1,-1,occ)
}
fn rook_attacks(sq: usize, occ: u64) -> u64 {
    ray(sq,1,0,occ) | ray(sq,-1,0,occ) | ray(sq,0,1,occ) | ray(sq,0,-1,occ)
}

// espelho EXATO do computeAttackMapsByType (C++)
pub fn compute_attack_maps_by_type(pos: &Pos) -> [[u64; 6]; 2] {
    let occ = pos.occ();
    let mut m = [[0u64; 6]; 2];
    for c in 0..2 {
        m[c][0] = pawn_attacks(c, pos.pieces[c][0]);
        let mut a = 0u64; let mut bb = pos.pieces[c][1];
        while bb != 0 { let s = bb.trailing_zeros() as usize; bb &= bb - 1; a |= knight_attacks(s); }
        m[c][1] = a;
        a = 0; bb = pos.pieces[c][2];
        while bb != 0 { let s = bb.trailing_zeros() as usize; bb &= bb - 1; a |= bishop_attacks(s, occ); }
        m[c][2] = a;
        a = 0; bb = pos.pieces[c][3];
        while bb != 0 { let s = bb.trailing_zeros() as usize; bb &= bb - 1; a |= rook_attacks(s, occ); }
        m[c][3] = a;
        a = 0; bb = pos.pieces[c][4];
        while bb != 0 { let s = bb.trailing_zeros() as usize; bb &= bb - 1;
                        a |= bishop_attacks(s, occ) | rook_attacks(s, occ); }
        m[c][4] = a;
        let ks = pos.pieces[c][5].trailing_zeros() as usize;
        m[c][5] = if ks < 64 { king_attacks(ks) } else { 0 };
    }
    m
}

// espelho EXATO do gatherThreatsFull (C++)
pub fn gather_threats_full(pos: &Pos, persp: usize, attacked_by: &[[u64; 6]; 2],
                           out: &mut Vec<usize>) {
    let me = persp;
    let them = me ^ 1;
    for vc in 0..2 {                       // cor da vítima: 0 = minha, 1 = deles
        let v_col = if vc == 0 { me } else { them };
        for vt in 0..6 {
            let mut victims = pos.pieces[v_col][vt];
            while victims != 0 {
                let sq = victims.trailing_zeros() as usize;
                victims &= victims - 1;
                let sq_p = if persp == 0 { sq } else { sq ^ 56 };
                let sq_bb = 1u64 << sq;
                for at in 0..6 {
                    if attacked_by[v_col ^ 1][at] & sq_bb != 0 {
                        out.push(make_threat_full(vc, 0, at, vt, sq_p));   // rel 0 = ataque
                    }
                    if attacked_by[v_col][at] & sq_bb != 0 {
                        out.push(make_threat_full(vc, 1, at, vt, sq_p));   // rel 1 = defesa
                    }
                }
            }
        }
    }
}

// ═══ PEÇAS HalfK2 (22528) — fórmula IDÊNTICA ao binpack_to_data.rs / motor ═══
//   bucket = BUCKET_MAP[ksq da perspetiva (^56 se preta)]; 704 = 11 planes × 64.
//   plane 0 = REI ADVERSÁRIO; planes 1-5 = minhas Q,R,B,N,P; 6-10 = deles (PIECE_IDX+5).
pub const BUCKET_MAP: [usize; 64] = [
     0,  1,  2,  3,  3,  2,  1,  0,
     4,  5,  6,  7,  7,  6,  5,  4,
     8,  9, 10, 11, 11, 10,  9,  8,
    12, 13, 14, 15, 15, 14, 13, 12,
    16, 17, 18, 19, 19, 18, 17, 16,
    20, 21, 22, 23, 23, 22, 21, 20,
    24, 25, 26, 27, 27, 26, 25, 24,
    28, 29, 30, 31, 31, 30, 29, 28,
];
#[inline]
pub fn piece_idx(pt: usize) -> usize { match pt { 0=>5, 1=>4, 2=>3, 3=>2, 4=>1, _=>0 } }

pub const PIECE_FEATURES: usize = 22528;
pub const TOTAL_INPUTS_V10: usize = PIECE_FEATURES + THREAT_FEATURES_FULL; // 31744

/// Features de PEÇAS p/ a perspetiva (0=branca). Rei próprio define o bucket (não é feature);
/// rei ADVERSÁRIO = plane 0. Espelho do piece_features do converter e do motor.
pub fn gather_pieces(pos: &Pos, persp: usize, out: &mut Vec<usize>) {
    let ksq_raw = pos.pieces[persp][5].trailing_zeros() as usize;
    let ksq = if persp == 0 { ksq_raw } else { ksq_raw ^ 56 };
    let bucket = BUCKET_MAP[ksq];
    for c in 0..2 {
        for t in 0..6 {
            let mut bb = pos.pieces[c][t];
            while bb != 0 {
                let sq_raw = bb.trailing_zeros() as usize; bb &= bb - 1;
                let sq = if persp == 0 { sq_raw } else { sq_raw ^ 56 };
                if t == 5 {
                    if c != persp { out.push(bucket * 704 + sq); }
                } else {
                    let idx = piece_idx(t) + (if c == persp { 0 } else { 5 });
                    out.push(bucket * 704 + idx * 64 + sq);
                }
            }
        }
    }
}

/// ⭐ MAP PAREADO p/ o bullet: (feat_stm, feat_ntm) por ELEMENTO FÍSICO (peça/threat).
///   O CONJUNTO por perspetiva é IGUAL ao gather_pieces/gather_threats_full (auto-teste valida).
///   stm: 0 = brancas a jogar. Threats com offset PIECE_FEATURES.
pub fn map_features_pairs<F: FnMut(usize, usize)>(pos: &Pos, stm: usize, f: &mut F) {
    let ntm = stm ^ 1;
    let ks_w = pos.pieces[0][5].trailing_zeros() as usize;
    let ks_b = pos.pieces[1][5].trailing_zeros() as usize;
    let bucket = [BUCKET_MAP[ks_w], BUCKET_MAP[ks_b ^ 56]];
    let piece_feat = |persp: usize, c: usize, t: usize, sq_raw: usize| -> Option<usize> {
        let sq = if persp == 0 { sq_raw } else { sq_raw ^ 56 };
        if t == 5 {
            if c != persp { Some(bucket[persp] * 704 + sq) } else { None }
        } else {
            Some(bucket[persp] * 704 + (piece_idx(t) + if c == persp { 0 } else { 5 }) * 64 + sq)
        }
    };
    // os 2 reis: cada um é feature numa SÓ perspetiva → emparelham entre si
    {
        let a = piece_feat(stm, ntm, 5, pos.pieces[ntm][5].trailing_zeros() as usize);
        let b = piece_feat(ntm, stm, 5, pos.pieces[stm][5].trailing_zeros() as usize);
        if let (Some(x), Some(y)) = (a, b) { f(x, y); }
    }
    for c in 0..2 {
        for t in 0..5 {
            let mut bb = pos.pieces[c][t];
            while bb != 0 {
                let sq_raw = bb.trailing_zeros() as usize; bb &= bb - 1;
                let x = piece_feat(stm, c, t, sq_raw).unwrap();
                let y = piece_feat(ntm, c, t, sq_raw).unwrap();
                f(x, y);
            }
        }
    }
    // THREATS físicos → par nas 2 perspetivas
    let maps = compute_attack_maps_by_type(pos);
    for v_col in 0..2 {
        for vt in 0..6 {
            let mut victims = pos.pieces[v_col][vt];
            while victims != 0 {
                let sq_raw = victims.trailing_zeros() as usize; victims &= victims - 1;
                let sq_bb = 1u64 << sq_raw;
                for at in 0..6 {
                    for rel in 0..2 {
                        let att_col = if rel == 0 { v_col ^ 1 } else { v_col };
                        if maps[att_col][at] & sq_bb != 0 {
                            let side_stm = if v_col == stm { 0 } else { 1 };
                            let sq_stm = if stm == 0 { sq_raw } else { sq_raw ^ 56 };
                            let sq_ntm = if ntm == 0 { sq_raw } else { sq_raw ^ 56 };
                            f(PIECE_FEATURES + make_threat_full(side_stm, rel, at, vt, sq_stm),
                              PIECE_FEATURES + make_threat_full(side_stm ^ 1, rel, at, vt, sq_ntm));
                        }
                    }
                }
            }
        }
    }
}

pub fn parse_fen(fen: &str) -> Option<Pos> {
    let board_part = fen.split_whitespace().next()?;
    let mut pos = Pos::default();
    let mut rank: i32 = 7; let mut file: i32 = 0;
    for ch in board_part.chars() {
        match ch {
            '/' => { rank -= 1; file = 0; }
            '1'..='8' => { file += ch.to_digit(10)? as i32; }
            _ => {
                let c = if ch.is_uppercase() { 0 } else { 1 };
                let t = match ch.to_ascii_lowercase() {
                    'p' => 0, 'n' => 1, 'b' => 2, 'r' => 3, 'q' => 4, 'k' => 5,
                    _ => return None,
                };
                if !(0..8).contains(&rank) || !(0..8).contains(&file) { return None; }
                pos.pieces[c][t] |= 1u64 << (rank * 8 + file);
                file += 1;
            }
        }
    }
    Some(pos)
}

// (o main do dump vive em dump_threats.rs — este ficheiro é INCLUÍVEL: sem main)

// ═══ 🦅 s29: THREATS COMO PARÂMETRO — modo 0=none, 1=clássico 640, 2=full 9216 ═══
pub const THREAT_FEATURES_640: usize = 640;
#[inline] pub fn make_threat_640(dir: usize, victim: usize, sq: usize) -> usize {
    (dir * 5 + victim) * 64 + sq      // espelho EXATO do binpack_to_data.rs / motor gatherThreats
}
pub const TOTAL_INPUTS_NONE: usize = PIECE_FEATURES;                       // 22528
pub const TOTAL_INPUTS_640:  usize = PIECE_FEATURES + THREAT_FEATURES_640; // 23168

/// map_features com o tipo de threats como parâmetro. mode 2 delega no full;
/// mode 0 = só peças (filtra os pares de threat do full — pares são homogéneos);
/// mode 1 = peças (filtradas do full) + threats 640 gerados com o contrato clássico:
///   vítimas P..Q (sem rei), dir0 = MINHAS peças atacadas por eles, dir1 = as DELES
///   que eu ataco — iterando o threat FÍSICO 1× e emitindo o par (stm, ntm).
pub fn map_features_pairs_mode<F: FnMut(usize, usize)>(pos: &Pos, stm: usize, mode: u8, f: &mut F) {
    if mode == 2 { map_features_pairs(pos, stm, f); return; }
    map_features_pairs(pos, stm, &mut |a, b| { if a < PIECE_FEATURES { f(a, b); } });
    if mode == 0 { return; }
    let ntm = stm ^ 1;
    let maps = compute_attack_maps_by_type(pos);
    let any = [
        (0..6).fold(0u64, |a, t| a | maps[0][t]),
        (0..6).fold(0u64, |a, t| a | maps[1][t]),
    ];
    for v_col in 0..2usize {
        for vt in 0..5usize {              // P,N,B,R,Q — rei NÃO é vítima no 640
            let mut victims = pos.pieces[v_col][vt];
            while victims != 0 {
                let sq_raw = victims.trailing_zeros() as usize; victims &= victims - 1;
                if any[v_col ^ 1] & (1u64 << sq_raw) != 0 {
                    let dir_stm = if v_col == stm { 0 } else { 1 };
                    let sq_stm = if stm == 0 { sq_raw } else { sq_raw ^ 56 };
                    let sq_ntm = if ntm == 0 { sq_raw } else { sq_raw ^ 56 };
                    f(PIECE_FEATURES + make_threat_640(dir_stm, vt, sq_stm),
                      PIECE_FEATURES + make_threat_640(dir_stm ^ 1, vt, sq_ntm));
                }
            }
        }
    }
}
