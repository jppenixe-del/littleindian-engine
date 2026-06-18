// binpack_to_data.rs — Napoleon / Nap2Siriux
// ============================================
// 🦅 Converte um binpack do Stockfish → DIRETAMENTE o .data do bullet (NapkRecord, 268 bytes),
//    calculando as features de peças (22528) + os 640 threats EM RUST.
//
//    SALTA TODO o pipeline lento: binpack → .data num passo. Sem prepare-plain, sem
//    build-threats (o gargalo Python de HORAS), sem merge_to_bullet. Em Rust são MINUTOS.
//
// COMO: lê cada posição do binpack, obtém o FEN (entry.pos.fen()), e a partir do FEN calcula
//   em Rust EXATAMENTE a mesma indexação do pipeline Python (board_to_napkav2 + gather_threats
//   + merge_to_bullet), escrevendo o NapkRecord binário.
//
//   Usar o FEN (em vez da API interna da Position) dá controlo TOTAL e não depende de métodos
//   incertos do sfbinpack (color/attacks/etc). O fen() já está validado (binpack_to_plain).
//
// ESPECIFICAÇÃO (replicada do trainvsiriux.py + merge_to_bullet.py do Maréchal):
//   PEÇAS:  bucket_w = BUCKET_MAP[ksq_w];  bucket_b = BUCKET_MAP[ksq_b ^ 56]
//     rei:   us  += bucket_w*704 + 0*64 + sq    (rei PRETO)
//            them+= bucket_b*704 + 0*64 + (sq^56) (rei BRANCO)
//     outra: idx_w = PIECE_IDX[pt] + (0 se branca senão 5)   [Q=1,R=2,B=3,N=4,P=5]
//            us  += bucket_w*704 + idx_w*64 + sq
//            idx_b = PIECE_IDX[pt] + (0 se preta senão 5)
//            them+= bucket_b*704 + idx_b*64 + (sq^56)
//   THREATS (victim order [P,N,B,R,Q], v=0..4):  make_threat(d,v,s) = (d*5+v)*64 + s
//     persp 0 (us):   d0 = minhas(brancas) atacadas por pretas;  d1 = pretas que as brancas atacam
//     persp 1 (them): igual mas me=pretas, e sq espelhado ^56
//   FUSÃO: us_comb = us_pieces ++ [22528 + t for t in threats(persp0)]  (cortar/pad a 64)
//          them_comb = them_pieces ++ [22528 + t for t in threats(persp1)]
//   FLIP: se turn=BLACK → trocar (us,them), cp=-cp, wdl=1-wdl. material bucket=(npieces-1)/4 [0,7].
//   REGISTO: <64H 64H B B B B f f> = us[64] them[64] us_count them_count mb 0 cp wdl (268 bytes).
//
// COMPILAR: pôr em /mnt/c/data/env/bullet/examples/ e declarar no Cargo.toml (como binpack_to_plain).
//   cargo run --release -p bullet_lib --example binpack_to_data -- in.binpack out.data [max_pos]
//
// ⚠️ Filtro SF (igual ao binpack_to_plain): ply>=16, sem xeque, |score|<=10000, mv Normal, destino vazio.
//
// 🦅 SHUFFLE EMBUTIDO (s29 cont. 22, processo "Coda"): o Coda lê o binpack a direito e baralha
//   num buffer interno do SfBinpackLoader (buffer_size_mb=256, 4 threads — confirmado lendo
//   github.com/adamtwiss/coda/training/configs/v7_1024h16x32s.rs, a config dele mais próxima
//   da nossa rede L1=1024; conceptual só, sem copiar código — Coda não tem licença declarada).
//   Nós não podemos trocar de loader (NapkInputV10 exige NapkRecordV2, não TrainingDataEntry)
//   NESTE conversor offline, mas podemos ter o MESMO efeito sem 2º passo: streaming shuffle
//   buffer (reservoir clássico) dentro deste binário. Resultado: 1 só comando, binpack →
//   .data2 JÁ baralhado, sem precisar do shuffle_data2.py depois.
//   (Para treino 100% on-the-fly, sem .data2 nenhum, ver napk9_v10_binpack_loader.rs +
//   napk9_train_v10_binpack.rs — esse é o caminho mais próximo do processo real do Coda.)
//   --shuffle-mb N (default 256, igual ao Coda; 0 desliga) ; --seed N (default 7).

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bullet_lib::value::loader::sfbinpack::{MoveType, PieceType, Color, Square};
// ⚠️ se Color/Square não estiverem neste módulo no teu bullet, ajusta o caminho do use acima
use sfbinpack::CompressedTrainingDataEntryReader;

// ── PRNG sem dependências externas (xorshift64, igual ao usado em gera_finais_syzygy.rs) ──
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed | 1) }
    fn next(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

// ── shuffle buffer streaming (reservoir): mantém `cap` registos em RAM; cada novo registo
//    troca de lugar com um slot aleatório e devolve o que saiu p/ escrever já baralhado.
//    Equivalente ao shuffle_buffer do Coda/TF, mas a 1 passo (sem .data2 intermédio no disco). ──
struct ShuffleBuf {
    rec_len: usize,
    cap: usize,
    buf: Vec<u8>,
    filled: usize,
    rng: Rng,
}
impl ShuffleBuf {
    fn new(rec_len: usize, cap: usize, seed: u64) -> Self {
        Self { rec_len, cap: cap.max(1), buf: vec![0u8; cap.max(1) * rec_len], filled: 0, rng: Rng::new(seed) }
    }
    /// Entra um registo novo; devolve um registo pronto a escrever (None só durante o aquecimento).
    fn push(&mut self, rec: &[u8]) -> Option<Vec<u8>> {
        if self.filled < self.cap {
            let off = self.filled * self.rec_len;
            self.buf[off..off + self.rec_len].copy_from_slice(rec);
            self.filled += 1;
            None
        } else {
            let slot = self.rng.below(self.cap);
            let off = slot * self.rec_len;
            let mut evicted = vec![0u8; self.rec_len];
            evicted.copy_from_slice(&self.buf[off..off + self.rec_len]);
            self.buf[off..off + self.rec_len].copy_from_slice(rec);
            Some(evicted)
        }
    }
    /// Fim do stream: baralha o que sobrou no buffer (Fisher-Yates) e devolve tudo concatenado.
    fn drain(&mut self) -> Vec<u8> {
        let mut order: Vec<usize> = (0..self.filled).collect();
        for i in (1..order.len()).rev() {
            let j = self.rng.below(i + 1);
            order.swap(i, j);
        }
        let mut out = Vec::with_capacity(self.filled * self.rec_len);
        for &idx in &order {
            let off = idx * self.rec_len;
            out.extend_from_slice(&self.buf[off..off + self.rec_len]);
        }
        out
    }
}

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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Uso: binpack_to_data <input.binpack> <output.data> [max_pos]");
        std::process::exit(1);
    }
	// 🦅 Configuração do Graceful Shutdown (Captura o Ctrl+C)
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Erro ao definir o handler de Ctrl+C");
    let in_path = &args[1];
    let out_path = &args[2];
    let max_pos: u64 = if args.len() > 3 { args[3].parse().unwrap_or(0) } else { 0 };

    // 🦅 FILTRO POR BUCKET (quota por material bucket mb 0..7):
    //   passar --dist "n0,n1,n2,n3,n4,n5,n6,n7" com as QUOTAS ABSOLUTAS por bucket.
    //   Aceita a posição só se o bucket dela ainda não atingiu a quota. Para quando todos cheios.
    //   Sem --dist → comportamento antigo (sem quota, usa max_pos).
    //   Ex p/ SF0 (só mb3-7 do SF; mb0-2 vêm dos finais Syzygy): --dist "0,0,0,22500000,25500000,25500000,21000000,15000000"
    // 🦅 V2 (--v2): escreve NapkRecordV2 (112B = bitboards + stm + meta) em vez do clássico
    //   (268B = features pré-computadas). O loader do bullet (NapkInputV10) gera as features
    //   on-the-fly com o código calibrado → muda-se o feature set SEM reconverter. ⭐
    let mut formato_v2 = std::env::args().any(|a| a == "--v2");
    let _ = &mut formato_v2;
    // 🦅 --min-ply N (default 16, o filtro SF clássico). 🔴 LIÇÃO s29: ply>=16 excluía TODAS as
    //   posições de abertura → rede com buracos nos lances 1-8 → Qh5/Qf3 cedo no Lichess.
    //   Decisão do Maréchal: ver TUDO desde o lance 1 → receita usa --min-ply 0.
    let min_ply: u16 = args.iter().position(|a| a == "--min-ply")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(16);
    if min_ply != 16 { println!("🦅 min_ply = {min_ply} (default SF era 16)"); }
    // censo de plies VISTOS (pré-filtro): revela se o binpack sequer TEM aberturas.
    // (jogos fishtest começam de um book — pode não haver plies baixos NENHUNS na fonte!)
    let mut plies_vistos: [u64; 3] = [0; 3];   // [<8, 8-15, >=16]
    let mut quota: Option<[u64; 8]> = None;
    for i in 0..args.len() {
        if args[i] == "--dist" && i + 1 < args.len() {
            let parts: Vec<u64> = args[i+1].split(',')
                .map(|s| s.trim().parse().unwrap_or(0)).collect();
            if parts.len() == 8 {
                let mut q = [0u64; 8];
                for j in 0..8 { q[j] = parts[j]; }
                quota = Some(q);
            } else {
                eprintln!("⚠️ --dist precisa de 8 valores (mb0..mb7); ignorado");
            }
        }
    }

    // 🦅 --total X: AUTO-distribuição. Dás o nº TOTAL de posições e o gerador calcula as quotas por
    //   bucket SOZINHO.
    //   🔄 DECISÃO V2 (Maréchal, rumo ao motor INDEPENDENTE sem HCE): os finais ENTRAM no treino.
    //   Receita V2: --dist "4000000,8000000,12000000,22500000,25500000,25500000,21000000,15000000"
    //   (mb0-2 dos binpacks SF, scores de busca profunda) + gera_finais_syzygy (TB perfeitos) por
    //   cima via cat. A decisão ANTIGA (abaixo) era válida enquanto o HCE+Syzygy jogavam os finais:
    //   DECISÃO ANTIGA: a NNUE NÃO treina finais — o Syzygy (em jogo) + o HCE do
    //   Sirius tratam dos finais (pureEndgamePieces=7 → HCE com ≤7 peças). Logo mb0/mb1 = 0 (o sf_mid
    //   também não os tem). Toda a quota vai p/ o MEIO-JOGO, onde a NNUE > HCE e a busca passa o tempo.
    //     mb0 mb1 mb2 mb3 mb4 mb5 mb6 mb7
    //      0   0   15  22  24  22  12   5   (%) — só meio-jogo, soma=100.
    const PHASE_W: [u64; 8] = [0, 0, 15, 22, 24, 22, 12, 5];
    for i in 0..args.len() {
        if args[i] == "--total" && i + 1 < args.len() {
            if let Ok(total) = args[i+1].trim().parse::<u64>() {
                let sum: u64 = PHASE_W.iter().sum();
                let mut q = [0u64; 8];
                for j in 0..8 { q[j] = total * PHASE_W[j] / sum; }
                quota = Some(q);
                println!("   🦅 --total {total} → auto-distribuição proporcional ao jogo real");
            } else {
                eprintln!("⚠️ --total precisa de um número; ignorado");
            }
        }
    }

    println!("🦅 binpack → .data (NapkRecord, threats calculados em Rust)");
    println!("   in : {in_path}");
    println!("   out: {out_path}");

    let file = File::open(in_path).expect("não consegui abrir o binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .expect("não consegui criar o reader do binpack");
    let out = File::create(out_path).expect("não consegui criar o .data");
    let mut w = BufWriter::with_capacity(1 << 22, out);

    let mut seen: u64 = 0;
    let mut kept: u64 = 0;
    let mut bucket_count: [u64; 8] = [0; 8];
    let mut buf = [0u8; 268];
    if let Some(q) = quota {
        println!("   filtro por bucket ATIVO — quotas mb0..7: {:?}", q);
    }

    // 🦅 shuffle embutido (processo "Coda" sem 2º passo): --shuffle-mb N (MB de buffer; 0=OFF).
    //   default 1024, igual ao SYK_BINPACK_BUFFER_MB de referência. rec_len segue --v2 (112B) ou
    //   o clássico (268B).
    let shuffle_mb: usize = args.iter().position(|a| a == "--shuffle-mb")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(256);
    let shuffle_seed: u64 = args.iter().position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(7);
    let rec_len = if formato_v2 { 112 } else { 268 };
    let mut shuf: Option<ShuffleBuf> = if shuffle_mb > 0 {
        let cap = (shuffle_mb * 1024 * 1024) / rec_len;
        println!("🦅 shuffle embutido ATIVO: buffer {shuffle_mb}MB ≈ {cap} registos (seed {shuffle_seed})");
        Some(ShuffleBuf::new(rec_len, cap, shuffle_seed))
    } else {
        println!("🦅 shuffle embutido OFF (--shuffle-mb 0) — saída na ordem do binpack");
        None
    };

    while reader.has_next() {
		// 🦅 Se o utilizador carregar em Ctrl+C, a flag muda e quebramos o ciclo aqui de forma limpa!
        if !running.load(Ordering::SeqCst) {
            println!("\n🛑 Interrupção manual (Ctrl+C) detetada!");
            println!("   A saltar para fora do ciclo e a despejar o buffer de escrita no disco...");
            break;
        }
        let entry = reader.next();
        seen += 1;

        // 🦅 TELEMETRIA POR BUCKET EM TEMPO REAL (A cada 5M de posições vasculhadas no binpack)
        if seen % 1_000_000 == 0 {
            print!("   ... [Lidas: {:>4}M] Total Escritas: {:>9} | ", seen / 1_000_000, kept);
            for i in 0..8 {
                if let Some(q) = quota {
                    if q[i] > 0 {
                        let pct = (bucket_count[i] as f64 / q[i] as f64) * 100.0;
                        print!("mb{}: {:>5.1}%  ", i, pct);
                    } else {
                        print!("mb{}: Off     ", i); // Se a quota para este bucket for 0
                    }
                } else {
                    print!("mb{}: {:<8} ", i, bucket_count[i]); // Fallback se correres sem --dist
                }
            }
            println!(); // Quebra de linha para o próximo log
        }
        // filtro SF (igual ao binpack_to_plain)
        let in_check = entry.pos.is_checked(entry.pos.side_to_move());
        let normal = entry.mv.mtype() == MoveType::Normal;
        let dest_empty = entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None;
        plies_vistos[if entry.ply < 8 { 0 } else if entry.ply < 16 { 1 } else { 2 }] += 1;
        if !(entry.ply >= min_ply && !in_check && entry.score.unsigned_abs() <= 10000
             && normal && dest_empty) {
            continue;
        }

        // ── 🦅 V2 FAST-PATH (s29): bitboards DIRETO do entry.pos — SEM fen() → String →
        //    parse → piece_features → 2× gather_threats (o V2 pagava o caminho clássico inteiro
        //    e deitava tudo fora!). Só 64× piece_at + pack 112B. Ganho grosso de velocidade.
        if formato_v2 {
            let mut bb = [[0u64; 6]; 2];
            let mut npieces: i32 = 0;
            for i in 0..64u32 {
                let pc = entry.pos.piece_at(Square::new(i));   // ⚠️ se Square::new não existir, troca p/ o construtor do teu sfbinpack
                let t: usize = match pc.piece_type() {
                    PieceType::Pawn => 0, PieceType::Knight => 1, PieceType::Bishop => 2,
                    PieceType::Rook => 3, PieceType::Queen => 4, PieceType::King => 5,
                    _ => { continue; }
                };
                let c: usize = if pc.color() == Color::White { 0 } else { 1 };
                bb[c][t] |= 1u64 << i;
                npieces += 1;
            }
            let mb = (((npieces - 1) / 4).clamp(0, 7)) as u8;
            if let Some(q) = quota {
                if bucket_count[mb as usize] >= q[mb as usize] { continue; }
            }
            // score do binpack JÁ é POV-stm → sem flips (o caminho antigo fazia 2 flips que se anulavam)
            let cp_stm = (entry.score as i32).clamp(-2500, 2500) as f32;
            let wdl_stm = (1.0_f64 / (1.0 + (-(cp_stm as f64) / 400.0).exp())) as f32;
            let stm: u8 = if entry.pos.side_to_move() == Color::White { 0 } else { 1 };
            let mut rec = [0u8; 112];
            for c in 0..2 { for t in 0..6 {
                let off = (c * 6 + t) * 8;
                rec[off..off + 8].copy_from_slice(&bb[c][t].to_le_bytes());
            }}
            rec[96] = stm; rec[97] = mb;
            rec[100..104].copy_from_slice(&cp_stm.to_le_bytes());
            rec[104..108].copy_from_slice(&wdl_stm.to_le_bytes());
            match &mut shuf {
                Some(sb) => { if let Some(out_rec) = sb.push(&rec) { w.write_all(&out_rec).ok(); } }
                None => { w.write_all(&rec).ok(); }
            }
            kept += 1;
            bucket_count[mb as usize] += 1;
            if kept % 5_000_000 == 0 { println!("   ... {kept} escritas / {seen} lidas (V2 fast)"); }
            if max_pos > 0 && kept >= max_pos { break; }
            if let Some(q) = &quota {
                let todos_cheios = (0..8).all(|i| q[i] == 0 || bucket_count[i] >= q[i]);
                if todos_cheios {
                    println!("🦅 TODAS as quotas mb atingidas ({kept} escritas) — a sair do loop.");
                    break;
                }
            }
            continue;
        }

        let fen = match entry.pos.fen() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let board = match Board::from_fen(&fen) { Some(b) => b, None => continue };

        // material bucket = (piece_count - 1) / 4, clamp [0,7] — calculado CEDO
        let npieces = board.piece_count() as i32;
        let mb = (((npieces - 1) / 4).clamp(0, 7)) as u8;

        // 🦅 FILTRO POR BUCKET: se há quota e este bucket já encheu, descarta JÁ (antes dos
        //   threats caros). Otimização: não calcula features/threats de posições que vão fora.
        if let Some(q) = quota {
            if bucket_count[mb as usize] >= q[mb as usize] {
                continue;
            }
        }

        // features de peças
        let (us_p, them_p) = match piece_features(&board) { Some(v) => v, None => continue };
        let occ = occ_bb(&board);
        // threats (persp 0 p/ us, persp 1 p/ them), deslocados +22528
        let thr_us: Vec<u16> = gather_threats(&board, 0, occ).iter().map(|t| t + 22528).collect();
        let thr_them: Vec<u16> = gather_threats(&board, 1, occ).iter().map(|t| t + 22528).collect();

        // fusão peças + threats
        let mut us_comb: Vec<u16> = us_p; us_comb.extend_from_slice(&thr_us);
        let mut them_comb: Vec<u16> = them_p; them_comb.extend_from_slice(&thr_them);
        us_comb.truncate(64);
        them_comb.truncate(64);

        // score POV brancas + wdl (sigmoid do cp, escala 400 — como nos binpacks "min")
        let score_i32 = entry.score as i32;
        let cp_white: i32 = if board.white_to_move { score_i32 } else { -score_i32 };
        let cp_white = cp_white.clamp(-2500, 2500);
        let wdl_white = 1.0_f64 / (1.0 + (-(cp_white as f64) / 400.0).exp());

        // FLIP se turn=BLACK: trocar us/them, cp=-cp, wdl=1-wdl (como board_to_napkav2)
        let (final_us, final_them, final_cp, final_wdl) = if board.white_to_move {
            (us_comb, them_comb, cp_white as f32, wdl_white as f32)
        } else {
            (them_comb, us_comb, (-cp_white) as f32, (1.0 - wdl_white) as f32)
        };

        // (bloco V2 antigo removido — o fast-path acima trata o --v2)
        // ── escrever o NapkRecord (268 bytes, <64H 64H B B B B f f>) ──
        for b in buf.iter_mut() { *b = 0; }
        // us_feats[64] @ 0
        for (i, &f) in final_us.iter().take(64).enumerate() {
            buf[i*2..i*2+2].copy_from_slice(&f.to_le_bytes());
        }
        // them_feats[64] @ 128
        for (i, &f) in final_them.iter().take(64).enumerate() {
            buf[128 + i*2..128 + i*2+2].copy_from_slice(&f.to_le_bytes());
        }
        buf[256] = final_us.len().min(64) as u8;   // us_count
        buf[257] = final_them.len().min(64) as u8; // them_count
        buf[258] = mb;                              // bucket
        buf[259] = 0;                               // padding
        buf[260..264].copy_from_slice(&final_cp.to_le_bytes());   // score f32
        buf[264..268].copy_from_slice(&final_wdl.to_le_bytes());  // wdl f32
        match &mut shuf {
            Some(sb) => { if let Some(out_rec) = sb.push(&buf) { w.write_all(&out_rec).ok(); } }
            None => { w.write_all(&buf).ok(); }
        }
        kept += 1;
        bucket_count[mb as usize] += 1;

        //if kept % 2_000_000 == 0 {
        //    println!("   ... {kept} escritas / {seen} lidas");
        //}
        if max_pos > 0 && kept >= max_pos { break; }
        // para quando TODOS os buckets com quota>0 atingiram a meta
        if let Some(q) = quota {
            let todos_cheios = (0..8).all(|i| q[i] == 0 || bucket_count[i] >= q[i]);
            if todos_cheios { break; }
        }
    }

    if let Some(mut sb) = shuf.take() {
        let rest = sb.drain();
        println!("🦅 a despejar o resto do shuffle buffer ({} registos)...", rest.len() / rec_len);
        w.write_all(&rest).ok();
    }
    w.flush().ok();
    println!("✅ FINI: {kept} posições (.data NapkRecord) de {seen} lidas → {out_path}");
    println!("   plies na FONTE (pré-filtro): <8: {}M | 8-15: {}M | >=16: {}M  {}",
        plies_vistos[0] / 1_000_000, plies_vistos[1] / 1_000_000, plies_vistos[2] / 1_000_000,
        if plies_vistos[0] + plies_vistos[1] < 1_000_000 {
            "⚠️ QUASE SEM ABERTURAS NA FONTE — min-ply não chega; precisamos doutra fonte (Lichess DB / datagen)"
        } else { "✅ a fonte tem material de abertura" });
    print!("   distribuição mb:");
    for i in 0..8 { if bucket_count[i] > 0 { print!(" mb{i}={}", bucket_count[i]); } }
    println!();
    println!("   PRONTO p/ treinar: NAPK_DATA={out_path} python3 napctl.py heads 1024");
}

// ────────────────────────────────────────────────────────────────────────────
// NOTAS (se o cargo der erro):
//  • entry.pos.fen() → assumi que devolve String (validámos no binpack_to_plain que dava p/
//    escrever). Se devolver Option/Result, ajusta o match (ex: `if let Ok(s) = entry.pos.fen()`).
//  • Tudo o resto (BUCKET_MAP, indexação, threats, 268 bytes) é Rust puro a partir do FEN —
//    NÃO depende de mais nada do sfbinpack além de fen/score/ply/mv/is_checked/piece_at (já
//    confirmados no binpack_to_plain que COMPILOU).
//  • A geração de ataques é própria (bitboard-ish por casa) → replica is_attacked_by do python-chess.
// ────────────────────────────────────────────────────────────────────────────
