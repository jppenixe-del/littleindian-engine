// gera_finais_syzygy.rs — Napoleon / Nap2Siriux
// =============================================
// 🦅 Gera FINAIS PERFEITOS (3-5 peças) com as Syzygy via FATHOM (FFI ao tbprobe.c do projeto),
//    no formato NapkRecord (268 bytes), prontos a juntar ao .data do SF para o dataset SF0.
//    RÁPIDO (Rust + Fathom C) — minutos, não horas como em Python.
//
// COMO funciona:
//   - gera posição aleatória legal de 3-5 peças (reis não adjacentes, peões em filas válidas)
//   - chama tb_probe_wdl_impl (Fathom) → WDL ótimo (a MESMA verdade que o motor usa)
//   - cp coerente com o motor: TB_WIN_SCORE=9000, ganho=9000, cursed=±50, empate=0 (syzygy.cpp)
//   - calcula peças (22528) + 640 threats (idêntico ao binpack_to_data.rs, já validado)
//   - escreve NapkRecord 268b
//
// SETUP FFI (Fathom):
//   Este exemplo precisa de LINKAR o tbprobe.c + tbchess.c. No bullet (Cargo), a forma mais
//   simples é um build.rs que compila o C com a crate `cc`. Ver instruções no fim do ficheiro.
//   Alternativa rápida: compilar à parte e linkar (também no fim).
//
// USO (depois de compilar):
//   ./gera_finais_syzygy <SYZYGY_PATH> <out.data> <n> [materiais]
//   ./gera_finais_syzygy /mnt/d/syzygy finais.data 40000000 KQvK,KRvK,KBNvK,KPvK,KQvKR
//
//   Junta ao SF:  cat napk_sf_jun.data finais.data > napk_sf0.data
// =============================================

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ffi::CString;
use std::os::raw::c_char;

// ── FFI ao Fathom (tbprobe.c do projeto) ──
// nomes REAIS dos símbolos: tb_init (não tb_init_impl) e tb_probe_wdl_impl. Confirmado no
// tbprobe.c (tb_init @850, tb_probe_wdl_impl @526) e no syzygy.cpp do motor (chama tb_init).
extern "C" {
    fn tb_init(path: *const c_char) -> bool;
    fn tb_probe_wdl_impl(
        white: u64, black: u64, kings: u64, queens: u64,
        rooks: u64, bishops: u64, knights: u64, pawns: u64,
        ep: u32, turn: bool) -> u32;
    // global do Fathom: nº máximo de peças suportado pelas TB carregadas (0 = nada carregou).
    static TB_LARGEST: u32;
}

// valores WDL do Fathom (tbprobe.h)
const TB_LOSS: u32 = 0;
const TB_BLESSED_LOSS: u32 = 1;
const TB_DRAW: u32 = 2;
const TB_CURSED_WIN: u32 = 3;
const TB_WIN: u32 = 4;
const TB_RESULT_FAILED: u32 = 0xFFFFFFFF;

// 🦅 FIX do bug "tudo é vitória": ANTES os finais ganhos escreviam cp=±9000 (TB_WIN_SCORE) + wdl=
//   1.0/0.0 EXATO. 40M de targets saturados (sigmoid(9000/400)≈1.0) ensinavam a rede a gritar
//   vitória em tudo (startpos dava ~0.95 em vez de ~0.5). Agora cap a ±1500: ainda diz "ganho forte"
//   (sigmoid(1500/400)=0.98) SEM saturação absoluta, e o wdl é COERENTE com o cp (sig do cp, não 1.0).
const NAPK_TBCAP: i32 = 1500;
fn sig(cp: i32) -> f64 { 1.0 / (1.0 + (-(cp as f64) / 400.0).exp()) }

// ── constantes do projeto (iguais ao binpack_to_data.rs) ──
const BUCKET_MAP: [usize; 64] = [
    0,1,2,3,3,2,1,0, 4,5,6,7,7,6,5,4, 8,9,10,11,11,10,9,8, 12,13,14,15,15,14,13,12,
    16,17,18,19,19,18,17,16, 20,21,22,23,23,22,21,20, 24,25,26,27,27,26,25,24, 28,29,30,31,31,30,29,28,
];
// pt interno: 0=P,1=N,2=B,3=R,4=Q,5=K
fn piece_idx(pt: u8) -> usize { match pt { 0=>5,1=>4,2=>3,3=>2,4=>1,_=>0 } }

// ── RNG simples (xorshift) — sem dependências ──
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed | 1) }
    fn next(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

// board: pieces[sq] = Some((pt, white)); pt 0..5
struct Board { pieces: [Option<(u8,bool)>;64], white_to_move: bool }
impl Board {
    fn king_sq(&self, white: bool) -> Option<usize> {
        (0..64).find(|&s| matches!(self.pieces[s], Some((5,w)) if w==white))
    }
    fn count(&self) -> usize { self.pieces.iter().filter(|p|p.is_some()).count() }
    fn bb(&self) -> (u64,u64,u64,u64,u64,u64,u64,u64) {
        let (mut w,mut b,mut k,mut q,mut r,mut bi,mut n,mut p)=(0u64,0,0,0,0,0,0,0);
        for sq in 0..64 { if let Some((pt,white))=self.pieces[sq] {
            let bit=1u64<<sq;
            if white {w|=bit;} else {b|=bit;}
            match pt {0=>p|=bit,1=>n|=bit,2=>bi|=bit,3=>r|=bit,4=>q|=bit,5=>k|=bit,_=>{}}
        }}
        (w,b,k,q,r,bi,n,p)
    }
}

fn dist(a: usize, b: usize) -> i32 {
    let (af,ar)=((a%8) as i32,(a/8) as i32);
    let (bf,br)=((b%8) as i32,(b/8) as i32);
    (af-bf).abs().max((ar-br).abs())
}

// gera posição aleatória legal p/ um material (white_pcs, black_pcs em pt interno)
fn random_pos(rng:&mut Rng, wp:&[u8], bp:&[u8]) -> Option<Board> {
    let n = 2 + wp.len() + bp.len();
    let mut squares=[0usize;8];
    let mut used=0u64;
    for i in 0..n {
        loop {
            let s=rng.below(64);
            if used & (1<<s)==0 { squares[i]=s; used|=1<<s; break; }
        }
    }
    let (wk,bk)=(squares[0],squares[1]);
    if dist(wk,bk)<=1 { return None; } // reis adjacentes = ilegal
    let mut pieces=[None;64];
    pieces[wk]=Some((5u8,true));
    pieces[bk]=Some((5u8,false));
    let mut i=2;
    for &pt in wp {
        let s=squares[i]; i+=1;
        if pt==0 && (s/8==0 || s/8==7) { return None; } // peão na 1ª/8ª = ilegal
        pieces[s]=Some((pt,true));
    }
    for &pt in bp {
        let s=squares[i]; i+=1;
        if pt==0 && (s/8==0 || s/8==7) { return None; }
        pieces[s]=Some((pt,false));
    }
    let white_to_move = rng.next()&1==0;
    let b = Board{pieces, white_to_move};
    // 🦅 LEGALIDADE: o rei do lado que NÃO joga não pode estar em xeque (senão o lado a mover
    //   capturaria o rei → posição ilegal → o Fathom rebenta no probe). Rejeitar.
    let o = occ(&b);
    let nonmover_white = !white_to_move; // o lado que NÃO joga
    if let Some(ksq) = b.king_sq(nonmover_white) {
        if attacked_by(&b, ksq, white_to_move, o) {
            return None; // rei do não-jogador atacado pelo jogador = ilegal
        }
    } else {
        return None;
    }
    Some(b)
}

// ── indexação peças (idêntica ao binpack_to_data) ──
fn piece_features(b:&Board)->Option<(Vec<u16>,Vec<u16>)>{
    let kw=b.king_sq(true)?; let kb=b.king_sq(false)?;
    let bw=BUCKET_MAP[kw]; let bb=BUCKET_MAP[kb^56];
    let mut us=Vec::new(); let mut them=Vec::new();
    for sq in 0..64 { if let Some((pt,white))=b.pieces[sq]{
        let sqf=sq^56;
        if pt==5 {
            if !white { us.push((bw*704+0*64+sq) as u16); }
            if white  { them.push((bb*704+0*64+sqf) as u16); }
        } else {
            let pidx=piece_idx(pt);
            us.push((bw*704+(pidx+if white{0}else{5})*64+sq) as u16);
            them.push((bb*704+(pidx+if !white{0}else{5})*64+sqf) as u16);
        }
    }}
    Some((us,them))
}

// ── threats (idêntico ao binpack_to_data) ──
const KNIGHT:[(i32,i32);8]=[(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)];
const KING:[(i32,i32);8]=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)];
const BISH:[(i32,i32);4]=[(1,1),(1,-1),(-1,1),(-1,-1)];
const ROOK:[(i32,i32);4]=[(1,0),(-1,0),(0,1),(0,-1)];
fn occ(b:&Board)->u64{ let mut o=0; for s in 0..64{if b.pieces[s].is_some(){o|=1<<s;}} o }
fn attacked_by(b:&Board,target:usize,by_white:bool,o:u64)->bool{
    let tf=(target%8) as i32; let tr=(target/8) as i32;
    let pd=if by_white{-1}else{1};
    for df in [-1i32,1]{let sf=tf+df;let sr=tr+pd;
        if sf>=0&&sf<8&&sr>=0&&sr<8{let s=(sr*8+sf) as usize;
            if let Some((0,w))=b.pieces[s]{if w==by_white{return true;}}}}
    for (df,dr) in KNIGHT{let sf=tf+df;let sr=tr+dr;
        if sf>=0&&sf<8&&sr>=0&&sr<8{let s=(sr*8+sf) as usize;
            if let Some((1,w))=b.pieces[s]{if w==by_white{return true;}}}}
    for (df,dr) in KING{let sf=tf+df;let sr=tr+dr;
        if sf>=0&&sf<8&&sr>=0&&sr<8{let s=(sr*8+sf) as usize;
            if let Some((5,w))=b.pieces[s]{if w==by_white{return true;}}}}
    for (df,dr) in BISH{let mut sf=tf+df;let mut sr=tr+dr;
        while sf>=0&&sf<8&&sr>=0&&sr<8{let s=(sr*8+sf) as usize;
            if (o>>s)&1==1{if let Some((pt,w))=b.pieces[s]{if w==by_white&&(pt==2||pt==4){return true;}}break;}
            sf+=df;sr+=dr;}}
    for (df,dr) in ROOK{let mut sf=tf+df;let mut sr=tr+dr;
        while sf>=0&&sf<8&&sr>=0&&sr<8{let s=(sr*8+sf) as usize;
            if (o>>s)&1==1{if let Some((pt,w))=b.pieces[s]{if w==by_white&&(pt==3||pt==4){return true;}}break;}
            sf+=df;sr+=dr;}}
    false
}
fn gather_threats(b:&Board,persp:usize,o:u64)->Vec<u16>{
    let me_white=persp==0; let mut out=Vec::new();
    for v in 0..5usize{ let pt=v as u8;
        for sq in 0..64{ if let Some((spt,sw))=b.pieces[sq]{ if spt!=pt{continue;}
            let si=if persp==0{sq}else{sq^56};
            if sw==me_white && attacked_by(b,sq,!me_white,o){ out.push(((0*5+v)*64+si) as u16); }
            if sw!=me_white && attacked_by(b,sq,me_white,o){ out.push(((1*5+v)*64+si) as u16); }
        }}}
    out
}

fn parse_material(s:&str)->(Vec<u8>,Vec<u8>){
    let parts:Vec<&str>=s.split('v').collect();
    let pcs=|side:&str|->Vec<u8>{ side.chars().filter_map(|c| match c {
        'K'=>None,'Q'=>Some(4),'R'=>Some(3),'B'=>Some(2),'N'=>Some(1),'P'=>Some(0),_=>None }).collect() };
    (pcs(parts[0]), pcs(parts.get(1).copied().unwrap_or("")))
}

fn main(){
    let args:Vec<String>=env::args().collect();
    if args.len()<4 {
        eprintln!("uso: gera_finais_syzygy <SYZYGY_PATH> <out.data> <n> [materiais]");
        std::process::exit(1);
    }
    let sz_path=&args[1];
    let out_path=&args[2];
    let n:u64=args[3].parse().unwrap_or(40_000_000);
    let materiais_str = args.get(4).map(|s|s.as_str())
        .unwrap_or("KQvK,KRvK,KBNvK,KPvK,KQvKR,KRvKB,KRvKN,KQvKP,KRPvKR,KPPvK,KBPvK,KNPvK");
    let materiais:Vec<(Vec<u8>,Vec<u8>)>=materiais_str.split(',').map(parse_material).collect();

    // init Fathom
    let c=CString::new(sz_path.as_str()).unwrap();
    let ok=unsafe{ tb_init(c.as_ptr()) };
    let largest = unsafe { TB_LARGEST };
    if !ok || largest == 0 {
        eprintln!("❌ tb_init falhou ou TB_LARGEST=0 (ok={ok}, largest={largest})");
        eprintln!("   verifica o SyzygyPath: {sz_path}");
        eprintln!("   precisa de .rtbw E .rtbz (não só .rtbw). Conta os .rtbz:");
        eprintln!("     ls {sz_path}/*.rtbz | wc -l");
        std::process::exit(1);
    }
    println!("🦅 Fathom OK — TB_LARGEST={largest} (peças máx. suportadas)");
    println!("🦅 gera {n} finais Syzygy (Fathom) → {out_path}");
    println!("   materiais: {materiais_str}");

    // filtrar materiais que excedem TB_LARGEST (não gerar 5 peças se só há TB de 4, etc)
    let materiais: Vec<(Vec<u8>,Vec<u8>)> = materiais
        .into_iter()
        .filter(|(w,b)| (2 + w.len() + b.len()) as u32 <= largest)
        .collect();
    if materiais.is_empty() {
        eprintln!("❌ nenhum material cabe em TB_LARGEST={largest}"); std::process::exit(1);
    }
    println!("   materiais usados (<= {largest} peças): {}", materiais.len());

    let out=File::create(out_path).expect("criar out");
    let mut w=BufWriter::with_capacity(1<<22, out);
    let mut rng=Rng::new(0x9E3779B97F4A7C15);
    let mut buf=[0u8;268];
    let mut escritas:u64=0; let mut tent:u64=0; let mut mbc=[0u64;8];

    while escritas<n {
        tent+=1;
        let (wp,bp)=&materiais[rng.below(materiais.len())];
        let board=match random_pos(&mut rng,wp,bp){Some(b)=>b,None=>continue};
        let (white,black,kings,queens,rooks,bishops,knights,pawns)=board.bb();
        let res=unsafe{ tb_probe_wdl_impl(white,black,kings,queens,rooks,bishops,knights,pawns,
                                          0u32, board.white_to_move) };
        if res==TB_RESULT_FAILED { continue; }
        // cp/wdl POV stm (como o syzygy.cpp)
        let (cp_stm, wdl_stm):(i32,f64)=match res {
            TB_WIN => (NAPK_TBCAP, sig(NAPK_TBCAP)),
            TB_LOSS => (-NAPK_TBCAP, sig(-NAPK_TBCAP)),
            TB_CURSED_WIN => (50, 0.55),
            TB_BLESSED_LOSS => (-50, 0.45),
            TB_DRAW => (0, 0.5),
            _ => continue,
        };
        // features POV brancas + flip se turn preto (idêntico ao binpack_to_data)
        let (us_p,them_p)=match piece_features(&board){Some(v)=>v,None=>continue};
        let o=occ(&board);
        let thr_us:Vec<u16>=gather_threats(&board,0,o).iter().map(|t|t+22528).collect();
        let thr_them:Vec<u16>=gather_threats(&board,1,o).iter().map(|t|t+22528).collect();
        let mut us:Vec<u16>=us_p; us.extend_from_slice(&thr_us); us.truncate(64);
        let mut them:Vec<u16>=them_p; them.extend_from_slice(&thr_them); them.truncate(64);

        // converter cp/wdl POV-stm → POV-brancas, depois aplicar o flip IGUAL ao binpack_to_data
        let cp_white: i32 = if board.white_to_move { cp_stm } else { -cp_stm };
        let wdl_white: f64 = if board.white_to_move { wdl_stm } else { 1.0 - wdl_stm };
        let (fus,fthem,fcp,fwdl)= if board.white_to_move {
            (us, them, cp_white as f32, wdl_white as f32)
        } else {
            (them, us, (-cp_white) as f32, (1.0 - wdl_white) as f32)
        };

        let npc=board.count() as i32;
        let mb=(((npc-1)/4).clamp(0,7)) as u8;

        for b in buf.iter_mut(){*b=0;}
        for (i,&f) in fus.iter().take(64).enumerate(){ buf[i*2..i*2+2].copy_from_slice(&f.to_le_bytes()); }
        for (i,&f) in fthem.iter().take(64).enumerate(){ buf[128+i*2..128+i*2+2].copy_from_slice(&f.to_le_bytes()); }
        buf[256]=fus.len().min(64) as u8;
        buf[257]=fthem.len().min(64) as u8;
        buf[258]=mb; buf[259]=0;
        buf[260..264].copy_from_slice(&fcp.to_le_bytes());
        buf[264..268].copy_from_slice(&fwdl.to_le_bytes());
        w.write_all(&buf).ok();
        escritas+=1; mbc[mb as usize]+=1;
        if escritas%1_000_000==0 { println!("   ... {escritas} escritas / {tent} tentativas"); }
    }
    w.flush().ok();
    println!("✅ FINI: {escritas} finais → {out_path}");
    print!("   mb:"); for i in 0..8 { if mbc[i]>0 { print!(" mb{i}={}",mbc[i]); } } println!();
}

// ════════════════════════════════════════════════════════════════════════════
// SETUP FFI (uma vez) — compilar com o Fathom (tbprobe.c):
//
// OPÇÃO 1 (standalone, simples — recomendada p/ um utilitário):
//   1. copia tbprobe.c, tbchess.c, tbprobe.h, tbconfig.h, stdendian.h p/ uma pasta.
//   2. compila o C:   gcc -O3 -c tbprobe.c -o tbprobe.o
//      (tbprobe.c já faz #include do tbchess.c, por isso só este; se não, compila os dois)
//   3. compila o Rust linkando o .o:
//      rustc -O gera_finais_syzygy.rs -o gera_finais_syzygy -L. -ltbprobe \
//        ou:  rustc -O gera_finais_syzygy.rs tbprobe.o -o gera_finais_syzygy -C link-args="-lm"
//
// OPÇÃO 2 (dentro do bullet/cargo, com build.rs + crate `cc`):
//   - põe gera_finais_syzygy.rs em examples/ e declara no Cargo.toml [[example]].
//   - cria build.rs:  fn main(){ cc::Build::new().file("syzygy/tbprobe.c").compile("fathom"); }
//   - add ao Cargo.toml [build-dependencies]: cc = "1"
//   - cargo build --release --example gera_finais_syzygy
//
// ⚠️ tb_init_impl precisa dos ficheiros .rtbw/.rtbz no SYZYGY_PATH (as 3-4-5 que o motor usa).
// ════════════════════════════════════════════════════════════════════════════
