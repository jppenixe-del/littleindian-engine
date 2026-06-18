// 🦅 dump_threats.rs — binário do HARNESS (calibra_threats.py). Compilar:
//   rustc -O bullet_integration/dump_threats.rs -o dump_threats
include!("napk9_v10_features.rs");

fn main() {
    use std::io::BufRead;
    // ── modo --test-pairs: valida o map pareado contra os geradores calibrados ──
    if std::env::args().any(|a| a == "--test-pairs") {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
            "8/2P5/8/8/4k3/8/6p1/4K3 w - - 0 1",
            "2kr3r/ppp2ppp/2n5/8/2B5/2N5/PPP2PPP/2KR3R b - - 0 12",
            "k7/8/8/3Nn3/8/8/8/K7 w - - 0 1",
        ];
        let mut falhas = 0;
        for fen in fens {
            let pos = parse_fen(fen).unwrap();
            let stm = if fen.contains(" b ") { 1usize } else { 0usize };
            // referência: geradores calibrados, por perspetiva
            let maps = compute_attack_maps_by_type(&pos);
            let mk_ref = |persp: usize| -> Vec<usize> {
                let mut v = Vec::new();
                gather_pieces(&pos, persp, &mut v);
                let mut t = Vec::new();
                gather_threats_full(&pos, persp, &maps, &mut t);
                v.extend(t.iter().map(|x| PIECE_FEATURES + x));
                v.sort_unstable(); v
            };
            let ref_stm = mk_ref(stm);
            let ref_ntm = mk_ref(stm ^ 1);
            // o map pareado
            let (mut got_stm, mut got_ntm) = (Vec::new(), Vec::new());
            map_features_pairs(&pos, stm, &mut |a, b| { got_stm.push(a); got_ntm.push(b); });
            let n_pairs = got_stm.len();
            got_stm.sort_unstable(); got_ntm.sort_unstable();
            let ok = got_stm == ref_stm && got_ntm == ref_ntm
                     && got_stm.iter().all(|&x| x < TOTAL_INPUTS_V10);
            println!("{}  pares={}  {}", if ok {"✅"} else {"🔴 MISMATCH"}, n_pairs, &fen[..40.min(fen.len())]);
            if !ok { falhas += 1;
                println!("   stm: ref={} got={}  ntm: ref={} got={}",
                         ref_stm.len(), got_stm.len(), ref_ntm.len(), got_ntm.len()); }
        }
        if falhas == 0 { println!("🦅 MAP PAREADO CALIBRADO (conjuntos idênticos aos geradores de referência)."); }
        else { std::process::exit(1); }
        return;
    }
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let fen = match line { Ok(l) => l, Err(_) => break };
        let fen = fen.trim();
        if fen.is_empty() { continue; }
        let pos = match parse_fen(fen) { Some(p) => p, None => { println!("FEN_INVALIDO"); continue; } };
        let maps = compute_attack_maps_by_type(&pos);
        for p in 0..2 {
            let mut v = Vec::with_capacity(128);
            gather_threats_full(&pos, p, &maps, &mut v);
            v.sort_unstable();
            print!("persp{}:", p);
            for f in &v { print!(" {}", f); }
            println!();
        }
    }
}
