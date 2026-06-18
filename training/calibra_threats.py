#!/usr/bin/env python3
# 🦅 NapK9 V10 — HARNESS DE CALIBRAÇÃO DOS FULL THREATS.
#   Compara as features do MOTOR (C++ `dumpthreats`) com as do GERADOR DE TREINO (Rust
#   dump_threats) nas mesmas posições. TÊM de ser idênticas — senão o treino ensina features
#   diferentes das que o motor vê. Correr SEMPRE depois de mexer em qualquer dos lados.
#   USO: rustc -O bullet_integration/dump_threats.rs -o dump_threats
#        python3 training/calibra_threats.py --engine ./build/nap2siriux_bullet --rust ./dump_threats

import argparse, subprocess, sys

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
    "r4rk1/pp3ppp/2n1b3/q1pp4/8/P1Q1PN2/1P1B1PPP/R3K2R w KQ - 0 12",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/2P5/8/8/4k3/8/6p1/4K3 w - - 0 1",
    "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
    "4k3/8/4K3/8/8/8/8/8 w - - 0 1",
    "r1b1kb1r/pp3ppp/2n2n2/3qp3/8/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 7",
    "2kr3r/ppp2ppp/2n5/8/2B5/2N5/PPP2PPP/2KR3R w - - 0 12",
    "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/r4PPP/4R1K1 w - - 0 1",
    "k7/8/8/3Nn3/8/8/8/K7 w - - 0 1",
]

def dump_cpp(engine, fen):
    inp = f"uci\nisready\nposition fen {fen}\ndumpthreats\nquit\n"
    out = subprocess.run([engine], input=inp, capture_output=True, text=True, timeout=60).stdout
    return [l.strip() for l in out.splitlines() if l.startswith("persp")][:2]

def dump_rust(rust, fens):
    out = subprocess.run([rust], input="\n".join(fens) + "\n",
                         capture_output=True, text=True, timeout=60).stdout
    linhas = [l.strip() for l in out.splitlines() if l.startswith("persp")]
    return [linhas[i:i+2] for i in range(0, len(linhas), 2)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./build/nap2siriux_bullet")
    ap.add_argument("--rust", default="./dump_threats")
    args = ap.parse_args()
    rust_all = dump_rust(args.rust, FENS)
    falhas = 0
    for i, fen in enumerate(FENS):
        cpp = dump_cpp(args.engine, fen)
        rust = rust_all[i] if i < len(rust_all) else ["", ""]
        ok = cpp == rust
        nf = [len(l.split()) - 1 for l in cpp] if cpp else [0, 0]
        print(f"{'✅' if ok else '🔴 MISMATCH'}  [{i:2d}] {nf} features  {fen[:50]}")
        if not ok:
            falhas += 1
            for p in range(2):
                c = set((cpp[p] if p < len(cpp) else "").split()[1:])
                r = set((rust[p] if p < len(rust) else "").split()[1:])
                sc = sorted(int(x) for x in c - r); sr = sorted(int(x) for x in r - c)
                if sc: print(f"      persp{p} SÓ no C++ : {sc[:12]}")
                if sr: print(f"      persp{p} SÓ no Rust: {sr[:12]}")
    print()
    if falhas == 0:
        print(f"🦅 CALIBRADO: {len(FENS)}/{len(FENS)} posições IDÊNTICAS C++ ↔ Rust. Podes treinar a V10.")
    else:
        print(f"🔴 {falhas} posições divergem — NÃO treinar até resolver.")
        sys.exit(1)

if __name__ == "__main__":
    main()
