#!/usr/bin/env python3
"""Gauntlet littleindian vs Stockfish a vários níveis (Skill Level 0-20).

Não é um teste de SPRT (não decide "ganha Elo ou não") — é um benchmark
absoluto: para cada nível do Stockfish, corre N jogos e reporta a
percentagem de pontos do littleindian. Serve para situar cada versão do
motor numa escala familiar ("bate o nível X do Stockfish").

Uso:
    python3 training/gauntlet_stockfish.py --engine ./littleindian.new \
        --levels 1-20 --games 8 --tc 10+0.1
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BOOK = HERE / "openings.epd"


def parse_levels(spec: str):
    levels = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            levels.extend(range(int(lo), int(hi) + 1))
        else:
            levels.append(int(part))
    return levels


def run_level(engine, stockfish, level, games, tc, book, concurrency, timemargin):
    pgnout = HERE / f"gauntlet_{Path(engine).stem}_lvl{level}.pgn"
    cmd = [
        "stdbuf", "-oL", "cutechess-cli",
        "-engine", "name=littleindian", f"cmd={engine}", "proto=uci",
        "-engine", "name=stockfish", f"cmd={stockfish}", "proto=uci",
        f"option.Skill Level={level}",
        "-each", f"tc={tc}", f"timemargin={timemargin}", "restart=on",
        "-openings", f"file={book}", "format=epd", "order=random",
        "-repeat",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=3", "score=600", "twosided=true",
        "-recover",
        "-concurrency", str(concurrency),
        "-rounds", str(max(1, games // 2)),
        "-games", "2",
        "-pgnout", str(pgnout),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def parse_score(output):
    matches = re.findall(r"Score of littleindian vs stockfish: (\d+) - (\d+) - (\d+)", output)
    if not matches:
        return None
    w, l, d = (int(x) for x in matches[-1])
    return w, l, d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="binário littleindian a testar")
    ap.add_argument("--stockfish", default=shutil.which("stockfish") or "/usr/games/stockfish")
    ap.add_argument("--levels", default="1-20", help="ex.: 1-20 ou 1,5,10,15,20")
    ap.add_argument("--games", type=int, default=8, help="jogos por nível (default: 8)")
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--book", default=str(DEFAULT_BOOK))
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timemargin", type=int, default=200)
    ap.add_argument("--stop-after-losses", type=int, default=2,
                     help="pára o sweep após N níveis consecutivos com <25%% de pontos (0 = nunca parar)")
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    if not engine.exists():
        sys.exit(f"erro: motor não encontrado: {engine}")
    if not Path(args.stockfish).exists():
        sys.exit(f"erro: stockfish não encontrado: {args.stockfish}")

    levels = parse_levels(args.levels)
    print(f"=== gauntlet: {engine.name} vs Stockfish (níveis {levels[0]}-{levels[-1]}) ===")

    consecutive_bad = 0
    for level in levels:
        out = run_level(str(engine), args.stockfish, level, args.games,
                         args.tc, args.book, args.concurrency, args.timemargin)
        parsed = parse_score(out)
        if parsed is None:
            print(f"nível {level:2d}: SEM RESULTADO (ver log) ")
            print(out[-2000:])
            continue
        w, l, d = parsed
        n = w + l + d
        pct = (w + 0.5 * d) / n * 100 if n else 0.0
        print(f"nível {level:2d}: +{w} ={d} -{l}  ({pct:.1f}%, n={n})")
        sys.stdout.flush()

        if args.stop_after_losses:
            if pct < 25.0:
                consecutive_bad += 1
                if consecutive_bad >= args.stop_after_losses:
                    print(f"... {consecutive_bad} níveis seguidos <25%, a parar o sweep aqui.")
                    break
            else:
                consecutive_bad = 0


if __name__ == "__main__":
    main()
