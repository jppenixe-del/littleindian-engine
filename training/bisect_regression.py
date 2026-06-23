#!/usr/bin/env python3
"""Bissecção automatizada: para cada commit num intervalo, faz checkout (num worktree
separado, sem tocar no checkout principal), compila, e corre um match curto contra
littleindian.stable via cutechess-cli. Reporta o score por commit -- o objetivo é
localizar ONDE no histórico o score começou a cair, sem precisar de testar todos os
commits individualmente até ter uma pista (bissecção binária estilo `git bisect`).

Uso:
    python3 training/bisect_regression.py --since 35d8390 --until HEAD --games 200 \
        --tc 8+0.08 --concurrency 8

    # ou um conjunto explícito de commits (mais controlo):
    python3 training/bisect_regression.py --commits 1fb14c5 79e6198 a9638a7 ... --games 200
"""
import argparse
import functools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # 🦅 sem isto, o stdout fica bufferizado
                                                # quando redirecionado p/ ficheiro (nohup
                                                # ... > log.txt) -- só aparecia no fim.

REPO = Path(__file__).resolve().parent.parent
WORKTREE = REPO / "training" / "_bisect_worktree"
BOOK = REPO / "training" / "openings.epd"


def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"FALHOU: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
        sys.exit(1)
    return r.stdout + r.stderr


def commit_list(since, until):
    out = run(["git", "log", "--oneline", f"{since}..{until}", "--",
               "src/search.cpp", "src/uci.cpp", "src/search.h"], cwd=REPO)
    lines = [l for l in out.strip().split("\n") if l]
    lines.reverse()  # mais antigo primeiro
    return [l.split()[0] for l in lines]


def build_at_commit(sha, net):
    if WORKTREE.exists():
        run(["git", "worktree", "remove", "--force", str(WORKTREE)], cwd=REPO, check=False)
    run(["git", "worktree", "add", "--detach", str(WORKTREE), sha], cwd=REPO)
    out = run(["make", "native-embed", f"NET={net}"], cwd=WORKTREE, check=False)
    binpath = WORKTREE / "littleindian"
    if not binpath.exists():
        print(f"  ⚠️ build falhou em {sha}:\n{out[-2000:]}")
        return None
    dest = REPO / f"littleindian.bisect_{sha}"
    shutil.copy(binpath, dest)
    dest.chmod(0o755)
    return dest


def run_match(engine_a, engine_b, games, tc, concurrency, book):
    pgnout = REPO / "training" / f"_bisect_{Path(engine_a).stem}.pgn"
    cmd = [
        "cutechess-cli",
        "-engine", "name=A", f"cmd={engine_a}", "proto=uci",
        "-engine", "name=B", f"cmd={engine_b}", "proto=uci",
        "-each", f"tc={tc}", "timemargin=200", "restart=on",
        "-openings", f"file={book}", "format=epd", "order=random",
        "-repeat",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=3", "score=600", "twosided=true",
        "-recover",
        "-concurrency", str(concurrency),
        "-rounds", str(max(1, games // 2)), "-games", "2",
        "-pgnout", str(pgnout),
    ]
    out = run(cmd, check=False)
    matches = re.findall(r"Score of A vs B: (\d+) - (\d+) - (\d+)", out)
    if not matches:
        return None
    w, l, d = (int(x) for x in matches[-1])
    n = w + l + d
    return (w + 0.5 * d) / n if n else None, w, l, d, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="35d8390")
    ap.add_argument("--until", default="HEAD")
    ap.add_argument("--commits", nargs="*", default=None, help="lista explícita de SHAs, ignora --since/--until")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--tc", default="8+0.08")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--net", default="nets/littleindian_1024.napk9")
    ap.add_argument("--baseline", default=str(REPO / "littleindian.stable"))
    ap.add_argument("--mode", choices=["sequential", "bisect"], default="bisect",
                     help="sequential: testa cada commit um a um. bisect: divide o intervalo a meio repetidamente")
    args = ap.parse_args()

    commits = args.commits or commit_list(args.since, args.until)
    if not commits:
        print("nenhum commit no intervalo (ou nenhum tocou search.cpp/uci.cpp/search.h)")
        return
    print(f"=== {len(commits)} commits candidatos ===")
    for c in commits:
        msg = run(["git", "log", "-1", "--format=%s", c], cwd=REPO).strip()
        print(f"  {c}  {msg}")

    results = {}
    progress_path = REPO / "training" / "_bisect_progress.json"

    def save_progress():
        with open(progress_path, "w") as f:
            json.dump(results, f, indent=2)

    def test_commit(sha):
        if sha in results:
            r = results[sha]
            return r["pct"] if isinstance(r, dict) else r
        print(f"\n--- testando {sha} ---")
        binpath = build_at_commit(sha, args.net)
        if binpath is None:
            results[sha] = None
            save_progress()
            return None
        res = run_match(str(binpath), args.baseline, args.games, args.tc, args.concurrency, BOOK)
        binpath.unlink(missing_ok=True)
        if res is None:
            print(f"  SEM RESULTADO (ver log)")
            results[sha] = None
            save_progress()
            return None
        pct, w, l, d, n = res
        print(f"  score: {pct*100:.1f}%  (+{w} ={d} -{l}, n={n})")
        results[sha] = {"pct": pct, "w": w, "l": l, "d": d, "n": n}
        save_progress()  # 🦅 grava a CADA commit testado -- sobrevive a interrupções,
                          # e permite ler o progresso de fora sem depender do stdout
        return pct

    if args.mode == "sequential":
        for sha in commits:
            test_commit(sha)
    else:
        # bissecção: testa o commit do meio; se já está mau (<0.46), o culpado está
        # ANTES ou NELE -- continua na metade esquerda. Se está bom (>=0.46), o
        # culpado está na metade direita (ainda por vir). Repete até sobrar 1 commit.
        lo, hi = 0, len(commits) - 1
        GOOD_THRESHOLD = 0.46
        while lo < hi:
            mid = (lo + hi) // 2
            pct = test_commit(commits[mid])
            if pct is None:
                print(f"  build/match falhou em {commits[mid]}, a tentar o vizinho")
                mid = min(mid + 1, hi)
                pct = test_commit(commits[mid])
                if pct is None:
                    break
            if pct < GOOD_THRESHOLD:
                hi = mid  # culpado está aqui ou antes
            else:
                lo = mid + 1  # culpado está depois
        if lo == hi:
            print(f"\n=== suspeito principal: {commits[lo]} ===")
            run(["git", "log", "-1", commits[lo]], cwd=REPO)

    print("\n=== resumo ===")
    for sha in commits:
        r = results.get(sha)
        if r is not None:
            pct = r["pct"] if isinstance(r, dict) else r
            print(f"  {sha}  {pct*100:.1f}%")
        elif sha in results:
            print(f"  {sha}  FALHOU")

    if WORKTREE.exists():
        run(["git", "worktree", "remove", "--force", str(WORKTREE)], cwd=REPO, check=False)


if __name__ == "__main__":
    main()
