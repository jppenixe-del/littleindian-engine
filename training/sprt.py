#!/usr/bin/env python3
"""Harness de SPRT para o littleindian, usando cutechess-cli.

Uso típico (Bloco 4 do INTEGRACAO_BLOCOS.md):
    python3 training/sprt.py --new ./littleindian --base ./littleindian.base \
        --tc 8+0.08 --elo0 0 --elo1 2

Os limiares por defeito (elo0=0.0, elo1=2.0, alpha=beta=0.05) seguem o
CLAUDE.md. STC para triagem (8+0.08), LTC para confirmar (40+0.4).

Em VPS/hardware partilhado (CPU virtualizada, "hypervisor" nas flags):
--concurrency não deve exceder núcleos/2 (cada jogo usa 2 processos) e
--timemargin deve subir (ex.: 500) para absorver jitter de scheduling —
caso contrário, perdas por tempo poluem o resultado da SPRT.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BOOK = HERE / "openings.epd"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="binário candidato (engine A)")
    ap.add_argument("--base", required=True, help="binário de referência (engine B)")
    ap.add_argument("--tc", default="8+0.08", help="time control (default: 8+0.08, STC)")
    ap.add_argument("--book", default=str(DEFAULT_BOOK), help="aberturas em EPD (ver gen_openings.py)")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=20000, help="limite superior de rondas (a SPRT para antes, em geral)")
    ap.add_argument("--hash", type=int, default=16, help="MB de TT por instância de motor")
    ap.add_argument("--timemargin", type=int, default=200,
                     help="ms de tolerância antes de marcar perda por tempo (sobe em VPS/hardware partilhado com jitter)")
    ap.add_argument("--pgnout", default=None, help="caminho do PGN (default: training/sprt_<tc>.pgn)")
    args = ap.parse_args()

    new_bin = Path(args.new).resolve()
    base_bin = Path(args.base).resolve()
    book = Path(args.book).resolve()

    for p, label in [(new_bin, "--new"), (base_bin, "--base"), (book, "--book")]:
        if not p.exists():
            sys.exit(f"erro: {label} não encontrado: {p}")

    if shutil.which("cutechess-cli") is None:
        sys.exit("erro: cutechess-cli não está no PATH")

    pgnout = args.pgnout or str(HERE / f"sprt_{args.tc.replace('+', '_').replace(':', '-')}.pgn")

    cmd = [
        "stdbuf", "-oL",  # sem isto, o cutechess-cli usa buffering total (stdout não é tty)
        "cutechess-cli",
        "-engine", f"name=new", f"cmd={new_bin}", "proto=uci",
        "-engine", f"name=base", f"cmd={base_bin}", "proto=uci",
        "-each", f"tc={args.tc}", f"timemargin={args.timemargin}", f"option.Hash={args.hash}", "restart=on",
        "-openings", f"file={book}", "format=epd", "order=random",
        "-repeat",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=3", "score=400", "twosided=true",
        "-recover",  # um crash/stall não pode abortar o match todo (perde-se só essa partida)
        "-concurrency", str(args.concurrency),
        "-rounds", str(args.rounds),
        "-games", "2",
        "-sprt", f"elo0={args.elo0}", f"elo1={args.elo1}", f"alpha={args.alpha}", f"beta={args.beta}",
        "-pgnout", pgnout,
        "-ratinginterval", "20",
    ]

    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
