#!/usr/bin/env python3
"""Gera um livro de aberturas EPD a partir de um livro Polyglot (.bin),
para uso com `cutechess-cli -openings file=... format=epd`.

Uso:
    python3 training/gen_openings.py --book training/book.bin \
        --out training/openings.epd --n 200 --plies 8
"""
import argparse
import random

import chess
import chess.polyglot


def random_line(book: chess.polyglot.MemoryMappedReader, plies: int) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        try:
            entry = book.weighted_choice(board)
        except IndexError:
            break  # fora do livro, fica com a linha que já tem
        board.push(entry.move)
    return board


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="número de posições/aberturas")
    ap.add_argument("--plies", type=int, default=8, help="profundidade em meios-lances")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    seen = set()
    lines = []
    with chess.polyglot.open_reader(args.book) as book:
        attempts = 0
        while len(lines) < args.n and attempts < args.n * 20:
            attempts += 1
            board = random_line(book, args.plies)
            fen = board.fen()
            if fen in seen:
                continue
            seen.add(fen)
            lines.append(fen)

    with open(args.out, "w") as f:
        for fen in lines:
            f.write(fen + "\n")

    print(f"escritas {len(lines)} aberturas em {args.out}")


if __name__ == "__main__":
    main()
