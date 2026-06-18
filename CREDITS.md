# Credits

## Original work

littleindian is written from scratch for this project: board representation, bitboards, Zobrist hashing, magic-bitboard move generation, search (PVS/iterative deepening/quiescence), transposition table, time management and the UCI front-end are all original code, with no code copied from another engine.

The **NapK9** evaluation format, the `nets/littleindian_1024.napk9` network, and the trainer in `bullet_integration/` are original work belonging to this project, carried over from prior in-house work (the `src/napoleon/` module). The *threats* feature concept itself originates with Stockfish (which introduced full threat features in its NNUE); Reckless later simplified/lightened the formula, and this project's full-threats implementation (`gatherThreatsFull` in `nnue_net.cpp`, `gather_threats_full` in `napk9_v10_features.rs`) follows Reckless's lighter formula, not Stockfish's original one — read **conceptually only**, no code copied (same conceptual-reference basis as the rest of this section).

## Conceptual references

Search and move-generation techniques draw on publicly documented ideas, not on copied code:

- [Chess Programming Wiki](https://www.chessprogramming.org/) — general algorithms and data structures (magic bitboards, PVS, quiescence search, transposition tables, move ordering, pruning/reduction techniques).
- Public engine-programming community discussion (e.g. the Engine Programming Discord/forums) for technique names and rough effect sizes used to prioritize `docs/INTEGRACAO_BLOCOS.md`.
- [Reckless](https://github.com/codedeliveryservice/Reckless) is referenced **conceptually only** (which techniques to try, in what order) in `docs/INTEGRACAO_BLOCOS.md`. Reckless is AGPL-3.0-licensed; **no code or network weights from Reckless are used or copied.**
- [Stockfish](https://github.com/official-stockfish/Stockfish) (GPL-3.0), [Reckless](https://github.com/codedeliveryservice/Reckless) (AGPL-3.0), and [Coda](https://github.com/adamtwiss/coda) (no declared license — all rights reserved) were read **conceptually only** to check the structural guards of null-move pruning, the log-depth×log-movecount shape of late move reductions, and the swap-list algorithm for static exchange evaluation, against established practice. No code or tuned constants were copied from any of the three; only the algorithm/guard *structure* informed `src/search.cpp`. littleindian's own neutral piece-value table (P=100, N=325, B=325, R=500, Q=975) is used throughout, not the reference engines' tuned values.
- [Coda](https://github.com/adamtwiss/coda)'s `training/configs/` (e.g. `v7_1024h16x32s.rs`) were also read **conceptually only** for its NapK9 V10 training schedule shape (`eval_scale`, binpack filter, batch/superbatch sizing, shuffle-buffer approach) — see `bullet_integration/napk9_train_v10_coda.rs` and `napk9_train_v10_binpack.rs`. No code was copied; the on-the-fly binpack loader (`napk9_v10_binpack_loader.rs`) is a NapK9-specific reimplementation, structurally modeled on the MIT-licensed [bullet](https://github.com/jw1912/bullet) project's own `SfBinpackLoader` (read directly from its public source, not from Coda).

## Tools used during development (not bundled)

- [cutechess-cli](https://github.com/cutechess/cutechess) — engine-vs-engine match runner used for SPRT testing (`training/sprt.py`). Not part of this repository; install separately.
- [python-chess](https://github.com/niklasf/python-chess) — used by `training/gen_openings.py` to sample opening positions from a Polyglot book.
- An external Polyglot opening book (`training/book.bin`, locally named `gm2001.bin`) is used to generate `training/openings.epd` for local SPRT testing. **Provenance/license of this specific book file has not been verified** — confirm it before committing it to a public repository, or replace it with a book of known-clear provenance (e.g. a book generated from this project's own self-play games).

## Test/build dependencies

- GCC/G++ (C++20) for the engine.
- Rust (`rustc`) for the threat-contract dump tool (`bullet_integration/dump_threats.rs`) and the NapK9 trainer.
- Python 3 for `training/` tooling.
