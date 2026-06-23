# littleindian

**littleindian** is a UCI chess engine written from scratch. It has two evaluations: an original neural network (NNUE, **NapK9** format, still under active redesign) and a complete handcrafted evaluation (HCE) tuned via Texel tuning, selectable at runtime with `setoption name EvalFile value none`. Board representation, move generation, search, transposition table, time management and UCI are all original code.

> **Status: early development.** The engine is in Phase F1/F2 of its roadmap (see [Roadmap](#roadmap) below): move generation is perft-verified and a minimal search exists, but search techniques beyond a basic PVS are still being added and validated one at a time. Not yet tournament-ready.

Other languages: [Português](README.pt.md) · [Français](README.fr.md)

---

## Two evaluations

- **NNUE (NapK9)**: custom format, under active redesign — the concrete architecture (layer sizes, buckets, features) changes between training sessions, so it isn't documented here in detail to avoid stale information. Active by default when a network is embedded in the binary.
- **HCE**: PSQT + Material + Threats + Mobility + KingSafety + PawnStructure, all jointly tuned via Texel tuning over real positions (Stockfish-format binpacks). Active with `setoption name EvalFile value none`. Used as a comparison baseline and in calibration experiments run in parallel to the network's.

Every search parameter (margins, reductions, time management) is tuned by SPSA/SPRT, not inherited from another engine. See [`docs/CLAUDE.md`](docs/CLAUDE.md) for the full design rationale.

## Building

The network is compiled directly into the binary via `.incbin` — there is no `.napk9` file read at runtime.

```bash
# core engine only, no embedded net (perft, development builds)
make f1

# release build for the host CPU, with the network embedded
make native-embed NET=nets/littleindian_1024.napk9

# AVX2-specific build
make avx2-embed NET=nets/littleindian_1024.napk9
```

Always pass `NET=nets/littleindian_1024.napk9` explicitly — the Makefile's default points at a different network used during development.

## Usage (UCI)

Standard UCI commands (`position`, `go`, `isready`, `setoption`, `quit`) plus a few engine-specific ones:

| Command | Purpose |
|---|---|
| `perft N` | Move generator correctness/speed check to depth N |
| `eval` | Static evaluation of the current position, side-to-move POV (NNUE only) |
| `bench` | Fixed search over a built-in position set — used as the nodes/nps regression signature |
| `threattest` | Validates the incremental accumulator against a full rebuild (must report `diff = 0`) |
| `d` | Prints the board and FEN |

## Testing changes (SPRT)

No search technique is kept unless it gains Elo, measured with [cutechess-cli](https://github.com/cutechess/cutechess) SPRT, never assumed. See [`training/sprt.py`](training/sprt.py):

```bash
# generate an opening set from a Polyglot book (once)
python3 training/gen_openings.py --book training/book.bin --out training/openings.epd --n 200 --plies 8

# STC triage
python3 training/sprt.py --new ./littleindian.candidate --base ./littleindian.stable --tc 8+0.08

# LTC confirmation of anything that passed STC
python3 training/sprt.py --new ./littleindian.candidate --base ./littleindian.stable --tc 40+0.4
```

Threat-contract validation before trusting a network:

```bash
rustc -O bullet_integration/dump_threats.rs -o dump_threats
python3 training/calibra_threats.py --engine ./littleindian --rust ./dump_threats   # must report 0 divergences
```

## Project structure

```
littleindian/
├── src/
│   ├── board.{h,cpp}, attacks.{h,cpp}, movegen.{h,cpp}   # board, bitboards, magic attacks
│   ├── search.{h,cpp}, tt.h                              # PVS/ID/qsearch, transposition table
│   ├── uci.{h,cpp}, main.cpp                             # UCI front-end
│   └── napoleon/                                          # NapK9 eval module (nnue_net, embedded_net)
├── nets/littleindian_1024.napk9   # network embedded into release binaries
├── bullet_integration/            # NapK9 trainer (Rust) and reference material
├── training/                      # SPRT harness, opening book tooling, threat-contract validator
└── docs/                          # design rationale and integration roadmap
```

## Roadmap

The engine is built up in gated phases — each phase has a concrete pass/fail condition before the next one starts. Search techniques themselves are introduced one at a time from [`docs/INTEGRACAO_BLOCOS.md`](docs/INTEGRACAO_BLOCOS.md), each validated by SPRT.

- **F1 — Foundations**: bitboards, magic move generation, Chess960, perft. ✅
- **F2 — NapK9 eval + minimal search**: embedded network, incremental accumulator, PVS/ID/qsearch/TT. ✅ (in progress: move ordering)
- **F3 — Modern search**: move ordering (hash move, MVV-LVA/SEE, killers, history), pruning/reductions (RFP, NMP, LMR, futility, LMP, SEE pruning, IIR) — each gated by SPRT.
- **F4 — Extensions & correction**: singular extensions, ProbCut, correction history, Lazy SMP.
- **F5 — Tuning & release**: full-engine SPSA, PGO build, regression suite, gauntlet.

Full detail in [`docs/CLAUDE.md`](docs/CLAUDE.md) and [`docs/INTEGRACAO_BLOCOS.md`](docs/INTEGRACAO_BLOCOS.md).

## License

littleindian is licensed under the [GNU General Public License v3.0](LICENSE).

See [CREDITS.md](CREDITS.md) for third-party tools used and conceptual references credited.
