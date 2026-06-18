#pragma once
#include "defs.h"
#include "board.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>

// ─── TT Entry flags ──────────────────────────────────────────────────────
enum class Bound : uint8_t { NONE=0, EXACT=1, LOWER=2, UPPER=3 };

// ─── TT Entry (10 bytes packed into 12 with padding) ─────────────────────
struct TTEntry {
    uint32_t key32;    // upper 32 bits of hash (for verification)
    int16_t  score;    // centipawns
    int16_t  eval;     // static eval
    uint16_t move;     // best move data
    uint8_t  depth;    // depth stored
    uint8_t  flags;    // Bound + generation (upper 6 bits gen, lower 2 bits bound)

    Bound   bound()  const { return Bound(flags & 3); }
    uint8_t gen()    const { return flags >> 2; }
    bool    valid()  const { return key32 != 0; }
};
static_assert(sizeof(TTEntry) == 12);

// ─── Transposition Table ─────────────────────────────────────────────────
struct TT {
    static constexpr int DEFAULT_MB = 16;

    TTEntry* table = nullptr;
    size_t   size  = 0;
    uint8_t  gen   = 0;

    ~TT() { free(table); }

    void resize(int mb) {
        free(table);
        size = (size_t)(mb) * 1024 * 1024 / sizeof(TTEntry);
        // aligned_alloc exige (norma C11) que o tamanho seja múltiplo do
        // alinhamento — a glibc tolera quando não é, o bionic (Android) não.
        // sizeof(TTEntry)=12 não é múltiplo de 64, por isso arredonda-se o
        // tamanho em bytes para cima; size (nº de entradas) fica igual.
        size_t bytes = size * sizeof(TTEntry);
        bytes = (bytes + 63) & ~size_t(63);
        table = (TTEntry*)aligned_alloc(64, bytes);
        clear();
    }

    void clear() {
        if (table) memset(table, 0, size * sizeof(TTEntry));
        gen = 0;
    }

    void newSearch() { gen = (gen + 1) & 0x3F; }

    TTEntry* probe(uint64_t hash, bool& hit) const {
        size_t idx = (size_t)((uint32_t)hash) % size;
        TTEntry* e = &table[idx];
        hit = (e->key32 == (uint32_t)(hash >> 32)) && e->valid();
        return e;
    }

    void store(uint64_t hash, int score, int eval, Move m, int depth, Bound bound) {
        size_t idx = (size_t)((uint32_t)hash) % size;
        TTEntry* e = &table[idx];

        // Always replace if not occupied, or if new entry is deeper/fresher
        if (!e->valid() || e->gen() != gen || depth >= e->depth || bound == Bound::EXACT) {
            e->key32 = (uint32_t)(hash >> 32);
            e->score = (int16_t)score;
            e->eval  = (int16_t)eval;
            e->move  = m.data;
            e->depth = (uint8_t)depth;
            e->flags = (uint8_t)((gen << 2) | uint8_t(bound));
        }
    }
};

extern TT gTT;
