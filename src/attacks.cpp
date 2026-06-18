#include "attacks.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

#if defined(__BMI2__)
#include <immintrin.h>   // _pext_u64 (BMI2)
#define PEXT64(val, mask) _pext_u64((val), (mask))
#else
// Sem BMI2 (ex.: ARM/Android): emulação por software, só usada nas
// plataformas sem a instrução. Mais lenta, mas portável — usada tanto na
// construção das tabelas como no lookup em runtime.
static inline uint64_t pextSoftware(uint64_t val, uint64_t mask) {
    uint64_t res = 0;
    int bit = 0;
    while (mask) {
        uint64_t lsb = mask & (~mask + 1);
        if (val & lsb) res |= (1ULL << bit);
        mask &= mask - 1;
        ++bit;
    }
    return res;
}
#define PEXT64(val, mask) pextSoftware((val), (mask))
#endif

namespace attacks {

// ─── Lookup tables ─────────────────────────────────────────────────────────
static Bitboard gKnightTable[64];
static Bitboard gKingTable[64];
static Bitboard gPawnTable[2][64];

// PEXT-based sliding attacks
struct PextEntry {
    Bitboard mask;      // blocker mask (edges excluded)
    int      offset;    // index into shared table
};

static PextEntry gBishopEntry[64];
static PextEntry gRookEntry[64];

// Shared attack tables (indexed via PEXT)
// Rooks: max 2^12 = 4096 entries per square × 64 ≈ 800K entries worst case
// Actually sum of 2^popcount(mask) across all squares = 102400 for rooks, 5248 for bishops
static Bitboard gBishopTable[5248];
static Bitboard gRookTable[102400];

// ─── Slow ray-based attack generators (used only at init) ──────────────────
static Bitboard slowBishopAttacks(int sq, Bitboard occ) {
    Bitboard attacks;
    int f = sq & 7, r = sq >> 3;
    for (int d = 1; f+d < 8 && r+d < 8; ++d) {
        Square s(sq + d + d*8); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    for (int d = 1; f+d < 8 && r-d >= 0; ++d) {
        Square s(sq + d - d*8); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    for (int d = 1; f-d >= 0 && r+d < 8; ++d) {
        Square s(sq - d + d*8); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    for (int d = 1; f-d >= 0 && r-d >= 0; ++d) {
        Square s(sq - d - d*8); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    return attacks;
}

static Bitboard slowRookAttacks(int sq, Bitboard occ) {
    Bitboard attacks;
    int f = sq & 7, r = sq >> 3;
    for (int c = f+1; c < 8; ++c) {
        Square s(r*8+c); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    for (int c = f-1; c >= 0; --c) {
        Square s(r*8+c); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    for (int ro = r+1; ro < 8; ++ro) {
        Square s(ro*8+f); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    for (int ro = r-1; ro >= 0; --ro) {
        Square s(ro*8+f); attacks |= Bitboard::fromSquare(s);
        if (occ.test(s)) break;
    }
    return attacks;
}

// Build PEXT table for one piece type
static void buildPextTable(int sq, bool bishop, int& offset, Bitboard* table) {
    int f = sq & 7, r = sq >> 3;
    Bitboard mask;

    if (bishop) {
        // Diagonal mask (edges excluded) — use single distance variable d
        for (int d = 1; f+d <= 6 && r+d <= 6; ++d) mask |= Bitboard::fromSquare(sq + d + d*8);
        for (int d = 1; f+d <= 6 && r-d >= 1; ++d) mask |= Bitboard::fromSquare(sq + d - d*8);
        for (int d = 1; f-d >= 1 && r+d <= 6; ++d) mask |= Bitboard::fromSquare(sq - d + d*8);
        for (int d = 1; f-d >= 1 && r-d >= 1; ++d) mask |= Bitboard::fromSquare(sq - d - d*8);
    } else {
        // Rook mask (edges excluded in relevant directions)
        for (int c = f+1; c <= 6; ++c) mask |= Bitboard::fromSquare(r*8+c);
        for (int c = f-1; c >= 1; --c) mask |= Bitboard::fromSquare(r*8+c);
        for (int ro = r+1; ro <= 6; ++ro) mask |= Bitboard::fromSquare(ro*8+f);
        for (int ro = r-1; ro >= 1; --ro) mask |= Bitboard::fromSquare(ro*8+f);
    }

    int bits = mask.popcount();
    int size = 1 << bits;

    // Store entry
    if (bishop) {
        gBishopEntry[sq] = { mask, offset };
    } else {
        gRookEntry[sq] = { mask, offset };
    }

    // Enumerate all occupancy subsets and compute attacks
    uint64_t m = mask.value(), subset = 0;
    do {
        Bitboard occ(subset);
        Bitboard atk = bishop ? slowBishopAttacks(sq, occ) : slowRookAttacks(sq, occ);
        int idx = (int)PEXT64(subset, m);
        table[offset + idx] = atk;
        subset = (subset - m) & m;
    } while (subset);

    offset += size;
}

void init() {
    // ── Knight attacks ─────────────────────────────────────────────────────
    for (int sq = 0; sq < 64; ++sq) {
        uint64_t b = 1ULL << sq;
        uint64_t a = 0;
        if (!(b & 0x0101010101010101ULL)) {  // not file A
            if (sq + 15 < 64) a |= 1ULL << (sq+15);
            if (sq - 17 >= 0) a |= 1ULL << (sq-17);
        }
        if (!(b & 0x8080808080808080ULL)) {  // not file H
            if (sq + 17 < 64) a |= 1ULL << (sq+17);
            if (sq - 15 >= 0) a |= 1ULL << (sq-15);
        }
        if (!(b & 0x0303030303030303ULL)) {  // not file A or B
            if (sq + 6  < 64) a |= 1ULL << (sq+6);
            if (sq - 10 >= 0) a |= 1ULL << (sq-10);
        }
        if (!(b & 0xC0C0C0C0C0C0C0C0ULL)) {  // not file G or H
            if (sq + 10 < 64) a |= 1ULL << (sq+10);
            if (sq - 6  >= 0) a |= 1ULL << (sq-6);
        }
        gKnightTable[sq] = Bitboard(a);
    }

    // ── King attacks ───────────────────────────────────────────────────────
    for (int sq = 0; sq < 64; ++sq) {
        uint64_t b = 1ULL << sq, a = 0;
        // N, S
        if (sq + 8 < 64) a |= 1ULL << (sq+8);
        if (sq - 8 >= 0) a |= 1ULL << (sq-8);
        // E (not file H)
        if (!(b & 0x8080808080808080ULL)) {
            if (sq + 1 < 64) a |= 1ULL << (sq+1);
            if (sq + 9 < 64) a |= 1ULL << (sq+9);
            if (sq - 7 >= 0) a |= 1ULL << (sq-7);
        }
        // W (not file A)
        if (!(b & 0x0101010101010101ULL)) {
            if (sq - 1 >= 0) a |= 1ULL << (sq-1);
            if (sq + 7 < 64) a |= 1ULL << (sq+7);
            if (sq - 9 >= 0) a |= 1ULL << (sq-9);
        }
        gKingTable[sq] = Bitboard(a);
    }

    // ── Pawn attacks ───────────────────────────────────────────────────────
    for (int sq = 0; sq < 64; ++sq) {
        uint64_t b = 1ULL << sq;
        // White attacks: up-left and up-right
        uint64_t wa = 0;
        if (!(b & 0x0101010101010101ULL) && sq+7 < 64) wa |= 1ULL << (sq+7);
        if (!(b & 0x8080808080808080ULL) && sq+9 < 64) wa |= 1ULL << (sq+9);
        gPawnTable[0][sq] = Bitboard(wa);
        // Black attacks: down-left and down-right
        uint64_t ba = 0;
        if (!(b & 0x8080808080808080ULL) && sq-7 >= 0) ba |= 1ULL << (sq-7);
        if (!(b & 0x0101010101010101ULL) && sq-9 >= 0) ba |= 1ULL << (sq-9);
        gPawnTable[1][sq] = Bitboard(ba);
    }

    // ── Bishop PEXT tables ─────────────────────────────────────────────────
    int bishopOffset = 0;
    for (int sq = 0; sq < 64; ++sq)
        buildPextTable(sq, true, bishopOffset, gBishopTable);

    // ── Rook PEXT tables ───────────────────────────────────────────────────
    int rookOffset = 0;
    for (int sq = 0; sq < 64; ++sq)
        buildPextTable(sq, false, rookOffset, gRookTable);
}

// ─── Public attack functions ────────────────────────────────────────────────
template<Color C>
Bitboard pawnAttacks(Bitboard pawns) {
    if constexpr (C == Color::WHITE) {
        return ((pawns & ~BB_FILE_A) << 7) | ((pawns & ~BB_FILE_H) << 9);
    } else {
        return ((pawns & ~BB_FILE_H) >> 7) | ((pawns & ~BB_FILE_A) >> 9);
    }
}

// Explicit instantiations
template Bitboard pawnAttacks<Color::WHITE>(Bitboard);
template Bitboard pawnAttacks<Color::BLACK>(Bitboard);

Bitboard pawnAttackSq(Color c, Square sq) {
    return gPawnTable[int(c)][sq.value()];
}

Bitboard knightAttacks(Square sq) {
    return gKnightTable[sq.value()];
}

Bitboard bishopAttacks(Square sq, Bitboard occ) {
    const PextEntry& e = gBishopEntry[sq.value()];
    int idx = (int)PEXT64(occ.value(), e.mask.value());
    return gBishopTable[e.offset + idx];
}

Bitboard rookAttacks(Square sq, Bitboard occ) {
    const PextEntry& e = gRookEntry[sq.value()];
    int idx = (int)PEXT64(occ.value(), e.mask.value());
    return gRookTable[e.offset + idx];
}

Bitboard queenAttacks(Square sq, Bitboard occ) {
    return bishopAttacks(sq, occ) | rookAttacks(sq, occ);
}

Bitboard kingAttacks(Square sq) {
    return gKingTable[sq.value()];
}

} // namespace attacks
