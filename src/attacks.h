#pragma once
#include "defs.h"

namespace attacks {

void init();  // call once at startup

// Pawn attacks (takes a bitboard of all pawns, returns all attacked squares)
template<Color C>
Bitboard pawnAttacks(Bitboard pawns);

// Single-square attacks
Bitboard knightAttacks(Square sq);
Bitboard bishopAttacks(Square sq, Bitboard occ);
Bitboard rookAttacks(Square sq, Bitboard occ);
Bitboard queenAttacks(Square sq, Bitboard occ);
Bitboard kingAttacks(Square sq);

// Precomputed single-square pawn attacks (for individual squares)
Bitboard pawnAttackSq(Color c, Square sq);

} // namespace attacks
