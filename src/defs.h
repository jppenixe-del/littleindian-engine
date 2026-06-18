#pragma once
#include <cstdint>
#include <bit>
#include <cassert>

// ─── Core Enumerations ─────────────────────────────────────────────────────
enum class Color : int { WHITE = 0, BLACK = 1 };
enum class PieceType : int { PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5, NONE=6 };

constexpr Color operator~(Color c) { return Color(1 ^ int(c)); }

// ─── Square (0=A1, 7=H1, 56=A8, 63=H8) ────────────────────────────────────
class Square {
    int v;
public:
    constexpr Square() : v(64) {}
    explicit constexpr Square(int v) : v(v) {}
    constexpr int  value() const { return v; }
    constexpr bool isValid() const { return v >= 0 && v < 64; }
    constexpr int  file()  const { return v & 7; }
    constexpr int  rank()  const { return v >> 3; }
    constexpr bool operator==(Square o) const { return v == o.v; }
    constexpr bool operator!=(Square o) const { return v != o.v; }
};
static constexpr Square SQ_NONE(64);

// ─── Bitboard ──────────────────────────────────────────────────────────────
class Bitboard {
    uint64_t bb;
public:
    constexpr Bitboard() : bb(0) {}
    explicit constexpr Bitboard(uint64_t v) : bb(v) {}
    constexpr uint64_t value()    const { return bb; }
    constexpr bool     any()      const { return bb != 0; }
    constexpr bool     empty()    const { return bb == 0; }
    constexpr int      popcount() const { return std::popcount(bb); }

    Square poplsb() {
        int s = std::countr_zero(bb);
        bb &= bb - 1;
        return Square(s);
    }
    constexpr Square lsb() const { return Square(std::countr_zero(bb)); }

    static constexpr Bitboard fromSquare(Square sq)  { return Bitboard(1ULL << sq.value()); }
    static constexpr Bitboard fromSquare(int sq)     { return Bitboard(1ULL << sq); }

    constexpr bool test(Square sq) const { return (bb >> sq.value()) & 1; }
    constexpr bool test(int sq)    const { return (bb >> sq) & 1; }

    constexpr Bitboard operator|(Bitboard o)  const { return Bitboard(bb | o.bb); }
    constexpr Bitboard operator&(Bitboard o)  const { return Bitboard(bb & o.bb); }
    constexpr Bitboard operator^(Bitboard o)  const { return Bitboard(bb ^ o.bb); }
    constexpr Bitboard operator~()            const { return Bitboard(~bb); }
    constexpr Bitboard operator<<(int n)      const { return Bitboard(bb << n); }
    constexpr Bitboard operator>>(int n)      const { return Bitboard(bb >> n); }

    constexpr Bitboard& operator|=(Bitboard o) { bb |= o.bb; return *this; }
    constexpr Bitboard& operator&=(Bitboard o) { bb &= o.bb; return *this; }
    constexpr Bitboard& operator^=(Bitboard o) { bb ^= o.bb; return *this; }

    constexpr bool operator==(Bitboard o) const { return bb == o.bb; }
    constexpr bool operator!=(Bitboard o) const { return bb != o.bb; }
};

// Useful masks
static constexpr Bitboard BB_FILE_A(0x0101010101010101ULL);
static constexpr Bitboard BB_FILE_B(0x0202020202020202ULL);
static constexpr Bitboard BB_FILE_G(0x4040404040404040ULL);
static constexpr Bitboard BB_FILE_H(0x8080808080808080ULL);
static constexpr Bitboard BB_RANK_1(0x00000000000000FFULL);
static constexpr Bitboard BB_RANK_2(0x000000000000FF00ULL);
static constexpr Bitboard BB_RANK_7(0x00FF000000000000ULL);
static constexpr Bitboard BB_RANK_8(0xFF00000000000000ULL);
static constexpr Bitboard BB_ALL(0xFFFFFFFFFFFFFFFFULL);

// Named squares
static constexpr Square SQ_A1(0), SQ_B1(1), SQ_C1(2), SQ_D1(3);
static constexpr Square SQ_E1(4), SQ_F1(5), SQ_G1(6), SQ_H1(7);
static constexpr Square SQ_A8(56), SQ_B8(57), SQ_C8(58), SQ_D8(59);
static constexpr Square SQ_E8(60), SQ_F8(61), SQ_G8(62), SQ_H8(63);
