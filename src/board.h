#pragma once
#include "defs.h"
#include "attacks.h"
#include <string>
#include <cstdint>

// ─── Move (16-bit encoding) ──────────────────────────────────────────────────
// bits 0-5:   from square
// bits 6-11:  to square
// bits 12-15: flags
struct Move {
    uint16_t data = 0;

    static constexpr uint16_t FLAG_QUIET          = 0;
    static constexpr uint16_t FLAG_DOUBLE_PAWN    = 1;
    static constexpr uint16_t FLAG_CASTLE_K       = 2;
    static constexpr uint16_t FLAG_CASTLE_Q       = 3;
    static constexpr uint16_t FLAG_CAPTURE        = 4;
    static constexpr uint16_t FLAG_EP             = 5;
    static constexpr uint16_t FLAG_PROMO_N        = 8;
    static constexpr uint16_t FLAG_PROMO_B        = 9;
    static constexpr uint16_t FLAG_PROMO_R        = 10;
    static constexpr uint16_t FLAG_PROMO_Q        = 11;
    static constexpr uint16_t FLAG_PROMO_N_CAP    = 12;
    static constexpr uint16_t FLAG_PROMO_B_CAP    = 13;
    static constexpr uint16_t FLAG_PROMO_R_CAP    = 14;
    static constexpr uint16_t FLAG_PROMO_Q_CAP    = 15;

    constexpr Move() = default;
    explicit constexpr Move(uint16_t raw) : data(raw) {}
    constexpr Move(int from, int to, uint16_t flags)
        : data((uint16_t)((flags << 12) | (to << 6) | from)) {}

    constexpr int     from()    const { return data & 0x3F; }
    constexpr int     to()      const { return (data >> 6) & 0x3F; }
    constexpr int     flags()   const { return data >> 12; }
    constexpr bool    isNull()  const { return data == 0; }

    constexpr bool isCapture()   const { return (flags() & 4) != 0; }
    constexpr bool isPromo()     const { return (flags() & 8) != 0; }
    constexpr bool isEP()        const { return flags() == FLAG_EP; }
    constexpr bool isCastle()    const { return flags() == FLAG_CASTLE_K || flags() == FLAG_CASTLE_Q; }

    constexpr PieceType promoType() const {
        switch (flags() & 3) {
            case 0: return PieceType::KNIGHT;
            case 1: return PieceType::BISHOP;
            case 2: return PieceType::ROOK;
            default: return PieceType::QUEEN;
        }
    }

    constexpr bool operator==(Move o) const { return data == o.data; }
    constexpr bool operator!=(Move o) const { return data != o.data; }
};

static constexpr Move NULL_MOVE;

// ─── Board State (saved/restored on make/unmakeMove) ──────────────────────
struct BoardState {
    uint8_t   castling;   // WK=1 WQ=2 BK=4 BQ=8
    Square    ep;         // en passant target square (SQ_NONE if none)
    int       halfmove;   // 50-move counter
    PieceType captured;   // piece captured (NONE if not a capture)
    uint64_t  hash;       // Zobrist hash
};

// ─── Board ────────────────────────────────────────────────────────────────
class Board {
public:
    // Bitboard arrays — accessed by the eval
    Bitboard pieceBB[2][6];  // [Color][PieceType]
    Bitboard occupied[2];    // all pieces of each color
    Bitboard allOcc;         // combined occupancy

    Color    stm;            // side to move
    int      gamePly_;       // plies played (incremented on makeMove)
    uint64_t hash;           // Zobrist hash

    // Saved state stack
    static constexpr int MAX_HIST = 512;
    BoardState history[MAX_HIST];

    // ─── Interface required by nnue_net.cpp ────────────────────────────────
    Bitboard pieces(Color c, PieceType pt) const {
        return pieceBB[int(c)][int(pt)];
    }
    Bitboard allPieces() const { return allOcc; }
    Square   kingSq(Color c) const {
        return pieceBB[int(c)][int(PieceType::KING)].lsb();
    }
    Color sideToMove() const { return stm; }
    int   gamePly()    const { return gamePly_; }

    // ─── Board operations ──────────────────────────────────────────────────
    void reset();
    bool setFen(const std::string& fen);
    std::string toFen() const;

    void makeMove(Move m);
    void unmakeMove(Move m);
    void makeNullMove();
    void unmakeNullMove();

    // Piece on a square
    PieceType pieceOn(int sq) const;
    PieceType pieceOn(Square sq) const { return pieceOn(sq.value()); }
    Color     colorOn(int sq) const;

    // Castling helpers
    bool canCastle(Color c, bool kingside) const;
    uint8_t castlingRights() const { return history[gamePly_].castling; }
    Square  epSquare()       const { return history[gamePly_].ep; }
    int     halfmoveClock()  const { return history[gamePly_].halfmove; }

    // Attack/check queries
    bool isSquareAttacked(int sq, Color byColor) const;
    bool isInCheck() const { return isSquareAttacked(kingSq(stm).value(), ~stm); }
    bool isLegal(Move m);

    // Perft helpers
    bool isRepetition() const;

private:
    void putPiece(Color c, PieceType pt, int sq);
    void removePiece(Color c, PieceType pt, int sq);
    void movePiece(Color c, PieceType pt, int from, int to);
};

// ─── Zobrist ─────────────────────────────────────────────────────────────
namespace zobrist {
void     init();
uint64_t piece(Color c, PieceType pt, int sq);
uint64_t side();     // XOR when it's black to move (or flip stm)
uint64_t castling(uint8_t rights);
uint64_t ep(int file);
} // namespace zobrist
