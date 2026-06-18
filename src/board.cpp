#include "board.h"
#include <cstring>
#include <cstdio>
#include <cctype>
#include <sstream>

// ─── Zobrist ─────────────────────────────────────────────────────────────
namespace zobrist {
static uint64_t gPiece[2][6][64];
static uint64_t gSide;
static uint64_t gCastling[16];
static uint64_t gEp[8];

// Simple splitmix64 PRNG for reproducible key generation
static uint64_t splitmix(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void init() {
    uint64_t seed = 0xDEADBEEF42424242ULL;
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < 6; ++pt)
            for (int sq = 0; sq < 64; ++sq)
                gPiece[c][pt][sq] = splitmix(seed);
    gSide = splitmix(seed);
    for (int i = 0; i < 16; ++i) gCastling[i] = splitmix(seed);
    for (int i = 0; i < 8;  ++i) gEp[i]       = splitmix(seed);
}

uint64_t piece(Color c, PieceType pt, int sq) { return gPiece[int(c)][int(pt)][sq]; }
uint64_t side()              { return gSide; }
uint64_t castling(uint8_t r) { return gCastling[r & 15]; }
uint64_t ep(int file)        { return gEp[file & 7]; }
} // namespace zobrist

// ─── Internal helpers ───────────────────────────────────────────────────────
void Board::putPiece(Color c, PieceType pt, int sq) {
    Bitboard b = Bitboard::fromSquare(sq);
    pieceBB[int(c)][int(pt)] |= b;
    occupied[int(c)]          |= b;
    allOcc                    |= b;
    hash ^= zobrist::piece(c, pt, sq);
}

void Board::removePiece(Color c, PieceType pt, int sq) {
    Bitboard b = Bitboard::fromSquare(sq);
    pieceBB[int(c)][int(pt)] ^= b;
    occupied[int(c)]          ^= b;
    allOcc                    ^= b;
    hash ^= zobrist::piece(c, pt, sq);
}

void Board::movePiece(Color c, PieceType pt, int from, int to) {
    Bitboard fb = Bitboard::fromSquare(from);
    Bitboard tb = Bitboard::fromSquare(to);
    Bitboard both = fb | tb;
    pieceBB[int(c)][int(pt)] ^= both;
    occupied[int(c)]          ^= both;
    allOcc                    ^= both;
    hash ^= zobrist::piece(c, pt, from) ^ zobrist::piece(c, pt, to);
}

// ─── reset ─────────────────────────────────────────────────────────────────
void Board::reset() {
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < 6; ++pt)
            pieceBB[c][pt] = Bitboard();
    occupied[0] = occupied[1] = Bitboard();
    allOcc = Bitboard();
    stm = Color::WHITE;
    gamePly_ = 0;
    hash = 0;
    for (int i = 0; i < MAX_HIST; ++i) {
        history[i] = { 0, SQ_NONE, 0, PieceType::NONE, 0 };
    }
}

// ─── pieceOn / colorOn ─────────────────────────────────────────────────────
PieceType Board::pieceOn(int sq) const {
    Bitboard b = Bitboard::fromSquare(sq);
    for (int pt = 0; pt < 6; ++pt)
        if ((pieceBB[0][pt] | pieceBB[1][pt]).value() & b.value())
            return PieceType(pt);
    return PieceType::NONE;
}

Color Board::colorOn(int sq) const {
    Bitboard b = Bitboard::fromSquare(sq);
    if (occupied[0].value() & b.value()) return Color::WHITE;
    return Color::BLACK;
}

// ─── setFen ────────────────────────────────────────────────────────────────
bool Board::setFen(const std::string& fen) {
    reset();
    std::istringstream ss(fen);
    std::string board_str, stm_str, castle_str, ep_str;
    int halfmove = 0, fullmove = 1;
    ss >> board_str >> stm_str >> castle_str >> ep_str >> halfmove >> fullmove;

    // Parse board
    int sq = 56;  // start at A8
    for (char c : board_str) {
        if (c == '/') {
            sq -= 16;  // go to next rank down
        } else if (c >= '1' && c <= '8') {
            sq += c - '0';
        } else {
            Color col = std::isupper(c) ? Color::WHITE : Color::BLACK;
            PieceType pt = PieceType::NONE;
            switch (std::tolower(c)) {
                case 'p': pt = PieceType::PAWN;   break;
                case 'n': pt = PieceType::KNIGHT; break;
                case 'b': pt = PieceType::BISHOP; break;
                case 'r': pt = PieceType::ROOK;   break;
                case 'q': pt = PieceType::QUEEN;  break;
                case 'k': pt = PieceType::KING;   break;
                default: return false;
            }
            if (sq < 0 || sq >= 64) return false;
            putPiece(col, pt, sq);
            ++sq;
        }
    }

    // Side to move
    stm = (stm_str == "b") ? Color::BLACK : Color::WHITE;
    if (stm == Color::BLACK) hash ^= zobrist::side();

    // Castling rights
    uint8_t castling = 0;
    if (castle_str != "-") {
        for (char c : castle_str) {
            switch (c) {
                case 'K': castling |= 1; break;
                case 'Q': castling |= 2; break;
                case 'k': castling |= 4; break;
                case 'q': castling |= 8; break;
            }
        }
    }
    hash ^= zobrist::castling(castling);

    // En passant
    Square ep = SQ_NONE;
    if (ep_str != "-" && ep_str.size() >= 2) {
        int file = ep_str[0] - 'a';
        int rank = ep_str[1] - '1';
        if (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
            ep = Square(rank * 8 + file);
            hash ^= zobrist::ep(file);
        }
    }

    // Game ply: 2*(fullmove-1) + stm
    gamePly_ = 2 * (fullmove - 1) + (stm == Color::BLACK ? 1 : 0);
    if (gamePly_ < 0) gamePly_ = 0;
    if (gamePly_ >= MAX_HIST) gamePly_ = MAX_HIST - 1;

    history[gamePly_] = { castling, ep, halfmove, PieceType::NONE, hash };
    return true;
}

// ─── toFen ─────────────────────────────────────────────────────────────────
std::string Board::toFen() const {
    std::string result;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            PieceType pt = pieceOn(sq);
            if (pt == PieceType::NONE) {
                ++empty;
            } else {
                if (empty) { result += ('0' + empty); empty = 0; }
                Color col = colorOn(sq);
                const char* letters = "pnbrqk";
                char c = letters[int(pt)];
                if (col == Color::WHITE) c = std::toupper(c);
                result += c;
            }
        }
        if (empty) result += ('0' + empty);
        if (rank > 0) result += '/';
    }
    result += ' ';
    result += (stm == Color::WHITE ? 'w' : 'b');
    result += ' ';
    uint8_t cr = castlingRights();
    if (!cr) result += '-';
    else {
        if (cr & 1) result += 'K';
        if (cr & 2) result += 'Q';
        if (cr & 4) result += 'k';
        if (cr & 8) result += 'q';
    }
    result += ' ';
    Square ep = epSquare();
    if (!ep.isValid()) result += '-';
    else {
        result += ('a' + ep.file());
        result += ('1' + ep.rank());
    }
    result += ' ';
    result += std::to_string(halfmoveClock());
    result += ' ';
    result += std::to_string((gamePly_ / 2) + 1);
    return result;
}

// ─── canCastle ─────────────────────────────────────────────────────────────
bool Board::canCastle(Color c, bool kingside) const {
    uint8_t cr = castlingRights();
    if (c == Color::WHITE) return kingside ? (cr & 1) : (cr & 2);
    else                   return kingside ? (cr & 4) : (cr & 8);
}

// ─── isSquareAttacked ──────────────────────────────────────────────────────
bool Board::isSquareAttacked(int sq, Color byColor) const {
    Square s(sq);
    // Pawn
    if ((attacks::pawnAttackSq(~byColor, s) & pieceBB[int(byColor)][int(PieceType::PAWN)]).any())
        return true;
    // Knight
    if ((attacks::knightAttacks(s) & pieceBB[int(byColor)][int(PieceType::KNIGHT)]).any())
        return true;
    // Bishop/Queen (diagonals)
    Bitboard bq = pieceBB[int(byColor)][int(PieceType::BISHOP)]
                | pieceBB[int(byColor)][int(PieceType::QUEEN)];
    if ((attacks::bishopAttacks(s, allOcc) & bq).any()) return true;
    // Rook/Queen (files/ranks)
    Bitboard rq = pieceBB[int(byColor)][int(PieceType::ROOK)]
                | pieceBB[int(byColor)][int(PieceType::QUEEN)];
    if ((attacks::rookAttacks(s, allOcc) & rq).any()) return true;
    // King
    if ((attacks::kingAttacks(s) & pieceBB[int(byColor)][int(PieceType::KING)]).any())
        return true;
    return false;
}

// ─── makeMove ──────────────────────────────────────────────────────────────
void Board::makeMove(Move m) {
    const int from  = m.from();
    const int to    = m.to();
    const int flags = m.flags();
    const Color us  = stm;
    const Color them = ~us;

    // Save state
    BoardState& prev = history[gamePly_];
    if (gamePly_ + 1 >= MAX_HIST) return;  // safety
    BoardState& cur = history[gamePly_ + 1];

    cur.castling = prev.castling;
    cur.ep       = SQ_NONE;
    cur.halfmove = prev.halfmove + 1;
    cur.captured = PieceType::NONE;
    cur.hash     = hash;

    // Remove old EP from hash
    if (prev.ep.isValid()) hash ^= zobrist::ep(prev.ep.file());

    PieceType movingPt = pieceOn(from);

    // Handle captures
    if (flags & Move::FLAG_CAPTURE) {
        if (flags == Move::FLAG_EP) {
            // En passant: captured pawn is on different rank
            int capSq = (us == Color::WHITE) ? to - 8 : to + 8;
            cur.captured = PieceType::PAWN;
            removePiece(them, PieceType::PAWN, capSq);
        } else {
            cur.captured = pieceOn(to);
            removePiece(them, cur.captured, to);
        }
        cur.halfmove = 0;
    }

    // Move the piece
    if (flags == Move::FLAG_CASTLE_K || flags == Move::FLAG_CASTLE_Q) {
        // King move
        movePiece(us, PieceType::KING, from, to);
        // Rook move
        if (flags == Move::FLAG_CASTLE_K) {
            int rookFrom = (us == Color::WHITE) ? 7  : 63;
            int rookTo   = (us == Color::WHITE) ? 5  : 61;
            movePiece(us, PieceType::ROOK, rookFrom, rookTo);
        } else {
            int rookFrom = (us == Color::WHITE) ? 0  : 56;
            int rookTo   = (us == Color::WHITE) ? 3  : 59;
            movePiece(us, PieceType::ROOK, rookFrom, rookTo);
        }
    } else if (flags & Move::FLAG_PROMO_N && !(flags & Move::FLAG_CAPTURE)) {
        // Promotion (quiet)
        removePiece(us, PieceType::PAWN, from);
        putPiece(us, m.promoType(), to);
        cur.halfmove = 0;
    } else if (flags >= Move::FLAG_PROMO_N_CAP) {
        // Promotion with capture
        removePiece(us, PieceType::PAWN, from);
        putPiece(us, m.promoType(), to);
        cur.halfmove = 0;
    } else {
        movePiece(us, movingPt, from, to);
    }

    // Pawn resets halfmove
    if (movingPt == PieceType::PAWN) cur.halfmove = 0;

    // Double pawn push sets EP square
    if (flags == Move::FLAG_DOUBLE_PAWN) {
        int epSq = (us == Color::WHITE) ? to - 8 : to + 8;
        cur.ep = Square(epSq);
        hash ^= zobrist::ep(epSq & 7);
    }

    // Update castling rights
    uint8_t newCastle = cur.castling;
    // Remove rights if king or rook moved
    if (movingPt == PieceType::KING) {
        if (us == Color::WHITE) newCastle &= ~3u;
        else                    newCastle &= ~12u;
    }
    // Remove rights if rook moved or was captured
    static const uint8_t castleMask[64] = {
        (uint8_t)~2u, 255,255,255,(uint8_t)~3u, 255,255,(uint8_t)~1u,  // rank 1
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        (uint8_t)~8u,255,255,255,(uint8_t)~12u,255,255,(uint8_t)~4u  // rank 8
    };
    newCastle &= castleMask[from];
    newCastle &= castleMask[to];

    // Update castling hash
    if (newCastle != prev.castling) {
        hash ^= zobrist::castling(prev.castling) ^ zobrist::castling(newCastle);
        cur.castling = newCastle;
    }

    cur.hash = hash;

    // Flip side to move
    stm = them;
    hash ^= zobrist::side();
    ++gamePly_;
}

// ─── unmakeMove ────────────────────────────────────────────────────────────
void Board::unmakeMove(Move m) {
    --gamePly_;
    stm = ~stm;

    const int from  = m.from();
    const int to    = m.to();
    const int flags = m.flags();
    const Color us  = stm;
    const Color them = ~us;

    BoardState& cur = history[gamePly_ + 1];

    if (flags == Move::FLAG_CASTLE_K || flags == Move::FLAG_CASTLE_Q) {
        movePiece(us, PieceType::KING, to, from);
        if (flags == Move::FLAG_CASTLE_K) {
            int rookFrom = (us == Color::WHITE) ? 7  : 63;
            int rookTo   = (us == Color::WHITE) ? 5  : 61;
            movePiece(us, PieceType::ROOK, rookTo, rookFrom);
        } else {
            int rookFrom = (us == Color::WHITE) ? 0  : 56;
            int rookTo   = (us == Color::WHITE) ? 3  : 59;
            movePiece(us, PieceType::ROOK, rookTo, rookFrom);
        }
    } else if (flags & Move::FLAG_PROMO_N) {
        // Promotion: remove promoted piece, restore pawn
        removePiece(us, m.promoType(), to);
        putPiece(us, PieceType::PAWN, from);
    } else {
        movePiece(us, pieceOn(to), to, from);
    }

    // Restore captured piece
    if (cur.captured != PieceType::NONE) {
        if (flags == Move::FLAG_EP) {
            int capSq = (us == Color::WHITE) ? to - 8 : to + 8;
            putPiece(them, PieceType::PAWN, capSq);
        } else {
            putPiece(them, cur.captured, to);
        }
    }

    // Restore hash + state
    hash = history[gamePly_].hash;
}

// ─── null move (NMP) ───────────────────────────────────────────────────────
// Mesma contabilidade de hash/estado do makeMove/unmakeMove, sem mover peças:
// passa a vez, limpa o EP, conta para o halfmove clock.
void Board::makeNullMove() {
    BoardState& prev = history[gamePly_];
    if (gamePly_ + 1 >= MAX_HIST) return;  // safety
    BoardState& cur = history[gamePly_ + 1];

    cur.castling = prev.castling;
    cur.ep       = SQ_NONE;
    cur.halfmove = prev.halfmove + 1;
    cur.captured = PieceType::NONE;

    if (prev.ep.isValid()) hash ^= zobrist::ep(prev.ep.file());

    cur.hash = hash;
    stm = ~stm;
    hash ^= zobrist::side();
    ++gamePly_;
}

void Board::unmakeNullMove() {
    --gamePly_;
    stm = ~stm;
    hash = history[gamePly_].hash;
}

// ─── isLegal ───────────────────────────────────────────────────────────────
bool Board::isLegal(Move m) {
    makeMove(m);
    // Check if our king is in check after the move
    Square ks = kingSq(~stm);  // stm flipped after makeMove
    bool legal = !isSquareAttacked(ks.value(), stm);
    unmakeMove(m);
    return legal;
}

// ─── isRepetition ─────────────────────────────────────────────────────────
bool Board::isRepetition() const {
    int end = gamePly_;
    int start = std::max(0, end - halfmoveClock());
    for (int i = start; i < end - 1; i += 2) {
        if (history[i].hash == hash) return true;
    }
    return false;
}
