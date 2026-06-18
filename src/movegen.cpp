#include "movegen.h"

// ─── Helper: add promotions ─────────────────────────────────────────────────
static inline void addPromos(MoveList& list, int from, int to, bool capture) {
    uint16_t base = capture ? Move::FLAG_PROMO_N_CAP : Move::FLAG_PROMO_N;
    list.add(Move(from, to, base));       // knight
    list.add(Move(from, to, base + 1));   // bishop
    list.add(Move(from, to, base + 2));   // rook
    list.add(Move(from, to, base + 3));   // queen
}

// ─── generateMoves ─────────────────────────────────────────────────────────
void generateMoves(const Board& board, MoveList& list) {
    list.clear();
    const Color us   = board.stm;
    const Color them = ~us;
    const Bitboard myPieces  = board.occupied[int(us)];
    const Bitboard theirPcs  = board.occupied[int(them)];
    const Bitboard occ       = board.allOcc;
    const Bitboard empty     = ~occ;

    // ── Pawns ──────────────────────────────────────────────────────────────
    {
        Bitboard pawns = board.pieceBB[int(us)][int(PieceType::PAWN)];
        const bool white = (us == Color::WHITE);

        // Single push
        Bitboard singles = white ? (pawns << 8) & empty : (pawns >> 8) & empty;
        // Promotions on rank 8 (white) or rank 1 (black)
        Bitboard promoMask = white ? BB_RANK_8 : BB_RANK_1;
        Bitboard promoSingles = singles & promoMask;
        Bitboard normalSingles = singles & ~promoMask;

        Bitboard tmp = promoSingles;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 8 : to.value() + 8;
            addPromos(list, from, to.value(), false);
        }
        tmp = normalSingles;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 8 : to.value() + 8;
            list.add(Move(from, to.value(), Move::FLAG_QUIET));
        }

        // Double push (from rank 2 for white, rank 7 for black)
        Bitboard rank2 = white ? BB_RANK_2 : BB_RANK_7;
        Bitboard rank4 = white ? Bitboard(0x00000000FF000000ULL)
                                : Bitboard(0x000000FF00000000ULL);
        Bitboard doubles = white
            ? ((pawns & rank2) << 8) & empty
            : ((pawns & rank2) >> 8) & empty;
        doubles = white ? (doubles << 8) & empty & rank4
                        : (doubles >> 8) & empty & rank4;
        tmp = doubles;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 16 : to.value() + 16;
            list.add(Move(from, to.value(), Move::FLAG_DOUBLE_PAWN));
        }

        // Pawn captures
        Bitboard leftCap  = white
            ? ((pawns & ~BB_FILE_A) << 7) & theirPcs
            : ((pawns & ~BB_FILE_H) >> 7) & theirPcs;
        Bitboard rightCap = white
            ? ((pawns & ~BB_FILE_H) << 9) & theirPcs
            : ((pawns & ~BB_FILE_A) >> 9) & theirPcs;

        // Left captures
        tmp = leftCap & promoMask;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 7 : to.value() + 7;
            addPromos(list, from, to.value(), true);
        }
        tmp = leftCap & ~promoMask;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 7 : to.value() + 7;
            list.add(Move(from, to.value(), Move::FLAG_CAPTURE));
        }

        // Right captures
        tmp = rightCap & promoMask;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 9 : to.value() + 9;
            addPromos(list, from, to.value(), true);
        }
        tmp = rightCap & ~promoMask;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 9 : to.value() + 9;
            list.add(Move(from, to.value(), Move::FLAG_CAPTURE));
        }

        // En passant
        Square ep = board.epSquare();
        if (ep.isValid()) {
            Bitboard epBB = Bitboard::fromSquare(ep);
            Bitboard epLeft  = white
                ? ((pawns & ~BB_FILE_A) << 7) & epBB
                : ((pawns & ~BB_FILE_H) >> 7) & epBB;
            Bitboard epRight = white
                ? ((pawns & ~BB_FILE_H) << 9) & epBB
                : ((pawns & ~BB_FILE_A) >> 9) & epBB;
            if (epLeft.any()) {
                int from = white ? ep.value() - 7 : ep.value() + 7;
                list.add(Move(from, ep.value(), Move::FLAG_EP));
            }
            if (epRight.any()) {
                int from = white ? ep.value() - 9 : ep.value() + 9;
                list.add(Move(from, ep.value(), Move::FLAG_EP));
            }
        }
    }

    // ── Knights ────────────────────────────────────────────────────────────
    {
        Bitboard knights = board.pieceBB[int(us)][int(PieceType::KNIGHT)];
        while (knights.any()) {
            Square from = knights.poplsb();
            Bitboard targets = attacks::knightAttacks(from) & ~myPieces;
            while (targets.any()) {
                Square to = targets.poplsb();
                bool cap = (theirPcs & Bitboard::fromSquare(to)).any();
                list.add(Move(from.value(), to.value(), cap ? Move::FLAG_CAPTURE : Move::FLAG_QUIET));
            }
        }
    }

    // ── Bishops ────────────────────────────────────────────────────────────
    {
        Bitboard bishops = board.pieceBB[int(us)][int(PieceType::BISHOP)];
        while (bishops.any()) {
            Square from = bishops.poplsb();
            Bitboard targets = attacks::bishopAttacks(from, occ) & ~myPieces;
            while (targets.any()) {
                Square to = targets.poplsb();
                bool cap = (theirPcs & Bitboard::fromSquare(to)).any();
                list.add(Move(from.value(), to.value(), cap ? Move::FLAG_CAPTURE : Move::FLAG_QUIET));
            }
        }
    }

    // ── Rooks ──────────────────────────────────────────────────────────────
    {
        Bitboard rooks = board.pieceBB[int(us)][int(PieceType::ROOK)];
        while (rooks.any()) {
            Square from = rooks.poplsb();
            Bitboard targets = attacks::rookAttacks(from, occ) & ~myPieces;
            while (targets.any()) {
                Square to = targets.poplsb();
                bool cap = (theirPcs & Bitboard::fromSquare(to)).any();
                list.add(Move(from.value(), to.value(), cap ? Move::FLAG_CAPTURE : Move::FLAG_QUIET));
            }
        }
    }

    // ── Queens ─────────────────────────────────────────────────────────────
    {
        Bitboard queens = board.pieceBB[int(us)][int(PieceType::QUEEN)];
        while (queens.any()) {
            Square from = queens.poplsb();
            Bitboard targets = attacks::queenAttacks(from, occ) & ~myPieces;
            while (targets.any()) {
                Square to = targets.poplsb();
                bool cap = (theirPcs & Bitboard::fromSquare(to)).any();
                list.add(Move(from.value(), to.value(), cap ? Move::FLAG_CAPTURE : Move::FLAG_QUIET));
            }
        }
    }

    // ── King ───────────────────────────────────────────────────────────────
    {
        Square from = board.kingSq(us);
        Bitboard targets = attacks::kingAttacks(from) & ~myPieces;
        while (targets.any()) {
            Square to = targets.poplsb();
            bool cap = (theirPcs & Bitboard::fromSquare(to)).any();
            list.add(Move(from.value(), to.value(), cap ? Move::FLAG_CAPTURE : Move::FLAG_QUIET));
        }

        // Castling
        if (!board.isSquareAttacked(from.value(), them)) {
            if (board.canCastle(us, true)) {
                // Kingside: squares F, G must be empty and not attacked
                int f = (us == Color::WHITE) ? 5 : 61;
                int g = (us == Color::WHITE) ? 6 : 62;
                if (!occ.test(f) && !occ.test(g)
                    && !board.isSquareAttacked(f, them)
                    && !board.isSquareAttacked(g, them))
                {
                    list.add(Move(from.value(), g, Move::FLAG_CASTLE_K));
                }
            }
            if (board.canCastle(us, false)) {
                // Queenside: squares B, C, D must be empty; C, D not attacked
                int b = (us == Color::WHITE) ? 1 : 57;
                int c = (us == Color::WHITE) ? 2 : 58;
                int d = (us == Color::WHITE) ? 3 : 59;
                if (!occ.test(b) && !occ.test(c) && !occ.test(d)
                    && !board.isSquareAttacked(c, them)
                    && !board.isSquareAttacked(d, them))
                {
                    list.add(Move(from.value(), c, Move::FLAG_CASTLE_Q));
                }
            }
        }
    }
}

// ─── generateCaptures ──────────────────────────────────────────────────────
void generateCaptures(const Board& board, MoveList& list) {
    list.clear();
    const Color us   = board.stm;
    const Color them = ~us;
    const Bitboard theirPcs  = board.occupied[int(them)];
    const Bitboard occ       = board.allOcc;

    // Pawns: captures + en passant + promotions
    {
        Bitboard pawns = board.pieceBB[int(us)][int(PieceType::PAWN)];
        const bool white = (us == Color::WHITE);
        Bitboard promoMask = white ? BB_RANK_8 : BB_RANK_1;

        // Quiet promotions
        Bitboard singles = white ? (pawns << 8) & ~occ : (pawns >> 8) & ~occ;
        Bitboard tmp = singles & promoMask;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 8 : to.value() + 8;
            addPromos(list, from, to.value(), false);
        }

        // Capture + promo
        Bitboard leftCap  = white
            ? ((pawns & ~BB_FILE_A) << 7) & theirPcs
            : ((pawns & ~BB_FILE_H) >> 7) & theirPcs;
        Bitboard rightCap = white
            ? ((pawns & ~BB_FILE_H) << 9) & theirPcs
            : ((pawns & ~BB_FILE_A) >> 9) & theirPcs;

        tmp = leftCap;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 7 : to.value() + 7;
            if ((Bitboard::fromSquare(to) & promoMask).any()) addPromos(list, from, to.value(), true);
            else list.add(Move(from, to.value(), Move::FLAG_CAPTURE));
        }
        tmp = rightCap;
        while (tmp.any()) {
            Square to = tmp.poplsb();
            int from = white ? to.value() - 9 : to.value() + 9;
            if ((Bitboard::fromSquare(to) & promoMask).any()) addPromos(list, from, to.value(), true);
            else list.add(Move(from, to.value(), Move::FLAG_CAPTURE));
        }

        // EP
        Square ep = board.epSquare();
        if (ep.isValid()) {
            Bitboard epBB = Bitboard::fromSquare(ep);
            Bitboard epLeft  = white ? ((pawns & ~BB_FILE_A) << 7) & epBB
                                     : ((pawns & ~BB_FILE_H) >> 7) & epBB;
            Bitboard epRight = white ? ((pawns & ~BB_FILE_H) << 9) & epBB
                                     : ((pawns & ~BB_FILE_A) >> 9) & epBB;
            if (epLeft.any())  list.add(Move(white ? ep.value()-7 : ep.value()+7, ep.value(), Move::FLAG_EP));
            if (epRight.any()) list.add(Move(white ? ep.value()-9 : ep.value()+9, ep.value(), Move::FLAG_EP));
        }
    }

    // Knights, Bishops, Rooks, Queens, King - only captures
    auto addCaptures = [&](PieceType pt) {
        Bitboard pcs = board.pieceBB[int(us)][int(pt)];
        while (pcs.any()) {
            Square from = pcs.poplsb();
            Bitboard atk;
            switch (pt) {
                case PieceType::KNIGHT: atk = attacks::knightAttacks(from); break;
                case PieceType::BISHOP: atk = attacks::bishopAttacks(from, occ); break;
                case PieceType::ROOK:   atk = attacks::rookAttacks(from, occ); break;
                case PieceType::QUEEN:  atk = attacks::queenAttacks(from, occ); break;
                case PieceType::KING:   atk = attacks::kingAttacks(from); break;
                default: break;
            }
            Bitboard caps = atk & theirPcs;
            while (caps.any()) {
                Square to = caps.poplsb();
                list.add(Move(from.value(), to.value(), Move::FLAG_CAPTURE));
            }
        }
    };

    addCaptures(PieceType::KNIGHT);
    addCaptures(PieceType::BISHOP);
    addCaptures(PieceType::ROOK);
    addCaptures(PieceType::QUEEN);
    addCaptures(PieceType::KING);
}
