#pragma once
#include "board.h"

// Move list with score (for ordering)
struct MoveList {
    static constexpr int MAX_MOVES = 256;
    Move  moves[MAX_MOVES];
    int   count = 0;

    void add(Move m) { moves[count++] = m; }
    void clear()     { count = 0; }
};

// Generate all pseudo-legal moves (legal check done via isLegal on each move)
void generateMoves(const Board& board, MoveList& list);

// Generate only captures + promotions (for quiescence search)
void generateCaptures(const Board& board, MoveList& list);
