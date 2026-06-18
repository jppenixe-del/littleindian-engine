#pragma once
#include "board.h"
#include "tt.h"
#include <cstdint>
#include <atomic>

// Search limits
struct Limits {
    int  depth    = 64;
    int  movetime = 0;   // ms, 0 = no limit
    int  wtime    = 0, btime = 0;
    int  winc     = 0, binc  = 0;
    int  movestogo = 0;
    bool infinite = false;
    uint64_t nodes = 0;  // 0 = no limit
};

// Per-search statistics
struct SearchInfo {
    uint64_t nodes    = 0;
    bool     stopped  = false;
    int64_t  startMs  = 0;
    int64_t  timeLimitMs = 0;  // hard stop
    int64_t  softLimitMs = 0;  // soft stop (check after root)
    uint64_t nodeLimit  = 0;   // "go nodes N" (0 = sem limite)
};

// Score constants
static constexpr int INF_SCORE  = 32000;
static constexpr int MATE_SCORE = 31000;
static constexpr int MATE_IN(int n) { return MATE_SCORE - n; }
static inline bool   isMate(int s)  { return std::abs(s) >= MATE_SCORE - 512; }

void search(Board& board, const Limits& limits);

// Static Exchange Evaluation — exposta para validação (ver UCI "seetest").
bool seeGE(const Board& board, Move m, int threshold);
int  seeValue(const Board& board, Move m);

// Validação do acumulador incremental (ver UCI "incrtest"): percorre todos os
// lances legais até `depth` plies, comparando o eval pelo caminho NOVO
// (NapkAccSlot/EvalState) com o caminho ANTIGO (plyResolve, já confirmado por
// "threattest"/verifyFinny) em CADA nó. Devolve o nº de posições onde
// divergiram (0 = caminhos idênticos em toda a árvore percorrida).
int napkIncrementalSelfTest(Board& board, int depth);
