#pragma once
#include "board.h"
#include "tt.h"
#include <cstdint>
#include <atomic>
#include <string>

// Margem de segurança subtraída ao tempo disponível antes de calcular os limites de
// busca — cobre a latência de comunicação com a GUI/rede (gap: não existia nenhuma
// opção UCI equivalente a "Move Overhead", presente em todos os motores de referência).
extern int gMoveOverheadMs;

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
    int      selDepth  = 0;    // maior ply visitado nesta iteração (qsearch incluída) — campo UCI "seldepth"
};

// Score constants
static constexpr int INF_SCORE  = 32000;
static constexpr int MATE_SCORE = 31000;
static constexpr int MATE_IN(int n) { return MATE_SCORE - n; }
static inline bool   isMate(int s)  { return std::abs(s) >= MATE_SCORE - 512; }

void search(Board& board, const Limits& limits, uint64_t* nodesOut = nullptr);

// Pedido externo de parada (comando UCI "stop") — search() já lê o mesmo flag
// internamente via checkTime(), só faltava uma forma de o escrever de fora.
void requestStop();

// Parâmetros de busca afináveis por SPSA (training/spsa_tune.py), expostos
// como opções UCI "type spin". printTunableOptions() imprime as linhas
// "option name ..." (chamar de dentro do handler "uci"); setTunableParam
// tenta aplicar um "setoption" — devolve false se o nome não é um tunable
// (uci.cpp tenta isto como fallback depois das opções específicas).
void printTunableOptions();
bool setTunableParam(const std::string& name, int value);

// MultiPV (análise; OFF em jogo — default 1 = comportamento inalterado).
void setMultiPV(int n);

// Lazy SMP: nº de threads de busca (default 1 = comportamento inalterado,
// só a thread principal corre — ver "Threads" no UCI).
void setThreads(int n);

// Static Exchange Evaluation — exposta para validação (ver UCI "seetest").
bool seeGE(const Board& board, Move m, int threshold);
int  seeValue(const Board& board, Move m);

// Validação do acumulador incremental (ver UCI "incrtest"): percorre todos os
// lances legais até `depth` plies, comparando o eval pelo caminho NOVO
// (NapkAccSlot/EvalState) com o caminho ANTIGO (plyResolve, já confirmado por
// "threattest"/verifyFinny) em CADA nó. Devolve o nº de posições onde
// divergiram (0 = caminhos idênticos em toda a árvore percorrida).
int napkIncrementalSelfTest(Board& board, int depth);
