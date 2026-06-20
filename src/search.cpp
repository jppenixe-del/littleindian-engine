#include "search.h"
#include "movegen.h"
#include "napoleon/nnue_net.h"
#include "napoleon/wdl_brain.h"
#include "napoleon/wdl_model.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <atomic>
#include <thread>
#include <vector>
#include <memory>

int gMoveOverheadMs = 10;

TT gTT;

// ─── Lazy SMP ────────────────────────────────────────────────────────────
// TT partilhada SEM locks (mesma filosofia do Stockfish/Reckless/Coda: a
// raciness ocasional é aceite — o key32 da TT já filtra a maioria do lixo,
// e um nó com info errada é só re-verificado pela própria busca, nunca
// confiado às cegas). O que TEM de ser por-thread (já marcado thread_local
// acima: gHistory, gCaptureHistory, gContHist1/2, gKillers, gContPieceAt/
// ToAt, gNmpMinPly, gOptimism, gEvalSlots, gRootMoves*, gRootExcluded*) é
// o estado de ordenação/ply que, ao contrário da TT, não tolera mistura
// entre threads (corromperia a própria heurística, não só "ruído").
// gPawnCorrHist fica partilhado de propósito — é uma tabela aprendida ao
// longo do jogo, não específica de um caminho de busca.
static std::atomic<bool> gGlobalStop{false};
static int gThreads = 1;
void setThreads(int n) { gThreads = std::max(1, n); }

// ─── Time helpers ──────────────────────────────────────────────────────────
static int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static bool checkTime(SearchInfo& info) {
    if (info.stopped) return true;
    if (gGlobalStop.load(std::memory_order_relaxed)) {
        info.stopped = true;
        return true;
    }
    if (info.nodeLimit > 0 && info.nodes >= info.nodeLimit) {
        info.stopped = true;
        return true;
    }
    if ((info.nodes & 2047) == 0) {
        if (info.timeLimitMs > 0 && nowMs() >= info.startMs + info.timeLimitMs) {
            info.stopped = true;
            return true;
        }
    }
    return false;
}

// Declarada mais abaixo (junto às outras funções de ataque/SEE); forward só p/ o
// fallback material+threats de staticEval poder reutilizá-la sem reordenar o ficheiro.
static Bitboard computeSideAttacks(const Board& board, Color side);

// ─── Eval ─────────────────────────────────────────────────────────────────
static int staticEval(const Board& board) {
    int score;
    if (napoleon::nnue::isLoaded()) {
        score = napoleon::nnue::evaluate(board);
    } else {
        // Material fallback (used when no NNUE net loaded), + um termo de threats
        // calculado on-the-fly (pedido para diagnóstico: isolar busca vs rede com uma
        // eval um pouco mais rica que material puro, sem precisar da NNUE). Para cada
        // peça atacada pelo adversário e NÃO defendida de volta por nós, penaliza uma
        // fração do valor dela (proxy simples de "pendurada", sem SEE completo).
        static const int pv[] = { 100, 325, 325, 500, 975, 20000, 0 };
        int s = 0;
        for (int pt = 0; pt < 5; ++pt) {
            s += board.pieceBB[0][pt].popcount() * pv[pt];
            s -= board.pieceBB[1][pt].popcount() * pv[pt];
        }
        Bitboard whiteAttacks = computeSideAttacks(board, Color::WHITE);
        Bitboard blackAttacks = computeSideAttacks(board, Color::BLACK);
        for (int pt = 0; pt < 5; ++pt) {
            Bitboard whitePieces = board.pieceBB[0][pt];
            while (whitePieces.any()) {
                int sq = whitePieces.poplsb().value();
                if (blackAttacks.test(sq) && !whiteAttacks.test(sq)) s -= pv[pt] / 8;
            }
            Bitboard blackPieces = board.pieceBB[1][pt];
            while (blackPieces.any()) {
                int sq = blackPieces.poplsb().value();
                if (whiteAttacks.test(sq) && !blackAttacks.test(sq)) s += pv[pt] / 8;
            }
        }
        score = board.sideToMove() == Color::WHITE ? s : -s;
    }
    // Escala pelo halfmove clock: aproxima a regra dos 50 lances — a eval
    // perde força à medida que o contador sobe (posição a tender a empate).
    score -= score * board.halfmoveClock() / 199;
    return score;
}

// ─── Correction History (peões) ────────────────────────────────────────
// Corrige a eval estática com o erro médio observado entre eval e o
// resultado real da busca, indexado pela estrutura de peões — a eval erra
// de forma sistemática em certas estruturas (peões passados, bloqueados,
// etc.) e isto aprende esse desvio ao longo da partida/teste.
static constexpr int CORR_HIST_SIZE = 16384;
static constexpr int CORR_HIST_GRAIN = 256;  // escala interna da tabela
static constexpr int CORR_HIST_MAX   = 1024; // limite do termo de correção, em cp
static int gPawnCorrHist[2][CORR_HIST_SIZE];

static uint64_t pawnKey(const Board& board) {
    uint64_t key = 0;
    Bitboard wp = board.pieces(Color::WHITE, PieceType::PAWN);
    Bitboard bp = board.pieces(Color::BLACK, PieceType::PAWN);
    while (wp.any()) key ^= zobrist::piece(Color::WHITE, PieceType::PAWN, wp.poplsb().value());
    while (bp.any()) key ^= zobrist::piece(Color::BLACK, PieceType::PAWN, bp.poplsb().value());
    return key;
}

static int pawnCorrTerm(const Board& board) {
    int idx = (int)(pawnKey(board) % CORR_HIST_SIZE);
    return gPawnCorrHist[int(board.sideToMove())][idx] / CORR_HIST_GRAIN;
}

static void updateCorrHist(const Board& board, int rawEval, int bestScore) {
    int idx = (int)(pawnKey(board) % CORR_HIST_SIZE);
    int& e = gPawnCorrHist[int(board.sideToMove())][idx];
    int diff = (bestScore - rawEval) * CORR_HIST_GRAIN;
    e += (diff - e) / 32;  // média móvel exponencial
    int limit = CORR_HIST_MAX * CORR_HIST_GRAIN;
    if (e > limit) e = limit;
    if (e < -limit) e = -limit;
}

// ─── Correction History (não-peão, por cor) ────────────────────────────────
// Mesma ideia do pawnCorrHist, mas indexada pelo material NÃO-PEÃO de cada
// lado (cavalos/bispos/torres/damas/rei) — a eval erra de forma sistemática
// também por desequilíbrios materiais (ex.: 2 menores vs torre), não só por
// estrutura de peões. SF e Reckless têm AMBOS este 2º bucket (não tínhamos —
// já estava nomeado no nosso INTEGRACAO_BLOCOS.md, nunca implementado).
// Duas tabelas (uma por cor do material-chave), cada indexada também pela
// perspetiva (quem tem a vez) — a correção final soma as duas.
static int gNonPawnCorrHist[2][2][CORR_HIST_SIZE];  // [perspetiva][cor do material][idx]

static uint64_t nonPawnKey(const Board& board, Color c) {
    uint64_t key = 0;
    for (PieceType pt : {PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN, PieceType::KING}) {
        Bitboard bb = board.pieces(c, pt);
        while (bb.any()) key ^= zobrist::piece(c, pt, bb.poplsb().value());
    }
    return key;
}

static int nonPawnCorrTerm(const Board& board) {
    int stm = int(board.sideToMove());
    int idxW = (int)(nonPawnKey(board, Color::WHITE) % CORR_HIST_SIZE);
    int idxB = (int)(nonPawnKey(board, Color::BLACK) % CORR_HIST_SIZE);
    // 🦅 FIX: dividia por 2×GRAIN (média dos dois buckets de cor) — confirmado lendo o
    // código real do SF e do Reckless que AMBOS SOMAM os dois buckets (branco+preto) e
    // dividem só por UM grain, igual ao pawnCorrTerm. A nossa versão estava a sub-pesar
    // o sinal não-pawn a metade do que devia relativamente ao pawn -- bug de escala, não
    // de sinal, mas real e novo de hoje (a tabela foi adicionada hoje).
    return (gNonPawnCorrHist[stm][0][idxW] + gNonPawnCorrHist[stm][1][idxB]) / CORR_HIST_GRAIN;
}

static void updateNonPawnCorrHist(const Board& board, int rawEval, int bestScore) {
    int stm = int(board.sideToMove());
    int diff = (bestScore - rawEval) * CORR_HIST_GRAIN;
    int limit = CORR_HIST_MAX * CORR_HIST_GRAIN;
    int idxW = (int)(nonPawnKey(board, Color::WHITE) % CORR_HIST_SIZE);
    int& eW = gNonPawnCorrHist[stm][0][idxW];
    eW += (diff - eW) / 32;
    if (eW > limit) eW = limit;
    if (eW < -limit) eW = -limit;
    int idxB = (int)(nonPawnKey(board, Color::BLACK) % CORR_HIST_SIZE);
    int& eB = gNonPawnCorrHist[stm][1][idxB];
    eB += (diff - eB) / 32;
    if (eB > limit) eB = limit;
    if (eB < -limit) eB = -limit;
}

// ─── Move ordering ──────────────────────────────────────────────────────
static const int kPieceValue[6] = { 100, 325, 325, 500, 975, 20000 };
static int DELTA_MARGIN = 352;  // Coda QS_DELTA_MARGIN (OUTPUT_SCALE_CP=400 now, no 408/400 rescale needed)
static thread_local int gHistory[2][64][64];
static thread_local int gCaptureHistory[2][6][6];   // [lado][atacante][vítima] — bónus/malus de capturas

// ─── Low-ply history ─────────────────────────────────────────────────────
// Gap vs SF: tabela ADICIONAL de history só para os primeiros plies (perto da raiz),
// somada ao history normal — sinal extra de ordenação onde mais conta (decisões perto
// da raiz afetam a árvore inteira). Influência decai com 1/(1+ply): forte em ply=0,
// fraca perto do limite LOW_PLY_MAX. Mesmo esquema de bónus/malus do gHistory normal.
static constexpr int LOW_PLY_MAX = 8;
static thread_local int gLowPlyHistory[LOW_PLY_MAX][64][64];

// ─── Material/Score Optimism ────────────────────────────────────────────
// Enviesa a eval a favor de quem está a ganhar na tendência da busca
// (média do score entre profundidades) — incentiva a pressionar vantagem,
// desincentiva otimismo quando a tendência é negativa. Fixo durante cada
// profundidade da busca iterativa (atualizado entre profundidades).
static thread_local int gOptimism[2] = {0, 0};

// ─── Threat-aware ordering (gap vs Reckless) ────────────────────────────────
// Todas as casas atacadas por UM lado (união sobre todas as peças desse lado) — usado
// para um bónus/malus de ordenação de quietos: lance que ESCAPA de uma casa atacada
// recebe bónus, lance que entra numa casa atacada (vindo de uma casa segura) recebe
// malus. Versão mais simples e segura do que reindexar gHistory com bits de ameaça
// (Reckless faz isso); aqui é só um termo aditivo, sem tocar nas tabelas existentes.
// 🦅 Cobertura completa (peão/cavalo/bispo/torre/dama/rei) — confirmado no código real
// do Reckless (src/board.rs::update_threats(), src/board/makemove.rs) que ele TAMBÉM
// inclui peças deslizantes, sem as evitar por custo. A diferença real não é "menos
// peças", é QUANDO/COMO computam: o Reckless calcula isto UMA VEZ por make_move() (não
// por nó de ordenação) e guarda no próprio estado do tabuleiro (pilha, junto com
// Zobrist/direitos de roque), reaproveitando o mesmo cálculo que pins/checkers/check
// extensions já precisam de qualquer forma — não é trabalho extra, é o MESMO trabalho
// partilhado. Aqui calcula-se uma vez por nó (mesma frequência: SortedMoves::init() só
// corre uma vez por nó também), por isso a frequência já bate — falta só decidir, via
// SPRT, se o custo do nosso loop por peça (sem o gerador setwise SIMD do Reckless) vale
// a pena pela ordenação melhor.
static Bitboard computeSideAttacks(const Board& board, Color side) {
    Bitboard occ = board.allOcc;
    Bitboard atk = (side == Color::WHITE) ? attacks::pawnAttacks<Color::WHITE>(board.pieces(side, PieceType::PAWN))
                                           : attacks::pawnAttacks<Color::BLACK>(board.pieces(side, PieceType::PAWN));
    Bitboard bb = board.pieces(side, PieceType::KNIGHT);
    while (bb.any()) atk |= attacks::knightAttacks(bb.poplsb());
    bb = board.pieces(side, PieceType::BISHOP);
    while (bb.any()) atk |= attacks::bishopAttacks(bb.poplsb(), occ);
    bb = board.pieces(side, PieceType::ROOK);
    while (bb.any()) atk |= attacks::rookAttacks(bb.poplsb(), occ);
    bb = board.pieces(side, PieceType::QUEEN);
    while (bb.any()) { Square qs = bb.poplsb(); atk |= attacks::bishopAttacks(qs, occ) | attacks::rookAttacks(qs, occ); }
    atk |= attacks::kingAttacks(board.kingSq(side));
    return atk;
}

// ─── Static Exchange Evaluation (SEE) ──────────────────────────────────────
// Todas as peças (ambas as cores) que atacam `sq`, dada uma ocupação
// arbitrária `occ` (usado para simular peças removidas durante a troca).
static Bitboard attackersTo(const Board& board, int sq, Bitboard occ) {
    Bitboard attackers;
    attackers |= attacks::pawnAttackSq(Color::BLACK, Square(sq)) & board.pieces(Color::WHITE, PieceType::PAWN);
    attackers |= attacks::pawnAttackSq(Color::WHITE, Square(sq)) & board.pieces(Color::BLACK, PieceType::PAWN);
    attackers |= attacks::knightAttacks(Square(sq)) &
                 (board.pieces(Color::WHITE, PieceType::KNIGHT) | board.pieces(Color::BLACK, PieceType::KNIGHT));
    attackers |= attacks::kingAttacks(Square(sq)) &
                 (board.pieces(Color::WHITE, PieceType::KING) | board.pieces(Color::BLACK, PieceType::KING));
    Bitboard bishops = board.pieces(Color::WHITE, PieceType::BISHOP) | board.pieces(Color::BLACK, PieceType::BISHOP)
                      | board.pieces(Color::WHITE, PieceType::QUEEN)  | board.pieces(Color::BLACK, PieceType::QUEEN);
    attackers |= attacks::bishopAttacks(Square(sq), occ) & bishops;
    Bitboard rooks   = board.pieces(Color::WHITE, PieceType::ROOK)   | board.pieces(Color::BLACK, PieceType::ROOK)
                      | board.pieces(Color::WHITE, PieceType::QUEEN)  | board.pieces(Color::BLACK, PieceType::QUEEN);
    attackers |= attacks::rookAttacks(Square(sq), occ) & rooks;
    return attackers & occ;
}

// Devolve true se o valor da troca (jogada bem capturada até ao fim, ambos
// os lados a jogar de forma otima) for >= threshold. Algoritmo clássico de
// swap-list (Stockfish/Reckless/Coda); atenção à contabilidade da promoção
// dentro do ciclo — usa-se o valor simples da peça (não o ganho da
// promoção), que é a forma correta (ver nota no Coda: a forma alternativa
// inverte o veredito em recapturas de promoção).
bool seeGE(const Board& board, Move m, int threshold) {
    if (m.isCastle()) return 0 >= threshold;

    int from = m.from(), to = m.to();
    PieceType targetPt   = board.pieceOn(to);
    PieceType attackerPt = board.pieceOn(from);
    bool isPromo = m.isPromo();

    int balance = m.isEP() ? kPieceValue[int(PieceType::PAWN)]
                : (targetPt != PieceType::NONE) ? kPieceValue[int(targetPt)]
                : 0;

    if (isPromo)
        balance += kPieceValue[int(m.promoType())] - kPieceValue[int(PieceType::PAWN)];

    balance -= threshold;
    if (balance < 0) return false;

    int riskValue = isPromo ? kPieceValue[int(m.promoType())] : kPieceValue[int(attackerPt)];
    balance -= riskValue;
    if (balance >= 0) return true;

    Bitboard occ = board.allOcc ^ Bitboard::fromSquare(from);
    if (m.isEP()) {
        int epVictimSq = (to & 7) | (from & ~7);
        occ ^= Bitboard::fromSquare(epVictimSq);
    }

    Bitboard bishops = board.pieces(Color::WHITE, PieceType::BISHOP) | board.pieces(Color::BLACK, PieceType::BISHOP)
                      | board.pieces(Color::WHITE, PieceType::QUEEN)  | board.pieces(Color::BLACK, PieceType::QUEEN);
    Bitboard rooks   = board.pieces(Color::WHITE, PieceType::ROOK)   | board.pieces(Color::BLACK, PieceType::ROOK)
                      | board.pieces(Color::WHITE, PieceType::QUEEN)  | board.pieces(Color::BLACK, PieceType::QUEEN);

    Color stm = ~board.sideToMove();
    Bitboard attackers = attackersTo(board, to, occ);

    while (true) {
        Bitboard stmAttackers = attackers & board.occupied[int(stm)];
        if (stmAttackers.empty()) break;

        PieceType lvaPt = PieceType::NONE;
        int lvaSq = -1;
        for (int pt = 0; pt < 6; ++pt) {
            Bitboard bb = board.pieces(stm, PieceType(pt)) & stmAttackers;
            if (bb.any()) { lvaPt = PieceType(pt); lvaSq = bb.lsb().value(); break; }
        }

        occ ^= Bitboard::fromSquare(lvaSq);
        if (lvaPt == PieceType::PAWN || lvaPt == PieceType::BISHOP || lvaPt == PieceType::QUEEN)
            attackers |= attacks::bishopAttacks(Square(to), occ) & bishops;
        if (lvaPt == PieceType::ROOK || lvaPt == PieceType::QUEEN)
            attackers |= attacks::rookAttacks(Square(to), occ) & rooks;
        attackers &= occ;

        stm = ~stm;
        balance = -balance - 1 - kPieceValue[int(lvaPt)];

        if (balance >= 0) {
            if (lvaPt == PieceType::KING && (attackers & board.occupied[int(stm)]).any())
                stm = ~stm;
            break;
        }
    }

    return board.sideToMove() != stm;
}

// ─── Continuation History ──────────────────────────────────────────────
// Histórico indexado por (peça+casa do lance anterior, peça+casa do lance
// atual): "depois de X, Y costuma ser bom". gContPieceAt/gContToAt guardam,
// por ply, qual foi o lance que levou a esse ply (NONE = nulo/raiz, nunca
// escrito, fica sempre a zeros — sentinela sem necessidade de guarda extra).
static thread_local int gContHist1[7][64][6][64];  // 1 ply atrás (lance do adversário)
static thread_local int gContHist2[7][64][6][64];  // 2 plies atrás (o nosso lance anterior)
static thread_local int gContPieceAt[130];
static thread_local int gContToAt[130];
// Gap vs SF/Reckless: lance jogado para chegar a cada ply, com from/to (gContPieceAt/
// gContToAt já tinham peça+to, mas não from) — necessário para o bónus de history por
// diferença de eval entre plies (precisa de atualizar gHistory[lado][from][to] do lance
// do PAI, não só identificar a peça/casa). Move() guarda from/to/flags num único u16,
// barato de copiar.
static thread_local Move gMoveAtPly[130];
static thread_local bool gMoveWasQuietAtPly[130];

static int contHistScore(int ply, PieceType curPiece, int curTo) {
    int score = 0;
    int p1 = gContPieceAt[ply], t1 = gContToAt[ply];
    score += gContHist1[p1][t1][int(curPiece)][curTo];
    if (ply >= 1) {
        int p2 = gContPieceAt[ply - 1], t2 = gContToAt[ply - 1];
        score += gContHist2[p2][t2][int(curPiece)][curTo] / 2;
    }
    return score;
}

// Non-overlapping bands: TT > capturas boas (MVV-LVA) > killers > history
// > capturas más (SEE < 0). Capturas boas em [80000, 197500], sempre acima
// de killers/history e abaixo do lance da TT.
// Bónus de xeque na ordenação de quietos: gap vs SF/Reckless — um lance quieto que dá
// xeque (e não perde material na troca) recebe um bónus na ordenação, distinto da
// extensão de xeque que já temos na busca (essa só afeta profundidade, não ordem).
// checkSquares[pt] = conjunto de casas-destino a partir das quais uma peça do tipo `pt`
// (da cor de quem vai jogar) dá xeque direto ao rei adversário — precalculado uma vez
// por nó, reaproveitado em todos os lances (mesmo padrão do check_squares() do SF).
// theirAttacks = casas atacadas pelo adversário (gap vs Reckless: bónus de escapar duma
// casa atacada, malus de entrar numa vindo duma casa segura — versão simples e aditiva,
// sem reindexar as tabelas de history como o Reckless faz).
static int moveScore(const Board& board, Move m, Move ttMove, const int killers[2], int ply,
                      const Bitboard checkSquares[6], Bitboard theirAttacks) {
    if (m.data == ttMove.data) return 1000000;
    if (m.isCapture()) {
        PieceType attacker = board.pieceOn(m.from());
        PieceType victim    = m.isEP() ? PieceType::PAWN : board.pieceOn(m.to());
        int mvvLva = kPieceValue[int(victim)] * 100 - kPieceValue[int(attacker)];
        // Capture History: ajuste fino DENTRO das boas/más capturas — não troca a
        // ordem grosseira de SEE/MVV-LVA (perturbação pequena vs. mvvLva).
        int capHist = gCaptureHistory[int(board.sideToMove())][int(attacker)][int(victim)];
        if (seeGE(board, m, 0))
            return 100000 + mvvLva + capHist / 64;
        return mvvLva / 100;  // má troca: abaixo de killers/history
    }
    if (m.data == killers[0]) return 18000;
    if (m.data == killers[1]) return 17000;
    int score = gHistory[int(board.sideToMove())][m.from()][m.to()]
              + contHistScore(ply, board.pieceOn(m.from()), m.to());
    if (ply < LOW_PLY_MAX)
        score += gLowPlyHistory[ply][m.from()][m.to()] / (1 + ply);
    score = std::min(score, 16500);  // mantém-se sempre abaixo dos killers
    PieceType movedPt = board.pieceOn(m.from());
    if (movedPt != PieceType::KING && checkSquares[int(movedPt)].test(m.to()) && seeGE(board, m, 0))
        score += 8000;
    // Assimetria confirmada no código real do Reckless (escape[pt]/+offense bem menores
    // que o malus de entrar numa ameaça, -8875 vs +3446 nos valores deles) — punir mais
    // do que premiar. Mantém a nossa escala própria (proporcional ao history, não cópia
    // direta dos números deles), só ajustada a razão malus:bónus para ~2:1 em vez de 1:1.
    bool fromThreatened = theirAttacks.test(m.from());
    bool toThreatened    = theirAttacks.test(m.to());
    if (fromThreatened && !toThreatened)
        score += 1500;   // escapa duma casa atacada para uma segura
    else if (!fromThreatened && toThreatened)
        score -= 3000;   // sai duma casa segura para uma atacada, sem necessidade
    return score;
}

struct SortedMoves {
    Move  moves[256];
    int   scores[256];
    int   count;

    void init(const Board& board, Move ttMove, const int killers[2], int ply) {
        MoveList list;
        generateMoves(const_cast<Board&>(board), list);
        count = list.count;
        Bitboard checkSquares[6];
        {
            Color us = board.sideToMove();
            Square eks = board.kingSq(~us);
            Bitboard occ = board.allOcc;
            checkSquares[int(PieceType::PAWN)]   = attacks::pawnAttackSq(~us, eks);
            checkSquares[int(PieceType::KNIGHT)] = attacks::knightAttacks(eks);
            checkSquares[int(PieceType::BISHOP)] = attacks::bishopAttacks(eks, occ);
            checkSquares[int(PieceType::ROOK)]   = attacks::rookAttacks(eks, occ);
            checkSquares[int(PieceType::QUEEN)]  = checkSquares[int(PieceType::BISHOP)] | checkSquares[int(PieceType::ROOK)];
            checkSquares[int(PieceType::KING)]   = Bitboard();
        }
        Bitboard theirAttacks = computeSideAttacks(board, ~board.sideToMove());
        for (int i = 0; i < count; ++i) {
            moves[i]  = list.moves[i];
            scores[i] = moveScore(board, list.moves[i], ttMove, killers, ply, checkSquares, theirAttacks);
        }
    }

    Move next(int& idx) {
        int best = idx;
        for (int i = idx+1; i < count; ++i)
            if (scores[i] > scores[best]) best = i;
        std::swap(moves[idx], moves[best]);
        std::swap(scores[idx], scores[best]);
        return moves[idx++];
    }
};

// ─── Node-fraction time management ─────────────────────────────────────
// Quantos nós cada lance de raiz consumiu na última iteração completa —
// se o melhor lance comeu quase todos os nós, a busca já está bem decidida
// (estilo Stockfish nodesEffort); usado no soft time check da busca
// iterativa, junto com a best-move stability.
static thread_local Move     gRootMoves[256];
static thread_local uint64_t gRootMoveNodes[256];
static thread_local int      gRootMoveCount = 0;

// MultiPV: lances de raiz já reportados nesta profundidade (excluídos da
// próxima passada). gRootExcludedCount fica a 0 quando MultiPV=1 (default)
// — comportamento idêntico ao de antes desta técnica existir.
static thread_local Move gRootExcluded[8];
static thread_local int  gRootExcludedCount = 0;
static int  gMultiPV = 1;
void setMultiPV(int n) { gMultiPV = std::max(1, std::min(8, n)); }

// ─── Acumulador incremental (EvalState) ────────────────────────────────────
// Liga o NapkAccSlot (nnue_net.h/cpp, já implementado mas nunca chamado) ao
// makeMove/unmakeMove daqui — sem isto g_napkCurrentSlot fica sempre nullptr
// e o evaluate cai sempre no caminho antigo (plyResolve/finny). O refresh
// completo só acontece quando o rei muda de king-bucket (napkLazyPush trata
// disso); de resto é só copiar o pai + aplicar os deltas do lance.
// ⚠️ ~22KB por NapkAccSlot × 260 plies × 2 redes ≈ 11.4MB — DEMASIADO para
//   viver direto na TLS (estourava a reserva estática de TLS do processo em
//   threads novas: SIGSEGV logo na 1ª chamada a evaluate() na thread helper,
//   apanhado ao ligar Lazy SMP). Guarda-se só um ponteiro pequeno na TLS;
//   os 11.4MB ficam no heap, alocados uma vez por thread ao 1º uso.
struct EvalSlotsArray { napoleon::nnue::NapkAccSlot s[260][2]; };
static thread_local std::unique_ptr<EvalSlotsArray> gEvalSlotsHolder;
static inline napoleon::nnue::NapkAccSlot (&evalSlots())[260][2] {
    if (!gEvalSlotsHolder) gEvalSlotsHolder = std::make_unique<EvalSlotsArray>();
    return gEvalSlotsHolder->s;
}
#define gEvalSlots evalSlots()

struct MoveDelta {
    napoleon::nnue::NapkDelta adds[2]; int nAdds = 0;
    napoleon::nnue::NapkDelta rems[2]; int nRems = 0;
};

// Espelho exato dos ramos do Board::makeMove (board.cpp) — só extrai O QUE
// muda, em vez de mudar. Chamar ANTES de board.makeMove(m): precisa do
// estado pré-lance (pieceOn(from)/pieceOn(to) antes de mexer nas peças).
static MoveDelta computeMoveDelta(const Board& board, Move m) {
    MoveDelta d;
    const int from = m.from(), to = m.to(), flags = m.flags();
    const Color us = board.sideToMove(), them = ~us;
    auto add = [&](Color c, PieceType pt, int sq) { d.adds[d.nAdds++] = {int(c), int(pt), sq}; };
    auto rem = [&](Color c, PieceType pt, int sq) { d.rems[d.nRems++] = {int(c), int(pt), sq}; };
    PieceType movingPt = board.pieceOn(from);

    if (flags == Move::FLAG_CASTLE_K || flags == Move::FLAG_CASTLE_Q) {
        int rookFrom, rookTo;
        if (flags == Move::FLAG_CASTLE_K) { rookFrom = (us == Color::WHITE) ? 7  : 63; rookTo = (us == Color::WHITE) ? 5  : 61; }
        else                              { rookFrom = (us == Color::WHITE) ? 0  : 56; rookTo = (us == Color::WHITE) ? 3  : 59; }
        rem(us, PieceType::KING, from); add(us, PieceType::KING, to);
        rem(us, PieceType::ROOK, rookFrom); add(us, PieceType::ROOK, rookTo);
    } else if ((flags & Move::FLAG_PROMO_N) && !(flags & Move::FLAG_CAPTURE)) {
        rem(us, PieceType::PAWN, from);
        add(us, m.promoType(), to);
    } else if (flags >= Move::FLAG_PROMO_N_CAP) {
        rem(us, PieceType::PAWN, from);
        rem(them, board.pieceOn(to), to);
        add(us, m.promoType(), to);
    } else if (flags == Move::FLAG_EP) {
        int capSq = (us == Color::WHITE) ? to - 8 : to + 8;
        rem(us, PieceType::PAWN, from);
        rem(them, PieceType::PAWN, capSq);
        add(us, PieceType::PAWN, to);
    } else if (flags & Move::FLAG_CAPTURE) {
        rem(us, movingPt, from);
        rem(them, board.pieceOn(to), to);
        add(us, movingPt, to);
    } else {
        rem(us, movingPt, from);
        add(us, movingPt, to);
    }
    return d;
}

// Chamar DEPOIS de board.makeMove(m) (precisa do board no estado novo p/
// calcular o king-bucket). d = computeMoveDelta(board, m) calculado ANTES.
static inline void evalPush(const Board& board, const MoveDelta& d, int ply) {
    if (!napoleon::nnue::napkIncrementalEnabled()) return;
    if (ply < 0 || ply + 1 >= 260) { napoleon::nnue::napkSetCurrentSlot(nullptr); return; }
    for (int net = 0; net < 2; ++net)
        napoleon::nnue::napkLazyPush(board, gEvalSlots[ply][net], gEvalSlots[ply + 1][net],
                                     d.adds, d.nAdds, d.rems, d.nRems);
    napoleon::nnue::napkSetCurrentSlot(gEvalSlots[ply + 1]);
}
// Chamar ANTES de board.unmakeMove(m) — restaura o slot do pai como atual.
static inline void evalPop(int ply) {
    if (!napoleon::nnue::napkIncrementalEnabled()) return;
    napoleon::nnue::napkSetCurrentSlot((ply >= 0 && ply < 260) ? gEvalSlots[ply] : nullptr);
}

// ─── Quiescence search ─────────────────────────────────────────────────────
static int qsearch(Board& board, int alpha, int beta, int ply, SearchInfo& info) {
    if (info.stopped || checkTime(info)) return 0;
    ++info.nodes;
    if (ply > info.selDepth) info.selDepth = ply;

    int standPat = staticEval(board);
    if (standPat >= beta) return standPat;
    if (standPat > alpha) alpha = standPat;

    MoveList list;
    generateCaptures(board, list);

    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];

        // Delta Pruning: mesmo ganhando a peça capturada, não chega perto
        // de alfa — não vale a pena gerar/testar esta captura.
        if (!m.isPromo()) {
            PieceType victim = m.isEP() ? PieceType::PAWN : board.pieceOn(m.to());
            if (standPat + kPieceValue[int(victim)] + DELTA_MARGIN <= alpha)
                continue;
        }

        if (!board.isLegal(m)) continue;
        MoveDelta delta = computeMoveDelta(board, m);
        board.makeMove(m);
        evalPush(board, delta, ply);
        int score = -qsearch(board, -beta, -alpha, ply + 1, info);
        evalPop(ply);
        board.unmakeMove(m);

        if (info.stopped) return 0;
        if (score >= beta) return score;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ─── Mate distance pruning ─────────────────────────────────────────────────
static inline int mateAlpha(int alpha, int ply) {
    return std::max(alpha, -(MATE_SCORE - ply));
}
static inline int mateBeta(int beta, int ply) {
    return std::min(beta, MATE_SCORE - ply - 1);
}

// ─── killers per ply ───────────────────────────────────────────────────────
static thread_local int gKillers[128][2];

// ─── Hindsight depth adjustment ────────────────────────────────────────────
// Ideia independente convergida pelo SF e Reckless: ao ENTRAR num nó, olha
// para a redução LMR que o PAI aplicou a descer até aqui e para a variação
// da eval entre o pai e agora. Se o pai reduziu MUITO e a posição não piorou,
// a redução foi provavelmente excessiva → devolve 1 ply. Se o pai reduziu
// pouco/nada e a eval MELHOROU muito, a redução (se houve) foi justificada e
// ainda há margem → reduz mais 1 ply. Ajuste pequeno e local, sem tabelas
// novas — só 2 arrays indexados por ply (mesmo padrão dos killers).
static thread_local int gReductionAtPly[128];  // redução LMR aplicada pelo PAI a descer para esta ply
static thread_local int gEvalAtPly[128];       // eval (corrigida) do nó nesta ply, POV de quem joga aí

static void formatMoveUci(Move m, char* out) {
    out[0] = 'a' + (m.from() & 7);
    out[1] = '1' + (m.from() >> 3);
    out[2] = 'a' + (m.to() & 7);
    out[3] = '1' + (m.to() >> 3);
    out[4] = '\0';
    if (m.isPromo()) {
        const char pc[] = "nbrq";
        out[4] = pc[m.flags() & 3];
        out[5] = '\0';
    }
}

// Tabela PV triangular: gPvTable[ply] guarda a continuação completa a partir desse ply
// (até gPvLength[ply] lances), atualizada DURANTE a busca sempre que um lance melhora
// alpha. Gap real corrigido: a linha "pv" só mostrava o primeiro lance ("just best move
// for now", segundo o próprio comentário antigo — nunca tinha sido terminado). Uma
// primeira tentativa reconstruía a PV percorrendo a TT no FIM da busca — falhava na
// prática: confirmado por instrumentação que a entrada da TT da posição-filha era quase
// sempre um MISS nesse momento (substituída por outras posições exploradas depois na
// mesma busca, índice partilhado sem buckets) — a PV parava sempre no 1º lance. A
// tabela triangular não tem este problema: o valor é capturado no INSTANTE em que a
// linha é a melhor conhecida, não reconstruído depois a partir duma tabela que já mudou.
static thread_local Move gPvTable[130][130];
static thread_local int  gPvLength[130];

static void updatePv(int ply, Move m) {
    int p = std::min(ply, 129);
    gPvTable[p][0] = m;
    int childLen = std::min(gPvLength[std::min(p + 1, 129)], 128);
    for (int i = 0; i < childLen; ++i)
        gPvTable[p][i + 1] = gPvTable[std::min(p + 1, 129)][i];
    gPvLength[p] = childLen + 1;
}

static int formatPv(const Move* moves, int len, char* out, size_t outSize) {
    int written = 0;
    for (int i = 0; i < len; ++i) {
        char mstr[8];
        formatMoveUci(moves[i], mstr);
        int n = snprintf(out + written, outSize - written, "%s%s", written ? " " : "", mstr);
        if (n < 0 || (size_t)(written + n) >= outSize) break;
        written += n;
    }
    return written;
}

// ─── Aspiration Windows ──────────────────────────────────────────────────
static int ASPIRATION_MIN_DEPTH = 4;
static int ASPIRATION_DELTA     = 16;

// ─── Razoring ────────────────────────────────────────────────────────────
static int RAZOR_BASE = 300;
static int RAZOR_MULT = 300;

// ─── Reverse Futility Pruning ──────────────────────────────────────────────
// Magnitude informada pelo Coda (RFP_MARGIN_NOIMP=43; sem rescale agora que
// OUTPUT_SCALE_CP=400; sem flag "improving" ainda, por isso só um valor).
// Por afinar com SPSA depois de validado.
static int RFP_MAX_DEPTH = 7;
static int RFP_MARGIN    = 43;  // OUTPUT_SCALE_CP=400 now, no 408/400 rescale needed

// ─── Null Move Pruning ──────────────────────────────────────────────────────
// NMP_BASE_R/NMP_DIV informados pelo Coda (7.8 / 7.5 → arredondado);
// não são valores em cp, não se escalam por OUTPUT_SCALE_CP.
static int NMP_MIN_DEPTH = 3;
static int NMP_BASE_R    = 8;
static int NMP_DIV       = 7;

// ─── Late Move Reductions ───────────────────────────────────────────────
// Tabela log(depth)*log(moveCount)/C, C=1.3 neutro (ponto de partida comum
// a Stockfish/Reckless/Coda antes de SPSA). Limitada a depth-2 para nunca
// reduzir abaixo de profundidade 1.
static int LMR_MIN_DEPTH = 3;
static int LMR_MIN_MOVES = 3;
// LMR_C como int×100 (SPSA/UCI só faz spin de inteiros) — afinável via
// rebuildLmrTable() sempre que mudar (a tabela é precomputada).
static int LMR_C_X100 = 130;
static int gLmrTable[64][64];

static bool rebuildLmrTable() {
    double c = LMR_C_X100 / 100.0;
    for (int d = 1; d < 64; ++d)
        for (int n = 1; n < 64; ++n) {
            if (d >= LMR_MIN_DEPTH && n >= LMR_MIN_MOVES) {
                int r = int(std::log(d) * std::log(n) / c);
                gLmrTable[d][n] = std::min(r, d - 2);
            } else {
                gLmrTable[d][n] = 0;
            }
        }
    return true;
}
static const bool gLmrInit = rebuildLmrTable();

// ─── Internal Iterative Reduction ───────────────────────────────────────
static int IIR_MIN_DEPTH = 4;

// ─── ProbCut ─────────────────────────────────────────────────────────────
static int PROBCUT_MIN_DEPTH = 5;
static int PROBCUT_MARGIN    = 220;  // ~ordem de SEE_PRUNE_MARGIN

// ─── Singular Extensions ────────────────────────────────────────────────
static int SE_MIN_DEPTH = 6;
static int SE_MARGIN    = 64;
// Gap vs SF/Reckless: o nosso singularExt era binário (0 ou 1). Ambos os motores de
// referência graduam por quanto a busca de verificação ficou abaixo de singularBeta —
// mais confiança no lance único, mais se aprofunda (até 3). Constantes próprias, nascem
// neutras. Extensão negativa (-3) confirmada e implementada também, lendo o código real
// do Reckless (ver bloco da busca de verificação) em vez de adivinhar a condição.
static int SE_DOUBLE_MARGIN = 16;
static int SE_TRIPLE_MARGIN = 80;

// ─── Late Move Pruning ───────────────────────────────────────────────────
// Magnitudes informadas pelo Coda (engine de referência mais próximo,
// também 100% vibe-coded), constantes de profundidade/contagem sem escala
// (não são cp); margens em cp sem rescale agora que OUTPUT_SCALE_CP=400.
static int LMP_MAX_DEPTH = 8;
static int LMP_BASE      = 6;
static int LMP_MULT      = 1;

// ─── Futility Pruning ───────────────────────────────────────────────────
static int FUTILITY_MAX_DEPTH = 8;
static int FUTILITY_BASE      = 80;   // OUTPUT_SCALE_CP=400 now, no 408/400 rescale needed
static int FUTILITY_MARGIN    = 110;  // OUTPUT_SCALE_CP=400 now, no 408/400 rescale needed

// ─── SEE Pruning ────────────────────────────────────────────────────────
static int SEE_PRUNE_MAX_DEPTH = 7;
static int SEE_PRUNE_MARGIN    = 215; // OUTPUT_SCALE_CP=400 now, no 408/400 rescale needed

// 🦅 Gap vs SF/Reckless: ambos também SEE-prunam QUIETOS (não só capturas), com limiar
// QUADRÁTICO em depth (mais permissivo a depth baixa, mais estrito conforme depth sobe —
// o inverso do linear que já usamos para capturas). Constante própria, nasce neutra
// (não copiada de SF/Reckless, só a FORMA -depth² é a ideia importada).
static int QUIET_SEE_PRUNE_MAX_DEPTH = 7;
static int QUIET_SEE_PRUNE_MARGIN    = 20; // limiar = -MARGIN * depth * depth

// ─── History Pruning ────────────────────────────────────────────────────
static int HIST_PRUNE_MAX_DEPTH = 8;
static int HIST_PRUNE_MARGIN    = 1500;

static bool hasNonPawnMaterial(const Board& board, Color c) {
    return board.pieces(c, PieceType::KNIGHT).any() ||
           board.pieces(c, PieceType::BISHOP).any() ||
           board.pieces(c, PieceType::ROOK).any()   ||
           board.pieces(c, PieceType::QUEEN).any();
}

// Bloqueia novo NMP recursivo até este ply — busca de verificação contra
// zugzwang (mesmo mecanismo do Stockfish/Reckless: nmpMinPly).
static thread_local int gNmpMinPly = 0;

// ─── PVS / Alpha-Beta ─────────────────────────────────────────────────────
static int search(Board& board, int depth, int alpha, int beta,
                  int ply, bool pvNode, SearchInfo& info, bool prevNull = false,
                  Move excludedMove = NULL_MOVE) {
    if (info.stopped || checkTime(info)) return 0;

    const bool root = (ply == 0);
    gPvLength[std::min(ply, 129)] = 0;  // limpa lixo de uma chamada anterior a este mesmo ply

    // Empate por repetição ou regra dos 50 lances — antes de tudo, inclusive
    // da TT (uma posição repetida não deve confiar num score de outro caminho).
    // Pequeno jitter (±1cp pela paridade da contagem de nós) em vez de 0 fixo: gap vs
    // SF (value_draw) — evita que a busca fique "cega" à ordem de evitar/buscar repetição
    // quando todas as alternativas dão exatamente 0 (essencialmente gratuito, sem custo
    // de Elo mensurável, só desempata).
    if (!root && (board.halfmoveClock() >= 100 || board.isRepetition()))
        return 1 - (int)(info.nodes & 2);

    // Mate distance pruning
    alpha = mateAlpha(alpha, ply);
    beta  = mateBeta(beta, ply);
    if (alpha >= beta) return alpha;

    // Quiescence at leaf
    if (depth <= 0) return qsearch(board, alpha, beta, ply, info);

    ++info.nodes;
    if (ply > info.selDepth) info.selDepth = ply;

    // TT probe
    bool ttHit = false;
    TTEntry* tte = gTT.probe(board.hash, ttHit);
    Move ttMove = ttHit ? Move(tte->move) : NULL_MOVE;
    int  ttScore = ttHit ? tte->score : 0;

    // Gap vs SF/Reckless: perto da regra dos 50 lances, um corte de TT pode vir duma
    // posição idêntica em bitboard mas alcançada por um caminho com OUTRA distância ao
    // empate (graph-history interaction) — o score guardado já não é fiável aqui. Os
    // dois motores de referência desconfiam da TT a partir de halfmoveClock alto;
    // mitigação de correção, não só de Elo.
    if (!root && ttHit && tte->depth >= depth && board.halfmoveClock() < 90) {
        Bound b = tte->bound();
        if (b == Bound::EXACT) return ttScore;
        if (b == Bound::LOWER && ttScore >= beta) return ttScore;
        if (b == Bound::UPPER && ttScore <= alpha) return ttScore;
        // Gap vs SF: o bound guardado não serviu para cortar aqui (era quase certo, mas
        // não para esta janela) — penaliza a profundidade guardada para a entrada ser
        // substituída mais cedo, já que não está a ser útil neste contexto.
        if (tte->depth > 0) --tte->depth;
    }

    const bool inCheck = board.isInCheck();

    // TT eval adjustment: reaproveita a eval guardada na TT (de uma visita
    // anterior) em vez de recalcular o forward pass da NNUE — mesmo valor,
    // mais barato. Um só cálculo por nó, partilhado por todas as podas.
    int rawEval = (ttHit && tte->eval != 0) ? tte->eval
                : staticEval(board) + gOptimism[int(board.sideToMove())];
    int corrTerm = pawnCorrTerm(board) + nonPawnCorrTerm(board);
    int eval    = rawEval + corrTerm;

    // Hindsight depth adjustment, AMBAS as direções — confirmado lendo o código real do
    // Reckless (src/search.rs, bloco "Hindsight reductions") em vez de adivinhar o sinal:
    //   eval_delta = eval (este nó) + stack[ply-1].eval (pai) -- soma, não subtração,
    //   porque negamax já inverte o sinal entre plies consecutivos; a soma dá a variação
    //   "traduzida" para a perspetiva de quem jogou o lance do pai.
    //   - reduction>=2249 (~2.2 plies nas unidades internas deles) && eval_delta<0 → +1 ply
    //     (a redução foi grande E a posição piorou mais do que o pai esperava — devolve).
    //   - reduction>0 (qualquer redução) && eval_delta>57 && !tt_pv && depth>=2 → -1 ply
    //     (mesmo com pouca/nenhuma redução, a posição melhorou bastante — corta mais).
    // Constantes próprias (3 plies, 50cp), não copiadas das deles (escala diferente). Sem
    // "tt_pv" próprio na nossa TT -- uso !pvNode como aproximação razoável.
    // 🦅 FIX (bug real, encontrado por revisão depois do SPRT negativo do bloco7): faltava
    // o guard excludedMove.isNull(). A busca de verificação do singular extension chama
    // search() na MESMA ply com excludedMove=m -- sem este guard, este bloco corria
    // OUTRA VEZ nessa chamada extra, ajustando depth com base num sinal que não devia
    // disparar ali (confirmado: Reckless tem !excluded em todos os guards equivalentes).
    if (!root && !inCheck && excludedMove.isNull() && ply >= 1) {
        int pPly = std::min(ply, 127);
        int priorReduction = gReductionAtPly[pPly];
        int parentEval = gEvalAtPly[std::min(ply - 1, 127)];
        int evalDelta = eval + parentEval;
        if (priorReduction >= 3 && evalDelta < 0)
            ++depth;
        else if (!pvNode && depth >= 2 && priorReduction > 0 && evalDelta > 57)
            --depth;
    }
    gEvalAtPly[std::min(ply, 127)] = eval;

    // Bónus/malus de history por diferença de eval entre plies — confirmado lendo o
    // código real do Reckless (src/search.rs): atualiza o history do lance do PAI (não
    // deste nó) consoante a eval melhorou ou piorou entre o pai e agora. value = K *
    // -(eval + evalDoPai) -- soma (não subtração) porque negamax já inverte o sinal entre
    // plies; o resultado fica na perspetiva de quem jogou o lance do pai. Só para lances
    // quietos do pai, só a profundidades baixas OU sem TT hit (evita reforçar repetido em
    // posições já bem exploradas — mesmo guard do Reckless). Constantes próprias (K e
    // clamp), não copiadas das deles (escala diferente).
    // 🦅 FIX (mesmo motivo do bloco hindsight acima): falta excludedMove.isNull() —
    // sem isto, a busca de verificação do singular extension ATUALIZAVA O HISTORY DO
    // LANCE DO PAI EM DUPLICADO (uma vez na entrada normal do nó, outra na chamada de
    // verificação excluída) -- corrompe ordenação em qualquer nó onde o singular
    // extension dispare (depth>=6, comum). Confirmado: Reckless tem !excluded aqui.
    if (!root && !inCheck && excludedMove.isNull() && ply >= 1 && gMoveWasQuietAtPly[std::min(ply, 127)] && (depth < 6 || !ttHit)) {
        int pPly = std::min(ply, 127);
        Move parentMove = gMoveAtPly[pPly];
        int parentEval = gEvalAtPly[std::min(ply - 1, 127)];
        int value = -(eval + parentEval);
        // Multiplicador e clamp aproximados ao valor real do Reckless (812/128≈6.34,
        // clamp [-144,324]) em vez do "/2, [-150,300]" anterior — escalas de eval
        // plausivelmente comparáveis entre motores modernos (ambos cp-like), e o clamp
        // já estava muito próximo por coincidência.
        int bonus = std::clamp((value * 812) / 128, -144, 324);
        int moverSide = int(~board.sideToMove());
        int& h = gHistory[moverSide][parentMove.from()][parentMove.to()];
        h += bonus;
        if (h > 16000) h = 16000;
        if (h < -16000) h = -16000;
    }

    // |corrTerm| como sinal de confiança: quando a correção teve de ajustar muito a eval,
    // a eval estática é menos fiável aqui — alarga as margens de poda (RFP/futility/SEE)
    // proporcionalmente. Mesma ideia usada pelo SF/Reckless (correction_value.abs()/N em
    // várias fórmulas); o valor já estava calculado, só faltava ser lido por outros sítios.
    int corrConfDiv = std::clamp(std::abs(corrTerm), 0, CORR_HIST_MAX);

    // Razoring: eval estática muito abaixo de alfa, mesmo com várias
    // jogadas de margem — cai direto em qsearch (ordem de consenso:
    // razor -> RFP -> NMP).
    if (!pvNode && !inCheck && std::abs(alpha) < MATE_SCORE - 512
        && eval < alpha - RAZOR_BASE - RAZOR_MULT * depth * depth) {
        return qsearch(board, alpha, beta, ply, info);
    }

    // Reverse Futility Pruning: eval estática já muito acima de beta a
    // poucos plies da folha — o adversário não vai deixar a posição chegar
    // aqui, corta sem gerar lances.
    if (!root && !pvNode && !inCheck && depth <= RFP_MAX_DEPTH) {
        int margin = RFP_MARGIN * depth + corrConfDiv / 4;
        if (eval - margin >= beta)
            return eval - margin;
    }

    // Null Move Pruning: passa a vez; se a posição continua a ganhar mesmo
    // assim, o adversário não tem ameaça real aqui — corta. Evita-se perto
    // de mate, em finais só de peões (risco de zugzwang), e não se confia
    // num score de mate não comprovado vindo do ramo nulo.
    if (!root && !pvNode && !inCheck && !prevNull && depth >= NMP_MIN_DEPTH
        && ply >= gNmpMinPly
        && beta < MATE_SCORE - 512
        && hasNonPawnMaterial(board, board.sideToMove())
        && eval >= beta) {
        int R = NMP_BASE_R + depth / NMP_DIV;
        if (ply + 1 < 130) {
            gContPieceAt[ply + 1] = int(PieceType::NONE);
            gContToAt[ply + 1]    = 0;
        }
        board.makeNullMove();
        evalPush(board, MoveDelta{}, ply);   // passa a vez: 0 peças mudam, bucket não muda
        gReductionAtPly[std::min(ply + 1, 127)] = 0;  // NMP não é LMR — não confundir no hindsight depth adjustment
        // 🦅 FIX: gMoveWasQuietAtPly[ply+1] também tem de ser limpo — sem isto, o bloco
        // de history por diferença de eval no nó filho lia o LANCE/flag-quieto de outro
        // lance qualquer (de um irmão anterior no MESMO ply, lixo de outra parte da
        // árvore), bombardeando o history desse lance ERRADO com um bónus/malus que não
        // lhe pertence. NMP não tem lance real (passa a vez) — false desliga o bloco.
        gMoveWasQuietAtPly[std::min(ply + 1, 127)] = false;
        int score = -search(board, depth - 1 - R, -beta, -beta + 1, ply + 1, false, info, true);
        evalPop(ply);
        board.unmakeNullMove();
        if (info.stopped) return 0;

        if (score >= beta) {
            // Não propagar distância de mate não comprovada vinda do ramo
            // nulo — mas mantém-se o corte com um score seguro (beta).
            int cutScore = isMate(score) ? beta : score;

            // A profundidades altas, confirma com uma busca de verificação
            // (sem NMP até este ply) antes de confiar no corte — protege
            // contra falsos cortes por zugzwang.
            if (gNmpMinPly > 0 || depth < 16)
                return cutScore;

            gNmpMinPly = ply + (3 * (depth - R)) / 4;
            int verify = search(board, depth - R, beta - 1, beta, ply, false, info, prevNull);
            gNmpMinPly = 0;

            if (info.stopped) return 0;
            if (verify >= beta)
                return cutScore;
        }
    }

    // Internal Iterative Reduction: sem hash move e profundidade
    // suficiente, a TT não tem nada para guiar a ordenação aqui — reduz-se
    // um ply (a própria busca preenche a TT antes de voltar a este nó).
    if (!root && !ttHit && depth >= IIR_MIN_DEPTH)
        --depth;

    // ProbCut: uma captura suficientemente boa (SEE>=0) cuja procura
    // reduzida confirma ficar muito acima de beta — corta quase de
    // borla, sem percorrer o resto dos lances.
    if (!root && !pvNode && !inCheck && depth >= PROBCUT_MIN_DEPTH
        && beta < MATE_SCORE - 512) {
        int probCutBeta = beta + PROBCUT_MARGIN;
        // ProbCut "barato": se a TT já tem um bound inferior >= probCutBeta, corta sem
        // sequer gerar capturas — gap vs SF (mesma ideia, um corte extra antes do ciclo
        // de lances, de borla porque a TT já estava carregada). Reaproveita ttHit/tte já
        // lidos no topo da função.
        if (ttHit && tte->depth >= depth && tte->bound() == Bound::LOWER && ttScore >= probCutBeta)
            return ttScore;
        MoveList caps;
        generateCaptures(board, caps);
        for (int i = 0; i < caps.count; ++i) {
            Move m = caps.moves[i];
            if (!seeGE(board, m, 0)) continue;

            MoveDelta delta = computeMoveDelta(board, m);
            board.makeMove(m);
            if (board.isSquareAttacked(board.kingSq(~board.stm).value(), board.stm)) {
                board.unmakeMove(m);
                continue;
            }
            if (ply + 1 < 130) {
                gContPieceAt[ply + 1] = int(board.pieceOn(m.to()));
                gContToAt[ply + 1]    = m.to();
            }
            evalPush(board, delta, ply);

            int score = -qsearch(board, -probCutBeta, -probCutBeta + 1, ply + 1, info);
            int probCutDepth = depth - 4;
            if (!info.stopped && score >= probCutBeta && probCutDepth > 0) {
                gReductionAtPly[std::min(ply + 1, 127)] = 0;  // ProbCut não é LMR
                // 🦅 FIX: limpa explicitamente (mesmo motivo do NMP acima) -- o lance do
                // ProbCut É uma captura (logo já seria "false" se os dados fossem frescos),
                // mas este array a esta altura ainda tem o que sobrou doutro nó qualquer;
                // não depender disso, definir explicitamente.
                gMoveWasQuietAtPly[std::min(ply + 1, 127)] = false;
                score = -search(board, probCutDepth, -probCutBeta, -probCutBeta + 1, ply + 1, false, info);
            }
            evalPop(ply);
            board.unmakeMove(m);

            if (info.stopped) return 0;
            if (score >= probCutBeta) return score;
        }
    }

    // Generate and sort moves
    const int* killers = gKillers[std::min(ply, 127)];
    SortedMoves sm;
    sm.init(board, ttMove, killers, ply);

    int bestScore  = -INF_SCORE;
    Move bestMove  = NULL_MOVE;
    Bound bound    = Bound::UPPER;
    int  idx       = 0;
    int  legalCnt  = 0;
    if (root) gRootMoveCount = 0;  // recomeça a contagem por lance desta iteração
    int  quietTried = 0;
    Move triedQuiets[64];
    int  triedQuietCount = 0;
    Move triedCaptures[64];
    PieceType triedCaptureVictims[64];
    int  triedCaptureCount = 0;

    for (int i = 0; i < sm.count; ++i) {
        Move m = sm.next(idx);
        if (!excludedMove.isNull() && m.data == excludedMove.data) continue;
        // MultiPV (análise; off em jogo): exclui na RAIZ os lances já
        // reportados nas linhas anteriores desta mesma profundidade — só
        // gRootExcludedCount>0 quando MultiPV>1 está ativo (ver driver).
        if (root && gRootExcludedCount > 0) {
            bool excl = false;
            for (int ei = 0; ei < gRootExcludedCount; ++ei)
                if (m.data == gRootExcluded[ei].data) { excl = true; break; }
            if (excl) continue;
        }
        bool isQuiet = !m.isCapture() && !m.isPromo();

        // Late Move Pruning / Futility Pruning / History Pruning: só
        // depois de já termos pelo menos um lance legal (nunca arriscar
        // concluir mate/afogamento por ter prunado tudo). Lances tranquilos
        // tardios a pouca profundidade raramente mudam o resultado.
        if (isQuiet && !root && !pvNode && !inCheck && legalCnt >= 1) {
            if (depth <= LMP_MAX_DEPTH && quietTried > LMP_BASE + LMP_MULT * depth * depth) {
                ++quietTried;
                continue;
            }
            if (depth <= FUTILITY_MAX_DEPTH && eval + FUTILITY_BASE + FUTILITY_MARGIN * depth + corrConfDiv / 4 <= alpha) {
                ++quietTried;
                continue;
            }
            if (depth <= HIST_PRUNE_MAX_DEPTH) {
                int combined = gHistory[int(board.sideToMove())][m.from()][m.to()]
                             + contHistScore(ply, board.pieceOn(m.from()), m.to());
                if (combined < -HIST_PRUNE_MARGIN * depth) {
                    ++quietTried;
                    continue;
                }
            }
            // SEE Pruning para QUIETOS: gap vs SF/Reckless — só prunávamos capturas por SEE.
            // Um lance tranquilo que perde material na troca (ex.: mover p/ uma casa atacada
            // sem compensação) raramente vale a pena testar a pouca profundidade.
            // 🦅 FIX: usava `depth` (bruto) no limiar -- confirmado lendo o código real do
            // SF (src/search.cpp) que usa especificamente `lmrDepth` (a profundidade JÁ
            // reduzida pelo LMR que este lance levaria) para a versão quadrática em
            // quietos, não a profundidade bruta. lmrDepth <= depth sempre, por isso
            // depth² dava um limiar mais permissivo (mais negativo) do que devia —
            // prunava sistematicamente MENOS do que o SF pretende nesta técnica
            // específica. Reckless usa depth bruto nas duas (captura e quieto), por isso
            // esta escolha segue especificamente o SF, não é consenso entre os dois.
            int seeLmrDepth = depth;
            if (depth >= LMR_MIN_DEPTH && legalCnt >= LMR_MIN_MOVES)
                seeLmrDepth = std::max(depth - gLmrTable[std::min(depth, 63)][std::min(legalCnt, 63)], 0);
            if (depth <= QUIET_SEE_PRUNE_MAX_DEPTH
                && !seeGE(board, m, -QUIET_SEE_PRUNE_MARGIN * seeLmrDepth * seeLmrDepth)) {
                ++quietTried;
                continue;
            }
        }
        if (isQuiet) ++quietTried;

        // SEE Pruning: captura com troca claramente perdedora a pouca
        // profundidade — não vale a pena testar.
        if (m.isCapture() && !root && !pvNode && !inCheck && legalCnt >= 1
            && depth <= SEE_PRUNE_MAX_DEPTH
            && !seeGE(board, m, -SEE_PRUNE_MARGIN * depth - corrConfDiv / 8)) {
            continue;
        }

        // Futility Pruning para CAPTURAS: até agora só prunávamos quietos por
        // futility — capturas também podem ser claramente inúteis a pouca
        // profundidade (eval + valor da peça capturada + margem ainda fica
        // abaixo de alfa). Soma-se o valor da vítima (a captura "ganha" isso)
        // antes de comparar — gap encontrado a comparar com SF/Reckless
        // (ambos distinguem futility de quietos vs. capturas, nós só tínhamos
        // a versão de quietos).
        if (m.isCapture() && !root && !pvNode && !inCheck && legalCnt >= 1
            && depth <= FUTILITY_MAX_DEPTH) {
            PieceType victimPt = m.isEP() ? PieceType::PAWN : board.pieceOn(m.to());
            if (eval + FUTILITY_BASE + FUTILITY_MARGIN * depth + kPieceValue[int(victimPt)] + corrConfDiv / 4 <= alpha)
                continue;
        }

        // Singular Extension: o lance da TT, testado sem ele (janela
        // estreita abaixo do score da TT) — se nenhum outro lance chega lá,
        // este é claramente o único bom, vale a pena aprofundar.
        int singularExt = 0;
        if (!root && excludedMove.isNull() && m.data == ttMove.data && depth >= SE_MIN_DEPTH
            && ttHit && tte->depth >= depth - 3 && tte->bound() != Bound::UPPER
            && std::abs(ttScore) < MATE_SCORE - 512) {
            int singularBeta = ttScore - SE_MARGIN;
            int singularDepth = (depth - 1) / 2;
            int sScore = search(board, singularDepth, singularBeta - 1, singularBeta,
                                 ply, false, info, prevNull, m);
            if (info.stopped) return 0;
            if (sScore < singularBeta) {
                // Graduação: quanto mais a verificação ficou abaixo de singularBeta, mais
                // confiança de que o lance da TT é mesmo o único bom — aprofunda mais.
                singularExt = 1;
                if (sScore < singularBeta - SE_DOUBLE_MARGIN) ++singularExt;
                if (sScore < singularBeta - SE_TRIPLE_MARGIN) ++singularExt;
            } else if (sScore >= beta && std::abs(sScore) < MATE_SCORE - 512) {
                // Multi-cut: a verificação exclui o lance da TT e AINDA ASSIM bate a beta
                // exterior — outro lance qualquer já corta aqui, não vale a pena continuar
                // a testar lances neste nó. Gap vs SF/Reckless (não tínhamos nenhum uso do
                // resultado da verificação para além de decidir estender ou não).
                // 🦅 FIX: devolvia sScore em bruto. O Reckless aproxima-o de beta (~40% do
                // caminho) em vez de devolver o valor cheio — sScore vem duma busca a
                // profundidade REDUZIDA (singularDepth), por isso é menos fiável que um
                // valor normal desta profundidade; inflar o corte com o valor bruto
                // propaga um score otimista de mais para o pai (janelas de aspiração,
                // outras podas). Mistura para beta, não copia o valor exato deles.
                return beta + (sScore - beta) * 6 / 10;
            } else if (ttScore >= beta) {
                // Extensão negativa: confirmado no código real do Reckless (src/search.rs,
                // bloco singular) — gate exato é `tt_score >= beta || cut_node`. Não temos
                // o conceito de "cut_node" (flag de PVS para nós esperados a falhar alto)
                // distinto de pvNode na nossa busca, por isso só a metade ttScore>=beta —
                // o lance da TT já "parecia" bom o suficiente para cortar, mas a verificação
                // mostrou que NÃO é singular (outro lance também chega lá) — desconfia um
                // pouco deste lance, reduz em vez de assumir que é mesmo o melhor.
                singularExt = -3;
            }
        }

        PieceType capturedVictim = m.isCapture()
            ? (m.isEP() ? PieceType::PAWN : board.pieceOn(m.to())) : PieceType::NONE;
        MoveDelta delta = computeMoveDelta(board, m);
        uint64_t nodesBeforeMove = root ? info.nodes : 0;
        board.makeMove(m);
        // Verify move is legal (king of moving side not in check)
        if (board.isSquareAttacked(board.kingSq(~board.stm).value(), board.stm)) {
            board.unmakeMove(m);
            continue;
        }
        ++legalCnt;
        evalPush(board, delta, ply);

        if (ply + 1 < 130) {
            gContPieceAt[ply + 1] = int(board.pieceOn(m.to()));
            gContToAt[ply + 1]    = m.to();
            gMoveAtPly[ply + 1]      = m;
            gMoveWasQuietAtPly[ply + 1] = isQuiet;
        }

        // Check Extension: lance que dá xeque aprofunda 1 ply (sequências de
        // xeque tendem a ser forçadas/táticas). Limite de ply como rede de
        // segurança — a deteção de empate acima já trata repetição/50 lances.
        const bool givesCheck = board.isInCheck();
        const int checkExt = (givesCheck && ply < 100) ? 1 : 0;
        // 🦅 FIX (bug real encontrado depois do SPRT negativo do bloco7): a versão
        // anterior somava checkExt+singularExt sempre que singularExt era negativo — um
        // lance que dá xeque (checkExt=+1) e É TAMBÉM o lance da TT com extensão negativa
        // do singular (-3) ficava com ext=1-3=-2, REDUZINDO em vez de estender um xeque.
        // Xeque é um sinal de segurança táctica imediata — nunca deve ser anulado por um
        // sinal sobre OUTRA coisa (quão "único" é o lance da TT). Agora: havendo xeque,
        // a extensão negativa só pode reduzir a partir do check extension, nunca abaixo
        // dele NEM anulá-lo; sem xeque, comporta-se como antes.
        int ext;
        if (singularExt >= 0)
            ext = std::max(checkExt, singularExt);          // comportamento original, sem mudanças
        else if (checkExt > 0)
            ext = checkExt;                                  // xeque protegido — ignora a negativa
        else
            ext = singularExt;                                // sem xeque, a negativa aplica-se normalmente

        int score;
        if (legalCnt == 1) {
            gReductionAtPly[std::min(ply + 1, 127)] = 0;  // sem LMR neste ramo — limpa lixo de outro nó na mesma ply
            score = -search(board, depth-1+ext, -beta, -alpha, ply+1, pvNode, info);
        } else {
            int newDepth = depth - 1 + ext;
            int r = 0;
            if (depth >= LMR_MIN_DEPTH && legalCnt >= LMR_MIN_MOVES && !inCheck && !givesCheck
                && !m.isCapture() && !m.isPromo()) {
                r = gLmrTable[std::min(depth, 63)][std::min(legalCnt, 63)];
                if (pvNode && r > 0) --r;
            }

            // Busca reduzida em janela nula; se bater alpha, confirma a
            // profundidade completa antes de considerar reabrir a janela.
            gReductionAtPly[std::min(ply + 1, 127)] = r;  // p/ o filho ler (hindsight depth adjustment)
            score = -search(board, newDepth - r, -alpha-1, -alpha, ply+1, false, info);
            if (!info.stopped && score > alpha && r > 0) {
                gReductionAtPly[std::min(ply + 1, 127)] = 0;  // re-busca é a profundidade completa, sem redução
                score = -search(board, newDepth, -alpha-1, -alpha, ply+1, false, info);
            }
            if (!info.stopped && score > alpha && score < beta)
                score = -search(board, newDepth, -beta, -alpha, ply+1, true, info);
        }
        evalPop(ply);
        board.unmakeMove(m);

        if (root && gRootMoveCount < 256) {
            gRootMoves[gRootMoveCount] = m;
            gRootMoveNodes[gRootMoveCount] = info.nodes - nodesBeforeMove;
            ++gRootMoveCount;
        }

        if (info.stopped) return 0;

        if (isQuiet && triedQuietCount < 64)
            triedQuiets[triedQuietCount++] = m;
        else if (m.isCapture() && triedCaptureCount < 64) {
            triedCaptures[triedCaptureCount] = m;
            triedCaptureVictims[triedCaptureCount++] = capturedVictim;
        }

        if (score > bestScore) {
            bestScore = score;
            bestMove  = m;
            if (score > alpha) {
                alpha = score;
                bound = Bound::EXACT;
                updatePv(ply, m);
                if (score >= beta) {
                    if (m.isCapture()) {
                        // Capture History: bónus à captura que cortou, malus
                        // às tentadas antes que não cortaram — mesmo esquema
                        // do history de quiets, só que indexado por
                        // (lado, atacante, vítima) em vez de (from, to).
                        int side = int(board.sideToMove());
                        int& ch = gCaptureHistory[side][int(board.pieceOn(m.from()))][int(capturedVictim)];
                        ch += depth * depth;
                        if (ch > 16000) ch = 16000;
                        for (int ci = 0; ci < triedCaptureCount - 1; ++ci) {
                            Move cm = triedCaptures[ci];
                            int& chq = gCaptureHistory[side][int(board.pieceOn(cm.from()))][int(triedCaptureVictims[ci])];
                            chq -= depth * depth;
                            if (chq < -16000) chq = -16000;
                        }
                    }
                    if (!m.isCapture()) {
                        gKillers[std::min(ply,127)][1] = gKillers[std::min(ply,127)][0];
                        gKillers[std::min(ply,127)][0] = m.data;

                        int& h = gHistory[int(board.sideToMove())][m.from()][m.to()];
                        h += depth * depth;
                        if (h > 16000) h = 16000;

                        if (ply < LOW_PLY_MAX) {
                            int& hlp = gLowPlyHistory[ply][m.from()][m.to()];
                            hlp += depth * depth;
                            if (hlp > 16000) hlp = 16000;
                        }

                        // (m.to() já não tem a peça lá — o tabuleiro já foi
                        // desfeito acima; usa-se o valor guardado ao entrar
                        // no lance, que ainda é válido para este m.)
                        PieceType movedPt = PieceType(gContPieceAt[ply + 1]);
                        int p1 = gContPieceAt[ply], t1 = gContToAt[ply];
                        int& h1 = gContHist1[p1][t1][int(movedPt)][m.to()];
                        h1 += depth * depth;
                        if (h1 > 16000) h1 = 16000;
                        if (ply >= 1) {
                            int p2 = gContPieceAt[ply - 1], t2 = gContToAt[ply - 1];
                            int& h2 = gContHist2[p2][t2][int(movedPt)][m.to()];
                            h2 += depth * depth;
                            if (h2 > 16000) h2 = 16000;
                        }

                        // Malus: lances tranquilos tentados antes deste e
                        // que não cortaram ficam com history mais negativo
                        // (alimenta o History Pruning de outros nós).
                        for (int qi = 0; qi < triedQuietCount - 1; ++qi) {
                            Move qm = triedQuiets[qi];
                            int& hq = gHistory[int(board.sideToMove())][qm.from()][qm.to()];
                            hq -= depth * depth;
                            if (hq < -16000) hq = -16000;
                            if (ply < LOW_PLY_MAX) {
                                int& hqlp = gLowPlyHistory[ply][qm.from()][qm.to()];
                                hqlp -= depth * depth;
                                if (hqlp < -16000) hqlp = -16000;
                            }
                        }
                    }
                    bound = Bound::LOWER;
                    break;
                }
            }
        }
    }

    if (legalCnt == 0) {
        // Stalemate or checkmate
        return board.isInCheck() ? -(MATE_SCORE - ply) : 0;
    }

    // rawEval já calculado antes do ciclo de lances (mesma posição: o
    // tabuleiro fica sempre restaurado após cada makeMove/unmakeMove).
    if (!inCheck && !isMate(bestScore)) {
        updateCorrHist(board, rawEval, bestScore);
        updateNonPawnCorrHist(board, rawEval, bestScore);
    }

    gTT.store(board.hash, bestScore, rawEval, bestMove, depth, bound);
    return bestScore;
}

// Best-move stability: BASE/STEP como int×100/×1000 (UCI spin só dá inteiros;
// fórmula real em stabilityFactor, na busca iterativa abaixo).
static int BM_STABILITY_MAX      = 8;
static int BM_STABILITY_BASE_X100  = 120;   // 1.20
static int BM_STABILITY_STEP_X1000 = 75;    // 0.075

// ─── Parâmetros afináveis por SPSA (training/spsa_tune.py) ─────────────────
// Cada técnica entrou por SPRT com este valor neutro; SPSA só reafina a
// MAGNITUDE à escala da NapK9, nunca decide se a técnica fica (isso é SPRT,
// sempre — disciplina do projeto). Exposição via UCI "option type spin":
// uci.cpp chama printTunableOptions()/setTunableParam() abaixo.
struct TunableParam { const char* name; int* value; int lo; int hi; };
static const TunableParam gTunables[] = {
    { "AspirationDelta",   &ASPIRATION_DELTA,    4,    64   },
    { "RazorBase",         &RAZOR_BASE,          50,   600  },
    { "RazorMult",         &RAZOR_MULT,          50,   600  },
    { "RfpMaxDepth",       &RFP_MAX_DEPTH,       2,    12   },
    { "RfpMargin",         &RFP_MARGIN,          10,   120  },
    { "NmpMinDepth",       &NMP_MIN_DEPTH,       1,    6    },
    { "NmpBaseR",          &NMP_BASE_R,          2,    16   },
    { "NmpDiv",            &NMP_DIV,             2,    16   },
    { "LmrMinDepth",       &LMR_MIN_DEPTH,       1,    6    },
    { "LmrMinMoves",       &LMR_MIN_MOVES,       1,    8    },
    { "LmrCx100",          &LMR_C_X100,          60,   260  },
    { "IirMinDepth",       &IIR_MIN_DEPTH,       2,    10   },
    { "ProbcutMinDepth",   &PROBCUT_MIN_DEPTH,   3,    12   },
    { "ProbcutMargin",     &PROBCUT_MARGIN,      50,   500  },
    { "SeMinDepth",        &SE_MIN_DEPTH,        3,    14   },
    { "SeMargin",          &SE_MARGIN,           10,   250  },
    { "SeDoubleMargin",    &SE_DOUBLE_MARGIN,    4,    60   },
    { "SeTripleMargin",    &SE_TRIPLE_MARGIN,    20,   200  },
    { "LmpMaxDepth",       &LMP_MAX_DEPTH,       2,    16   },
    { "LmpBase",           &LMP_BASE,            1,    20   },
    { "LmpMult",           &LMP_MULT,            1,    8    },
    { "FutilityMaxDepth",  &FUTILITY_MAX_DEPTH,  2,    16   },
    { "FutilityBase",      &FUTILITY_BASE,       10,   300  },
    { "FutilityMargin",    &FUTILITY_MARGIN,     10,   300  },
    { "SeePruneMaxDepth",  &SEE_PRUNE_MAX_DEPTH, 2,    14   },
    { "SeePruneMargin",    &SEE_PRUNE_MARGIN,    50,   500  },
    { "QuietSeePruneMaxDepth", &QUIET_SEE_PRUNE_MAX_DEPTH, 2,  14  },
    { "QuietSeePruneMargin",   &QUIET_SEE_PRUNE_MARGIN,    5,  60  },
    { "HistPruneMaxDepth", &HIST_PRUNE_MAX_DEPTH,2,    16   },
    { "HistPruneMargin",   &HIST_PRUNE_MARGIN,   200,  4000 },
    { "DeltaMargin",       &DELTA_MARGIN,        100,  800  },
    { "BmStabilityMax",    &BM_STABILITY_MAX,    1,    16   },
    { "BmStabilityBaseX100", &BM_STABILITY_BASE_X100,  100, 250 },
    { "BmStabilityStepX1000", &BM_STABILITY_STEP_X1000, 0,   200 },
};
static constexpr int N_TUNABLES = sizeof(gTunables) / sizeof(gTunables[0]);

void printTunableOptions() {
    for (int i = 0; i < N_TUNABLES; ++i) {
        const TunableParam& p = gTunables[i];
        printf("option name %s type spin default %d min %d max %d\n",
               p.name, *p.value, p.lo, p.hi);
    }
}

bool setTunableParam(const std::string& name, int value) {
    for (int i = 0; i < N_TUNABLES; ++i) {
        if (name == gTunables[i].name) {
            *gTunables[i].value = std::max(gTunables[i].lo, std::min(gTunables[i].hi, value));
            if (name == "LmrCx100" || name == "LmrMinDepth" || name == "LmrMinMoves")
                rebuildLmrTable();
            return true;
        }
    }
    return false;
}

// ─── Iterative deepening ──────────────────────────────────────────────────
// Corpo da busca iterativa — corre em QUALQUER thread (principal ou
// helper de Lazy SMP). isMain controla só a impressão UCI (info/bestmove);
// helpers correm exatamente a mesma busca, partilham a TT, e o resultado
// delas é descartado — só ajudam a preencher a TT mais rápido (mesma
// filosofia dos 3 motores de referência, sem stagger de depth: a
// concorrência natural pela TT já basta para diversidade).
static void searchBody(Board& board, const Limits& limits, bool isMain, uint64_t* nodesOut) {
    SearchInfo info;
    info.startMs = nowMs();
    info.nodeLimit = limits.nodes;

    // Time management
    if (limits.movetime > 0) {
        info.timeLimitMs = limits.movetime;
        info.softLimitMs = limits.movetime;
    } else if (limits.wtime > 0 || limits.btime > 0) {
        int myTime = (board.sideToMove() == Color::WHITE) ? limits.wtime : limits.btime;
        int myInc  = (board.sideToMove() == Color::WHITE) ? limits.winc  : limits.binc;
        myTime = std::max(1, myTime - gMoveOverheadMs);
        int moves  = limits.movestogo > 0 ? limits.movestogo : 40;
        int base   = myTime / moves + myInc;
        info.softLimitMs = std::min(base, myTime / 2);
        info.timeLimitMs = std::min(base * 5, myTime * 3 / 4);
    } else if (!limits.infinite) {
        info.timeLimitMs = 5000;  // default 5s
        info.softLimitMs = 5000;
    }

    memset(gKillers, 0, sizeof(gKillers));
    gNmpMinPly = 0;
    gOptimism[0] = gOptimism[1] = 0;
    memset(gContHist1, 0, sizeof(gContHist1));
    memset(gContHist2, 0, sizeof(gContHist2));
    for (int i = 0; i < 130; ++i) { gContPieceAt[i] = int(PieceType::NONE); gContToAt[i] = 0; }
    memset(gHistory, 0, sizeof(gHistory));
    memset(gCaptureHistory, 0, sizeof(gCaptureHistory));

    // Acumulador incremental: refresh completo na raiz (ply 0), depois cada
    // makeMove/unmakeMove só empurra/recua deltas (evalPush/evalPop acima).
    if (napoleon::nnue::napkIncrementalEnabled()) {
        napoleon::nnue::napkRefresh(board, gEvalSlots[0][0]);
        napoleon::nnue::napkRefresh(board, gEvalSlots[0][1]);
        napoleon::nnue::napkSetCurrentSlot(gEvalSlots[0]);
    } else {
        napoleon::nnue::napkSetCurrentSlot(nullptr);
    }

    Move bestMove = NULL_MOVE;
    int maxDepth = limits.depth;
    if (maxDepth <= 0 || maxDepth > 64) maxDepth = 64;
    int avgScore = 0;
    bool haveAvgScore = false;
    int prevScore = 0;
    bool havePrevScore = false;
    // Falling-eval time extension: guarda o score da iteração ANTERIOR (não confundir com
    // prevScore, que já passa a ser o score DESTA iteração mal ela acaba — serve só p/ as
    // janelas de aspiração da iteração seguinte). Gap vs SF/Reckless: ambos usam a tendência
    // do score entre iterações como um dos principais multiplicadores de TM; nós só tínhamos
    // best-move-stability e node-fraction.
    int lastIterScore = 0;
    bool haveLastIterScore = false;
    // Best-move stability: nº de iterações consecutivas em que o melhor
    // lance não mudou — encolhe o tempo alocado quando a decisão já está
    // estável (ver soft time check abaixo). BM_STABILITY_* movidos p/
    // scope de ficheiro (perto de gTunables, abaixo) para SPSA os afinar.
    int bestMoveStability = 0;
    // Verificação PREVENTIVA antes de cada profundidade nova (gap real, não só de Elo):
    // o soft time check só corria DEPOIS de cada iteração terminar — numa posição em que
    // uma profundidade demora 2-4x mais que a anterior (normal em ID), o motor começava-a
    // mesmo perto do limite e só o limite DURO (bem mais generoso) a travava, ultrapassando
    // bastante o tempo pensado para esse lance. Estima o pior caso da próxima iteração
    // (tempo da última × fator de crescimento) e não a começa se isso já passar o soft
    // limit mais recente. effectiveSoft/lastIterMs ficam fora do loop para serem lidos
    // ANTES da iteração seguinte, não só depois da atual.
    int64_t effectiveSoft = info.softLimitMs;
    int64_t lastIterMs = 0;
    static constexpr double ITER_GROWTH_GUESS = 2.0;

    for (int depth = 1; depth <= maxDepth; ++depth) {
        if (depth > 1 && !limits.infinite && effectiveSoft > 0) {
            int64_t elapsed = nowMs() - info.startMs;
            if (elapsed + (int64_t)(lastIterMs * ITER_GROWTH_GUESS) > effectiveSoft)
                break;
        }
        int64_t iterStartMs = nowMs();
        info.stopped = false;
        info.selDepth = 0;  // recomeça a cada iteração (estilo Stockfish: seldepth é por depth, não cumulativo)
        gRootExcludedCount = 0;  // MultiPV: recomeça a exclusão a cada depth nova

        int score;
        if (depth >= ASPIRATION_MIN_DEPTH && havePrevScore) {
            // Janela estreita à volta do score da iteração anterior; alarga
            // progressivamente (delta crescente) sempre que falha fora dela.
            int delta = ASPIRATION_DELTA;
            int alpha = std::max(prevScore - delta, -INF_SCORE);
            int beta  = std::min(prevScore + delta, INF_SCORE);
            for (;;) {
                score = search(board, depth, alpha, beta, 0, true, info);
                if (info.stopped) break;
                // 🦅 FIX (bug real e provavelmente a causa principal do défice de hoje):
                // ao falhar, só se movia o limite que falhou — o OUTRO ficava obsoleto
                // (beta antigo num fail-low, alpha antigo num fail-high), produzindo
                // janelas cada vez mais desalinhadas/degeneradas a cada falha. Confirmado
                // lendo o código real do SF e do Reckless: AMBOS recalculam os DOIS
                // limites em conjunto a cada falha (SF: "beta=alpha; alpha=score-delta"
                // no fail-low; Reckless faz o mesmo com uma fórmula equivalente). Sem
                // isto, esta técnica nunca tinha sido validada (nem testada isolada) —
                // só foi apanhado agora, ao isolar aspiration windows para teste.
                if (score <= alpha) {
                    beta  = alpha;
                    alpha = std::max(score - delta, -INF_SCORE);
                } else if (score >= beta) {
                    alpha = std::max(beta - delta, alpha);
                    beta  = std::min(score + delta, INF_SCORE);
                } else {
                    break;
                }
                delta += delta / 2;
            }
        } else {
            score = search(board, depth, -INF_SCORE, INF_SCORE, 0, true, info);
        }

        if (info.stopped && depth > 1) break;  // discard incomplete iteration

        prevScore = score;
        havePrevScore = true;

        avgScore = haveAvgScore ? (score + avgScore) / 2 : score;
        haveAvgScore = true;
        int us = int(board.sideToMove());
        gOptimism[us]     = 100 * avgScore / (std::abs(avgScore) + 150);
        gOptimism[1 - us] = -gOptimism[us];

        // Retrieve best move from TT
        Move prevBestMove = bestMove;
        bool ttHit;
        TTEntry* tte = gTT.probe(board.hash, ttHit);
        if (ttHit && tte->move) bestMove = Move(tte->move);

        if (depth > 1 && !bestMove.isNull() && bestMove.data == prevBestMove.data)
            bestMoveStability = std::min(bestMoveStability + 1, BM_STABILITY_MAX);
        else
            bestMoveStability = 0;

        int64_t elapsed = nowMs() - info.startMs;
        uint64_t nps = elapsed > 0 ? info.nodes * 1000 / elapsed : info.nodes;

        // Format score
        char scoreStr[32];
        if (isMate(score)) {
            int mateIn = (MATE_SCORE - std::abs(score) + 1) / 2;
            snprintf(scoreStr, sizeof(scoreStr), "mate %d", score > 0 ? mateIn : -mateIn);
        } else {
            snprintf(scoreStr, sizeof(scoreStr), "cp %d", score);
        }

        // PV completa via tabela triangular (gap real: só mostrava o primeiro lance
        // antes). gPvTable[0]/gPvLength[0] refletem a melhor linha encontrada na raiz
        // nesta iteração.
        char pv[512] = {};
        if (!bestMove.isNull()) {
            int n = formatPv(gPvTable[0], gPvLength[0], pv, sizeof(pv));
            if (n == 0) formatMoveUci(bestMove, pv);  // PV vazia por algum motivo raro — fallback ao 1º lance
        }

        if (isMain) {
            napoleon::wdl::Probs wdl = napoleon::wdl::expectedWDL(score);
            printf("info depth %d seldepth %d multipv 1 score %s wdl %d %d %d nodes %llu nps %llu time %lld pv %s\n",
                   depth, info.selDepth, scoreStr,
                   (int)std::lround(wdl.win * 1000.0), (int)std::lround(wdl.draw * 1000.0), (int)std::lround(wdl.loss * 1000.0),
                   (unsigned long long)info.nodes,
                   (unsigned long long)nps, (long long)elapsed, pv);
            fflush(stdout);
        }

        // MultiPV (análise; gMultiPV=1 default → este bloco nunca corre,
        // comportamento idêntico a antes desta técnica existir). Cada linha
        // extra exclui da raiz os lances já reportados nesta profundidade —
        // mesmo truque do excludedMove das extensões singulares, só que por
        // um array de ficheiro (gRootExcluded) em vez de um parâmetro, para
        // não tocar na assinatura de search() usada em todo o resto do código.
        // ⚠️ LIMITAÇÃO CONHECIDA: como cada linha é uma busca de raiz à parte
        // (não 1 busca só com N linhas extraídas da PV), gHistory/gContHist/TT
        // ficam "contaminados" pelas passadas anteriores na MESMA depth — os
        // scores das linhas 2+ por vezes não saem em ordem decrescente
        // estrita. Inofensivo em jogo (MultiPV=1 nunca entra aqui), só afeta
        // a leitura da análise. Corrigir a sério exigiria isolar o estado de
        // ordenação por linha — fora do âmbito desta funcionalidade utilitária.
        if (isMain && gMultiPV > 1 && !bestMove.isNull() && !info.stopped) {
            gRootExcluded[0] = bestMove;
            gRootExcludedCount = 1;
            for (int pvIdx = 1; pvIdx < gMultiPV; ++pvIdx) {
                int pvScore = search(board, depth, -INF_SCORE, INF_SCORE, 0, true, info);
                if (info.stopped) break;

                bool pvHit;
                TTEntry* pvTte = gTT.probe(board.hash, pvHit);
                Move pvMove = (pvHit && pvTte->move) ? Move(pvTte->move) : NULL_MOVE;
                if (pvMove.isNull()) break;  // menos lances legais que MultiPV pedido

                char pvScoreStr[32];
                if (isMate(pvScore)) {
                    int mateIn = (MATE_SCORE - std::abs(pvScore) + 1) / 2;
                    snprintf(pvScoreStr, sizeof(pvScoreStr), "mate %d", pvScore > 0 ? mateIn : -mateIn);
                } else {
                    snprintf(pvScoreStr, sizeof(pvScoreStr), "cp %d", pvScore);
                }
                char pvStr[512] = {};
                int pvN = formatPv(gPvTable[0], gPvLength[0], pvStr, sizeof(pvStr));
                if (pvN == 0) formatMoveUci(pvMove, pvStr);
                int64_t pvElapsed = nowMs() - info.startMs;
                uint64_t pvNps = pvElapsed > 0 ? info.nodes * 1000 / pvElapsed : info.nodes;
                napoleon::wdl::Probs pvWdl = napoleon::wdl::expectedWDL(pvScore);
                printf("info depth %d seldepth %d multipv %d score %s wdl %d %d %d nodes %llu nps %llu time %lld pv %s\n",
                       depth, info.selDepth, pvIdx + 1, pvScoreStr,
                       (int)std::lround(pvWdl.win * 1000.0), (int)std::lround(pvWdl.draw * 1000.0), (int)std::lround(pvWdl.loss * 1000.0),
                       (unsigned long long)info.nodes,
                       (unsigned long long)pvNps, (long long)pvElapsed, pvStr);
                fflush(stdout);

                gRootExcluded[gRootExcludedCount++] = pvMove;
            }
            gRootExcludedCount = 0;  // não afeta a próxima depth (já reposto no topo do for, dupla garantia)
        }

        // Soft time check — modulado pelo WDL brain (opt-in, OFF por
        // defeito), pela best-move stability (lance estável há várias
        // profundidades → encolhe o tempo) E pela fração de nós no melhor
        // lance de raiz (estilo Stockfish nodesEffort: se ele comeu quase
        // todos os nós desta iteração, a decisão já está bem resolvida).
        double stabilityFactor = BM_STABILITY_BASE_X100 / 100.0
                                - (BM_STABILITY_STEP_X1000 / 1000.0) * bestMoveStability;
        uint64_t bestMoveNodes = 0, iterTotalNodes = 0;
        for (int ri = 0; ri < gRootMoveCount; ++ri) {
            iterTotalNodes += gRootMoveNodes[ri];
            if (gRootMoves[ri].data == bestMove.data) bestMoveNodes = gRootMoveNodes[ri];
        }
        double nodeFraction = iterTotalNodes > 0 ? (double)bestMoveNodes / iterTotalNodes : 0.0;
        double nodeFactor = std::max(0.7, std::min(1.3, 1.5 - nodeFraction));
        // Falling-eval: se o score piorou bastante desde a iteração anterior, a posição
        // pode estar a degradar-se ou a busca a encontrar algo inesperado — vale a pena
        // gastar mais tempo. Score melhor/igual → sem alteração (não encolhe tempo aqui,
        // o nodeFactor/stability já tratam de "decisão clara"). Escala 0.1 por cada 50cp de
        // queda, capado em ±0.3 — constante própria, nasce neutra.
        double fallingEvalFactor = 1.0;
        if (haveLastIterScore && !isMate(score) && !isMate(lastIterScore)) {
            int drop = lastIterScore - score;  // >0 quando o score piorou
            if (drop > 0)
                fallingEvalFactor = std::min(1.3, 1.0 + 0.1 * (drop / 50.0));
        }
        lastIterScore = score;
        haveLastIterScore = true;
        effectiveSoft = (int64_t)(info.softLimitMs * stabilityFactor * nodeFactor * fallingEvalFactor);
        if (napoleon::wdlbrain::g_config.enabled)
            effectiveSoft = (int64_t)(effectiveSoft * napoleon::wdlbrain::timeFactor(board, score));
        lastIterMs = nowMs() - iterStartMs;
        if (!limits.infinite && effectiveSoft > 0
            && nowMs() - info.startMs >= effectiveSoft) break;
        // Lance único na raiz: gap vs SF — não há decisão nenhuma a tomar (o lance é
        // forçado), continuar a aprofundar só gasta o tempo do relógio sem mudar o
        // resultado. Ainda corre 1 iteração completa (info/PV normais), só evita as
        // restantes.
        if (!limits.infinite && gRootMoveCount == 1) break;
    }
    (void)0;  // suppress unused warning

    // Liberta o slot incremental: um 'eval'/'d' standalone depois deste 'go'
    // não deve herdar um slot de uma posição/ply diferente — cai no caminho
    // antigo (plyResolve), sempre correto independentemente do board atual.
    napoleon::nnue::napkSetCurrentSlot(nullptr);

    // Output best move
    char mv[8] = "0000";
    if (!bestMove.isNull()) {
        mv[0] = 'a' + (bestMove.from() & 7);
        mv[1] = '1' + (bestMove.from() >> 3);
        mv[2] = 'a' + (bestMove.to() & 7);
        mv[3] = '1' + (bestMove.to() >> 3);
        if (bestMove.isPromo()) {
            const char pc[] = "nbrq";
            mv[4] = pc[bestMove.flags() & 3];
            mv[5] = '\0';
        } else {
            mv[4] = '\0';
        }
    }
    if (isMain) {
        printf("bestmove %s\n", mv);
        fflush(stdout);
    }

    if (nodesOut) *nodesOut = info.nodes;
}

// ─── Wrapper público: dispara os helpers de Lazy SMP (gThreads-1) e corre
//    a busca principal nesta própria thread. gTT.newSearch() e o reset do
//    stop global acontecem AQUI, UMA só vez (não por-thread). Os helpers
//    recebem cada um a SUA cópia do board (Board é trivialmente copiável,
//    sem ponteiros) — só a thread principal toca no board do chamador.
void search(Board& board, const Limits& limits, uint64_t* nodesOut) {
    gTT.newSearch();
    gGlobalStop.store(false, std::memory_order_relaxed);

    int nHelpers = std::max(0, gThreads - 1);
    std::vector<Board> helperBoards(nHelpers, board);
    std::vector<std::thread> helpers;
    helpers.reserve(nHelpers);
    for (int i = 0; i < nHelpers; ++i)
        helpers.emplace_back(searchBody, std::ref(helperBoards[i]), std::cref(limits), false, nullptr);

    searchBody(board, limits, true, nodesOut);

    gGlobalStop.store(true, std::memory_order_relaxed);
    for (auto& t : helpers) t.join();
}

void requestStop() { gGlobalStop.store(true, std::memory_order_relaxed); }

int seeValue(const Board& board, Move m) {
    int lo = -2000, hi = 2000;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (seeGE(board, m, mid)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// ─── Validação do acumulador incremental (UCI "incrtest") ─────────────────
// Percorre TODOS os lances legais até `maxDepth` plies (estilo perft), e em
// CADA nó compara o eval pelo slot incremental novo com o eval pelo caminho
// antigo (plyResolve — já provado correto por "threattest"). Uma divergência
// nesta árvore identifica exatamente o ply/FEN onde o push/pop tem um bug,
// em vez de só "o motor jogou mal" lá na frente, sem se saber porquê.
static void napkIncrementalWalk(Board& board, int ply, int maxDepth, int& mismatches) {
    if (mismatches > 0) return;   // já encontrámos um bug — não vale a pena continuar

    napoleon::nnue::napkSetCurrentSlot(gEvalSlots[ply]);
    napoleon::nnue::napkSetIncremental(true);
    int incScore = napoleon::nnue::evaluate(board);
    napoleon::nnue::napkSetIncremental(false);
    int refScore = napoleon::nnue::evaluate(board);
    napoleon::nnue::napkSetIncremental(true);

    if (incScore != refScore) {
        std::printf("incrtest MISMATCH ply=%d inc=%d ref=%d fen=%s\n",
                     ply, incScore, refScore, board.toFen().c_str());
        ++mismatches;
        return;
    }
    if (ply >= maxDepth || ply + 1 >= 259) return;

    MoveList list;
    generateMoves(board, list);
    for (int i = 0; i < list.count && mismatches == 0; ++i) {
        Move m = list.moves[i];
        if (!board.isLegal(m)) continue;
        MoveDelta delta = computeMoveDelta(board, m);
        board.makeMove(m);
        evalPush(board, delta, ply);
        napkIncrementalWalk(board, ply + 1, maxDepth, mismatches);
        evalPop(ply);
        board.unmakeMove(m);
    }
}

int napkIncrementalSelfTest(Board& board, int depth) {
    int mismatches = 0;
    napoleon::nnue::napkSetIncremental(true);
    napoleon::nnue::napkRefresh(board, gEvalSlots[0][0]);
    napoleon::nnue::napkRefresh(board, gEvalSlots[0][1]);
    napkIncrementalWalk(board, 0, depth, mismatches);
    napoleon::nnue::napkSetCurrentSlot(nullptr);
    return mismatches;
}
