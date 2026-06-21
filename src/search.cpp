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
        int64_t now = nowMs();
        if (info.timeLimitMs > 0 && now >= info.startMs + info.timeLimitMs) {
            info.stopped = true;
            return true;
        }
        // Iteração atual a demorar muito mais que o esperado (explosão real, não
        // crescimento normal) -- aborta SÓ esta depth, driver descarta e usa o melhor
        // resultado já conhecido da anterior. Funciona mesmo sem time control (go
        // infinite/go depth N), onde timeLimitMs acima nem está definido.
        if (info.iterDeadlineMs > 0 && now >= info.iterDeadlineMs) {
            info.stopped = true;
            return true;
        }
    }
    return false;
}

// Declarada mais abaixo (junto às outras funções de ataque/SEE); forward só p/ o
// fallback material+threats de staticEval poder reutilizá-la sem reordenar o ficheiro.
static Bitboard computeSideAttacks(const Board& board, Color side);

// 🦅 Tapered Eval (HCE fase 4): par {mg,eg} interpolado pela fase do jogo (material
// restante), em vez de um único valor por termo. Resolve a anomalia encontrada na 1ª
// calibragem sem fase (Texel tuning linear absorvia "peão em d2" vs "peão em d4" de
// forma invertida, por correlação espúria entre a casa do peão e a fase do jogo nos
// dados -- um único peso não consegue distinguir "isto é bom porque é cedo no jogo" de
// "isto é bom porque é esta casa"). Todas as tabelas abaixo foram inicializadas
// REPLICANDO o valor único da calibragem anterior em AMBAS as fases (mg=eg=valor antigo)
// -- placeholder até o próximo retreino do texel_tuner (que agora extrai dois pesos por
// feature, ponderados pela fase de CADA posição) diferenciar mg/eg de facto.
struct Score {
    int mg = 0, eg = 0;
    Score& operator+=(Score o) { mg += o.mg; eg += o.eg; return *this; }
    Score& operator-=(Score o) { mg -= o.mg; eg -= o.eg; return *this; }
    Score operator-() const { return {-mg, -eg}; }
};
static inline Score operator*(Score s, int k) { return {s.mg * k, s.eg * k}; }

// Pesos de fase clássicos (Fruit/SF antigo): peão=0, menor=1, torre=2, dama=4; rei não
// conta. MAX_PHASE = 4*1(cavalos)+4*1(bispos)+4*2(torres)+2*4(damas) = 24.
static constexpr int kPhaseWeight[6] = { 0, 1, 1, 2, 4, 0 };
static constexpr int MAX_PHASE = 24;
static int gamePhase(const Board& board) {
    int phase = 0;
    for (int side = 0; side < 2; ++side)
        for (int pt = 0; pt < 6; ++pt)
            phase += board.pieceBB[side][pt].popcount() * kPhaseWeight[pt];
    return std::min(phase, MAX_PHASE);
}

// PSQTs calibradas via Texel tuning (training/texel_tuner, Rust) a partir de 3.6 BILIÕES
// de posições reais do binpack test80-2024-06-jun-2tb7p.min-v2.v6 -- ficheiro inteiro, sem
// filtro de ply (abertura, meio-jogo, finais e posições de mate todos incluídos), com fit
// de K via line search (coarse-to-fine, mesma ideia do computeOptimalK do Ethereal,
// src/tuner.c) em vez do K=1/400 hardcoded das versões anteriores. Cada valor já inclui o
// "valor de material implícito" da peça (a tabela foi treinada do zero, pesos iniciais=0,
// sem termo de material separado) -- por isso staticEval() NÃO soma um termo de material à
// parte, só estas tabelas + os termos de threats (também calibrados juntos).
// Tabelas da perspetiva das BRANCAS (a8=0 .. h1=63); pretas leem espelhado (sq^56).
static const Score kPsqtPawn[64] = {
    {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
    {59,161}, {64,152}, {46,158}, {47,172}, {61,152}, {84,142}, {91,130}, {64,125},
    {75,136}, {67,139}, {66,134}, {57,117}, {76,121}, {89,131}, {93,119}, {73,106},
    {71,146}, {66,147}, {94,124}, {91,104}, {97,117}, {97,105}, {89,132}, {69,117},
    {89,149}, {99,151}, {95,136}, {102,116}, {106,132}, {134,130}, {96,135}, {102,129},
    {95,180}, {94,193}, {106,165}, {122,140}, {135,136}, {144,139}, {105,154}, {147,148},
    {30,129}, {-11,143}, {69,112}, {66,90}, {62,100}, {-1,96}, {-11,113}, {31,96},
    {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
};
static const Score kPsqtKnight[64] = {
    {-16,74}, {-23,84}, {-28,89}, {-11,105}, {-21,86}, {-21,62}, {-38,104}, {-62,116},
    {-40,99}, {-23,86}, {-18,78}, {9,76}, {22,77}, {-9,86}, {-25,91}, {-19,69},
    {-25,89}, {19,81}, {8,106}, {19,107}, {26,95}, {19,98}, {31,80}, {-20,103},
    {24,97}, {29,103}, {30,112}, {30,121}, {31,135}, {47,114}, {34,100}, {18,117},
    {27,103}, {29,118}, {29,123}, {46,126}, {29,126}, {39,126}, {42,123}, {63,107},
    {44,91}, {51,97}, {31,117}, {45,122}, {45,101}, {69,126}, {37,118}, {92,89},
    {33,75}, {33,89}, {85,92}, {46,124}, {95,123}, {99,82}, {74,93}, {91,80},
    {109,42}, {63,103}, {12,108}, {50,81}, {88,116}, {109,76}, {138,82}, {53,37},
};
static const Score kPsqtBishop[64] = {
    {16,96}, {59,115}, {-4,98}, {-13,107}, {-17,106}, {-18,109}, {6,92}, {1,103},
    {41,108}, {21,91}, {41,97}, {5,117}, {25,101}, {3,93}, {37,91}, {44,111},
    {30,122}, {44,126}, {19,110}, {41,116}, {21,114}, {14,110}, {36,112}, {37,106},
    {31,104}, {16,108}, {26,121}, {14,116}, {27,113}, {33,130}, {23,107}, {54,107},
    {35,103}, {40,128}, {32,125}, {23,126}, {48,136}, {41,123}, {37,132}, {46,110},
    {12,113}, {30,112}, {-14,133}, {15,137}, {14,123}, {-9,150}, {74,118}, {64,101},
    {6,116}, {15,120}, {61,120}, {27,128}, {2,122}, {46,100}, {-27,114}, {55,96},
    {28,85}, {23,124}, {-28,124}, {15,129}, {28,124}, {46,108}, {6,87}, {62,91},
};
static const Score kPsqtRook[64] = {
    {15,195}, {6,196}, {29,200}, {36,196}, {28,182}, {21,189}, {-14,197}, {8,179},
    {-8,188}, {-6,196}, {13,198}, {25,188}, {11,181}, {14,189}, {-6,168}, {-34,199},
    {-11,195}, {14,198}, {15,214}, {24,199}, {15,189}, {26,193}, {58,185}, {22,193},
    {20,206}, {30,208}, {21,214}, {30,207}, {21,208}, {30,223}, {54,183}, {41,181},
    {38,216}, {35,213}, {48,212}, {31,214}, {45,215}, {67,205}, {81,194}, {68,190},
    {34,218}, {54,208}, {40,226}, {55,215}, {59,207}, {87,212}, {99,199}, {73,200},
    {56,200}, {55,198}, {61,202}, {65,216}, {42,202}, {101,214}, {68,199}, {93,194},
    {-19,235}, {10,223}, {-55,244}, {-78,241}, {-52,239}, {-4,211}, {75,204}, {75,201},
};
static const Score kPsqtQueen[64] = {
    {-66,384}, {-56,414}, {-69,396}, {-59,433}, {-62,370}, {-87,366}, {-48,361}, {-81,371},
    {-66,394}, {-56,400}, {-44,414}, {-48,398}, {-40,396}, {-41,380}, {-55,332}, {-40,337},
    {-46,403}, {-56,426}, {-48,458}, {-57,434}, {-48,456}, {-48,433}, {-29,406}, {-18,379},
    {-48,403}, {-51,434}, {-47,456}, {-68,497}, {-36,474}, {-34,458}, {-18,413}, {-20,435},
    {-35,410}, {-39,441}, {-46,453}, {-44,474}, {-32,491}, {-16,434}, {-28,430}, {-29,409},
    {-69,408}, {-63,459}, {-35,449}, {-26,466}, {-34,491}, {-30,457}, {-12,422}, {-28,434},
    {-45,405}, {-64,428}, {-30,480}, {-87,511}, {-65,490}, {-30,435}, {-48,430}, {5,396},
    {3,354}, {15,377}, {-30,422}, {-163,502}, {-36,435}, {56,359}, {89,302}, {107,303},
};
static const Score kPsqtKing[64] = {
    {-9,-20}, {39,-29}, {7,-18}, {-88,3}, {-27,-19}, {-67,5}, {19,-16}, {-7,-25},
    {38,-15}, {29,-12}, {-21,4}, {-58,-2}, {-55,3}, {-31,-5}, {9,-17}, {-8,-21},
    {17,-24}, {52,-16}, {6,-11}, {-15,5}, {-25,-6}, {-28,-2}, {1,-26}, {-21,-16},
    {42,-23}, {72,-2}, {30,-1}, {23,17}, {-3,-1}, {-3,-3}, {25,1}, {-19,-18},
    {54,-10}, {89,5}, {42,24}, {-3,32}, {4,36}, {-6,29}, {43,3}, {-9,1},
    {87,-5}, {124,6}, {92,34}, {37,38}, {-2,51}, {68,46}, {48,29}, {52,28},
    {141,-32}, {163,3}, {143,16}, {83,37}, {55,43}, {110,34}, {66,29}, {39,7},
    {63,-9}, {124,-12}, {126,12}, {92,22}, {48,25}, {69,33}, {53,30}, {-7,24},
};
static const Score* const kPsqt[6] = { kPsqtPawn, kPsqtKnight, kPsqtBishop, kPsqtRook, kPsqtQueen, kPsqtKing };

// HCE (sem rede NNUE), fase 2: termos de threats inspirados na estrutura conceptual do
// threats() do Stockfish (src/evaluate.cpp, era clássica sf_12..sf_16 -- ideias/lista de
// termos estudados na fonte real, reimplementados aqui do zero, pesos próprios calibrados
// via Texel tuning, não copiados do SF). Subconjunto dos 11 termos reais: os 7 de maior
// impacto (ThreatByMinor, ThreatByRook, Hanging, ThreatBySafePawn, RestrictedPiece,
// WeakQueenProtection, ThreatByKing); ficam de fora por agora KnightOnQueen, SliderOnQueen,
// ThreatByPawnPush, WeakQueen (mais raros/marginais).
struct AttackInfo {
    Bitboard byPawn, byKnight, byBishop, byRook, byQueen, byKing, all, all2;
};
static AttackInfo computeAttackInfo(const Board& board, Color side) {
    AttackInfo info;
    Bitboard occ = board.allOcc;
    info.byPawn = (side == Color::WHITE)
        ? attacks::pawnAttacks<Color::WHITE>(board.pieces(side, PieceType::PAWN))
        : attacks::pawnAttacks<Color::BLACK>(board.pieces(side, PieceType::PAWN));
    info.byKnight = Bitboard(0ULL);
    Bitboard bb = board.pieces(side, PieceType::KNIGHT);
    while (bb.any()) info.byKnight |= attacks::knightAttacks(bb.poplsb());
    info.byBishop = Bitboard(0ULL);
    bb = board.pieces(side, PieceType::BISHOP);
    while (bb.any()) info.byBishop |= attacks::bishopAttacks(bb.poplsb(), occ);
    info.byRook = Bitboard(0ULL);
    bb = board.pieces(side, PieceType::ROOK);
    while (bb.any()) info.byRook |= attacks::rookAttacks(bb.poplsb(), occ);
    info.byQueen = Bitboard(0ULL);
    bb = board.pieces(side, PieceType::QUEEN);
    while (bb.any()) { Square qs = bb.poplsb(); info.byQueen |= attacks::bishopAttacks(qs, occ) | attacks::rookAttacks(qs, occ); }
    info.byKing = attacks::kingAttacks(board.kingSq(side));
    info.all = info.byPawn | info.byKnight | info.byBishop | info.byRook | info.byQueen | info.byKing;
    // Aproximação de attackedBy2: une as interseções par-a-par entre categorias de peça.
    // Não cobre 2 peões diferentes a atacar a mesma casa (within-category) -- imprecisão
    // pequena e aceitável aqui, não precisa de ser bit-exato ao SF.
    Bitboard cats[6] = { info.byPawn, info.byKnight, info.byBishop, info.byRook, info.byQueen, info.byKing };
    info.all2 = Bitboard(0ULL);
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            info.all2 |= (cats[i] & cats[j]);
    return info;
}
// Pesos treináveis via texel_tuner (training/texel_tuner/), valores iniciais = (mg=eg=
// valor da calibragem anterior sem fase) até ao próximo retreino com tapered eval.
struct ThreatWeights {
    Score threatByMinor[6] = {{7,6},{20,35},{43,25},{6,122},{-54,490},{-37,-87}};
    Score threatByRook[6]  = {{10,20},{-2,32},{13,45},{-3,16},{-93,483},{-63,-119}};
    Score threatByKing      = {54,50};
    Score hanging           = {23,43};
    Score weakQueenProt     = {8,-7};
    Score restrictedPiece   = {4,-3};
    Score threatBySafePawn  = {40,137};
    // 🦅 Completam os 11 termos reais do threats() clássico do SF (sf_12..sf_16) -- os 4
    // que faltavam (mais raros/marginais, mas o motor agora vai ao máximo, não fica a
    // meio): ThreatByPawnPush, KnightOnQueen, SliderOnQueen, WeakQueen.
    Score threatByPawnPush  = {8,13};
    Score knightOnQueen     = {5,5};
    Score sliderOnQueen     = {3,0};
    Score weakQueen         = {-13,8};
};
static const ThreatWeights kThreatW;

// HCE fase 3: Mobility geral -- tipicamente o termo de maior ganho único depois de
// material+PSQT em HCEs maduros (Ethereal, SF clássico). Para cada peça menor/maior
// (Knight/Bishop/Rook/Queen -- não Pawn/King, que têm dinâmicas próprias), conta quantas
// casas da "área de mobilidade" ela ataca: exclui casas com peças PRÓPRIAS e casas
// atacadas por PEÕES INIMIGOS (consideradas perigosas mesmo vazias, convenção SF/Ethereal
// clássica). Tamanhos das tabelas = máximo de casas alcançáveis por tipo de peça
// (Knight≤8, Bishop≤13, Rook≤14, Queen≤27 -- +1 cada p/ incluir o 0).
struct MobilityWeights {
    Score knight[9]  = {{3,147},{27,198},{25,232},{28,245},{33,259},{40,260},{49,257},{42,253},{39,259}};
    Score bishop[14] = {{11,221},{33,212},{51,232},{57,244},{58,256},{56,261},{58,271},{62,259},{56,261},{49,259},{59,262},{86,259},{70,260},{104,249}};
    Score rook[15]   = {{18,381},{35,361},{39,362},{43,373},{38,391},{43,388},{38,403},{37,398},{30,404},{31,406},{36,411},{37,409},{35,409},{46,396},{58,392}};
    Score queen[28]  = {{-132,926},{-120,839},{-111,760},{-103,755},{-102,788},{-101,768},{-100,782},{-93,795},{-91,791},{-85,784},{-90,799},{-83,806},{-93,799},{-93,810},{-93,815},{-98,799},{-93,804},{-93,795},{-82,788},{-64,773},{-41,756},{-49,751},{-8,687},{93,644},{71,632},{136,604},{320,446},{231,546}};
};
static const MobilityWeights kMobilityW;

// HCE fase 5: King Safety -- zona de perigo à volta do rei (king ring: a própria casa +
// as 8 adjacentes), conta ataques inimigos por tipo de peça nessa zona, pesados e
// indexados numa tabela não-linear (mesma ideia clássica SF/Ethereal: o perigo cresce
// mais que linearmente com o número de atacantes -- 1 atacante é normal, 4+ é critico).
// Separadamente, penaliza falta de peões-escudo nas 3 casas em frente ao rei (roque
// destruído/exposto).
struct KingSafetyWeights {
    // 🦅 FIX: índices 33-40 (muitas "unidades de ataque" simultâneas, RARO no dataset --
    // posições com 6+ peças a atacar a king ring ao mesmo tempo) saíam do treino com
    // valores absurdos ({-1354,2198} no índice 40!), comparados com vizinhos estáveis
    // ({-210,149} no 32, {-293,577} no 41) -- ruído de poucos dados nesses buckets, não
    // sinal real. Esses saltos bruscos entre índices ADJACENTES quebram a suavidade que
    // RFP/NMP/etc precisam para podar bem, causando uma explosão real de nós (depth12
    // >15x depth11 numa posição encontrada na validação). Suavizado por interpolação
    // linear entre os pontos estáveis vizinhos (32 e 41) + cap em ±300mg/±400eg.
    Score attackUnits[50] = {{17,-25},{30,18},{14,-7},{19,17},{21,-2},{23,0},{9,7},{13,-2},{12,-9},{17,-25},{26,-5},{10,-9},{24,-41},{-13,-22},{-5,-42},{19,-57},{-3,-41},{-7,-63},{-12,-63},{-20,-85},{-50,-4},{-57,-15},{-53,-41},{-55,-56},{-58,-48},{-21,-54},{-116,15},{-60,-23},{-123,8},{-127,7},{-127,19},{-128,31},{-210,149},{-219,197},{-228,244},{-238,292},{-247,339},{-256,387},{-265,400},{-275,400},{-284,400},{-293,400},{-158,247},{-150,135},{-46,171},{-74,15},{-47,29},{-42,12},{-27,-7},{-12,26}};
    Score pawnShieldMissing[4] = {{15,-12},{5,-4},{-7,0},{-14,4}};
    // 🦅 Safe check detection (Ethereal real, src/evaluate.c): distingue "muitos
    // atacantes sem entrada" de "rede de mate disponível" -- conta, por tipo de peça
    // inimiga, quantas casas de onde ela DARIA XEQUE ao nosso rei estão "safe" para ela
    // (atacada pelo inimigo, não suficientemente defendida por nós).
    Score safeCheck[4] = {{-15,-16},{-14,-12},{-5,-17},{-14,-13}};  // queen, rook, bishop, knight
};
static const KingSafetyWeights kKingSafetyW;

// HCE fase 6: Pawn Structure -- peões passados (bónus crescente por rank, mais valioso
// perto da promoção), isolados (sem peões próprios nas colunas adjacentes, qualquer
// rank) e dobrados (2+ peões próprios na mesma coluna). Backward pawn fica de fora por
// agora (mais complexo de definir corretamente -- precisa de saber se a casa de avanço
// está controlada pelo inimigo E se nenhum peão adjacente já avançou).
struct PawnStructureWeights {
    Score passed[8]   = {{0,0},{-93,-20},{-102,0},{-94,35},{-64,47},{-32,90},{53,234},{0,0}};
    Score isolated    = {-11,-2};
    Score doubled     = {-17,-15};
    // 🦅 Completam o quadro clássico de pawn structure (inspirados no Ethereal real,
    // src/evaluate.c, reimplementados do zero com pesos próprios):
    Score backward       = {-10,-1};  // peão sem vizinhos atrás dele, casa de avanço atacada por peão inimigo
    Score candidatePasser = {1,2}; // não passado ainda, mas ficaria passado depois duma troca planeada (simplificado)
    Score passedKingDist[2] = {{12,-17},{3,8}}; // [0]=distância ao NOSSO rei, [1]=distância ao rei inimigo
    Score passedSafeAdvance = {-7,23}; // a casa de avanço do peão passado não está ocupada nem atacada pelo inimigo
};
static const PawnStructureWeights kPawnStructW;

// Bispo/cavalo: par de bispos, bispo na diagonal longa central (sem bloqueio), peões
// "rammed" (travados, mesma cor da casa do bispo) -- nenhum destes existia antes,
// confirmado zero no nosso código contra o Ethereal real.
struct BishopWeights {
    Score pair          = {23,50};
    Score longDiagonal   = {7,0};
    Score rammedPawn     = {-5,-12};  // por peão próprio "rammed" (bloqueado por peão inimigo) na cor do bispo
};
static const BishopWeights kBishopW;

// Outpost: cavalo/bispo numa casa defendida por peão próprio, inalcançável por peões
// inimigos (nenhum peão inimigo pode chegar a uma casa que o ataque), ranks 4-6.
struct OutpostWeights {
    Score knight = {23,0};
    Score bishop = {36,-6};
};
static const OutpostWeights kOutpostW;

// Rook: coluna aberta (sem peões de ninguém) / semi-aberta (sem peão próprio, peão
// inimigo presente) / 7ª fila (só conta se o rei inimigo ainda está nas 2 últimas filas).
struct RookWeights {
    Score openFile     = {37,-4};
    Score semiOpenFile = {25,4};
    Score seventhRank  = {-20,40};
};
static const RookWeights kRookW;

static Score kTempoBonus = {91,71};  // bónus por ser a vez de jogar -- par mg/eg, ver uso em staticEval()
static Score computePawnStructureScore(const Board& board, Color side) {
    Bitboard ownPawns = board.pieces(side, PieceType::PAWN);
    Bitboard enemyPawns = board.pieces(~side, PieceType::PAWN);
    Square ownKing = board.kingSq(side);
    Square enemyKing = board.kingSq(~side);
    Bitboard occ = board.allOcc;
    Score score;
    Bitboard bb = ownPawns;
    while (bb.any()) {
        Square sq = bb.poplsb();
        int file = sq.file(), rank = sq.rank();

        // Doubled: outro peão próprio na MESMA coluna.
        bool doubledHere = false;
        for (int r = 0; r < 8; ++r) {
            if (r == rank) continue;
            if (ownPawns.test(Square((r << 3) | file).value())) { doubledHere = true; break; }
        }
        if (doubledHere) score += kPawnStructW.doubled;

        // Isolated: nenhum peão próprio nas colunas adjacentes (qualquer rank).
        bool hasNeighbor = false;
        for (int df = -1; df <= 1; df += 2) {
            int f = file + df;
            if (f < 0 || f > 7) continue;
            for (int r = 0; r < 8 && !hasNeighbor; ++r)
                if (ownPawns.test(Square((r << 3) | f).value())) hasNeighbor = true;
        }
        if (!hasNeighbor) score += kPawnStructW.isolated;

        // Backward: tem vizinhos (não isolado), mas nenhum peão próprio nas colunas
        // [file-1,file,file+1] está "atrás" dele (mesma rank ou mais atrás) -- é o mais
        // atrasado do grupo -- E a casa de avanço imediata é atacada por peão inimigo.
        if (hasNeighbor) {
            bool mostBackward = true;
            for (int df = -1; df <= 1 && mostBackward; ++df) {
                int f = file + df;
                if (f < 0 || f > 7) continue;
                int rStart = (side == Color::WHITE) ? 0 : rank;
                int rEnd   = (side == Color::WHITE) ? rank : 8;
                for (int r = rStart; r < rEnd; ++r) {
                    if (df == 0 && r == rank) continue;
                    if (ownPawns.test(Square((r << 3) | f).value())) { mostBackward = false; break; }
                }
            }
            if (mostBackward) {
                int advRank = (side == Color::WHITE) ? rank + 1 : rank - 1;
                bool advanceAttacked = false;
                if (advRank >= 0 && advRank < 8) {
                    // Casa de avanço atacada por peão inimigo: testa diretamente as duas
                    // diagonais de captura inimigas que apontam para essa casa.
                    int ef = file - 1, ef2 = file + 1;
                    int enemyRank = (side == Color::WHITE) ? advRank - 1 : advRank + 1;
                    if (enemyRank >= 0 && enemyRank < 8) {
                        if (ef >= 0 && enemyPawns.test(Square((enemyRank << 3) | ef).value())) advanceAttacked = true;
                        if (ef2 <= 7 && enemyPawns.test(Square((enemyRank << 3) | ef2).value())) advanceAttacked = true;
                    }
                }
                if (advanceAttacked) score += kPawnStructW.backward;
            }
        }

        // Passed: nenhum peão inimigo nas colunas [file-1,file,file+1], em qualquer rank
        // "à frente" deste peão (rank maior para brancas, menor para pretas).
        bool passed = true;
        bool ownFileBlocked = false;
        // 🦅 FIX: o "&& passed" aqui cortava o loop a meio quando df=-1 já marcava
        // passed=false, NUNCA chegando a df=0 -- ownFileBlocked ficava sempre false
        // mesmo quando a própria coluna TINHA um bloqueador (confirmado divergente do
        // Rust tuner, que não tem este early-exit, via revisão estrutural do Opus).
        for (int df = -1; df <= 1; ++df) {
            int f = file + df;
            if (f < 0 || f > 7) continue;
            int rStart = (side == Color::WHITE) ? rank + 1 : 0;
            int rEnd   = (side == Color::WHITE) ? 8 : rank;
            for (int r = rStart; r < rEnd; ++r)
                if (enemyPawns.test(Square((r << 3) | f).value())) {
                    passed = false;
                    if (df == 0) ownFileBlocked = true;
                }
        }
        if (passed) {
            int relRank = (side == Color::WHITE) ? rank : 7 - rank;
            score += kPawnStructW.passed[relRank];
            int distOwn = std::max(std::abs(file - ownKing.file()), std::abs(rank - ownKing.rank()));
            int distEnemy = std::max(std::abs(file - enemyKing.file()), std::abs(rank - enemyKing.rank()));
            score += kPawnStructW.passedKingDist[0] * distOwn;
            score += kPawnStructW.passedKingDist[1] * distEnemy;
            int advRank = (side == Color::WHITE) ? rank + 1 : rank - 1;
            if (advRank >= 0 && advRank < 8 && !occ.test(Square((advRank << 3) | file).value()))
                score += kPawnStructW.passedSafeAdvance;
        } else if (!ownFileBlocked) {
            // Candidate passer: só está "bloqueado" pelas colunas adjacentes, não pela
            // própria -- simplificação razoável de "ficaria passado depois duma troca".
            int relRank = (side == Color::WHITE) ? rank : 7 - rank;
            score += kPawnStructW.candidatePasser * (relRank >= 3 ? 1 : 0);
        }
    }
    return score;
}

// Bishop pair / long diagonal / rammed pawns -- ver comentário na struct BishopWeights.
static Score computeBishopScore(const Board& board, Color side) {
    Score score;
    Bitboard ownBishops = board.pieces(side, PieceType::BISHOP);
    if (ownBishops.popcount() >= 2) score += kBishopW.pair;
    Bitboard ownPawns = board.pieces(side, PieceType::PAWN);
    Bitboard enemyPawns = board.pieces(~side, PieceType::PAWN);
    static constexpr uint64_t kLightSquares = 0x55AA55AA55AA55AAULL;
    Bitboard bb = ownBishops;
    while (bb.any()) {
        Square sq = bb.poplsb();
        bool light = (kLightSquares >> sq.value()) & 1;
        // Long diagonal central: bispo numa das 2 diagonais principais (a1-h8/a8-h1) E
        // sem peças próprias a bloquear nas 2 casas centrais dessa diagonal.
        bool onA1H8 = (sq.file() == sq.rank());
        bool onA8H1 = (sq.file() + sq.rank() == 7);
        if (onA1H8 || onA8H1) {
            // Casas centrais da diagonal a1-h8: d4(27)/e5(36). Da a8-h1: d5(35)/e4(28).
            Bitboard centerSquares = onA1H8 ? Bitboard::fromSquare(Square(27)) | Bitboard::fromSquare(Square(36))
                                             : Bitboard::fromSquare(Square(35)) | Bitboard::fromSquare(Square(28));
            if (!(board.allOcc & centerSquares & ~Bitboard::fromSquare(sq)).any()) score += kBishopW.longDiagonal;
        }
        // Rammed pawns: peões PRÓPRIOS imediatamente bloqueados por um peão INIMIGO
        // mesmo em frente, na mesma cor de casa do bispo.
        Bitboard sameColor = Bitboard(light ? kLightSquares : ~kLightSquares);
        Bitboard rammed = ownPawns & sameColor;
        Bitboard rb = rammed;
        while (rb.any()) {
            Square psq = rb.poplsb();
            int aheadRank = (side == Color::WHITE) ? psq.rank() + 1 : psq.rank() - 1;
            if (aheadRank < 0 || aheadRank > 7) continue;
            if (enemyPawns.test(Square((aheadRank << 3) | psq.file()).value())) score += kBishopW.rammedPawn;
        }
    }
    return score;
}

// Outpost: cavalo/bispo numa casa defendida por peão próprio, em rank 4-6 (relativa),
// que NENHUM peão inimigo pode atacar (nas colunas adjacentes, à frente, em nenhuma rank).
static Score computeOutpostScore(const Board& board, Color side, const AttackInfo& us) {
    Bitboard enemyPawns = board.pieces(~side, PieceType::PAWN);
    Score score;
    auto isOutpost = [&](Square sq) -> bool {
        int relRank = (side == Color::WHITE) ? sq.rank() : 7 - sq.rank();
        if (relRank < 3 || relRank > 5) return false;
        if (!us.byPawn.test(sq.value())) return false;  // defendido por peão próprio
        int file = sq.file(), rank = sq.rank();
        for (int df = -1; df <= 1; df += 2) {
            int f = file + df;
            if (f < 0 || f > 7) continue;
            int rStart = (side == Color::WHITE) ? rank + 1 : 0;
            int rEnd   = (side == Color::WHITE) ? 8 : rank;
            for (int r = rStart; r < rEnd; ++r)
                if (enemyPawns.test(Square((r << 3) | f).value())) return false;
        }
        return true;
    };
    Bitboard bb = board.pieces(side, PieceType::KNIGHT);
    while (bb.any()) { Square sq = bb.poplsb(); if (isOutpost(sq)) score += kOutpostW.knight; }
    bb = board.pieces(side, PieceType::BISHOP);
    while (bb.any()) { Square sq = bb.poplsb(); if (isOutpost(sq)) score += kOutpostW.bishop; }
    return score;
}

// Rook: coluna aberta/semi-aberta, 7ª fila (só se o rei inimigo ainda não saiu das 2
// últimas filas -- bónus de pressão num rei ainda não "fugido").
static Score computeRookScore(const Board& board, Color side) {
    Bitboard ownPawns = board.pieces(side, PieceType::PAWN);
    Bitboard enemyPawns = board.pieces(~side, PieceType::PAWN);
    Square enemyKing = board.kingSq(~side);
    int enemyKingRelRank = (side == Color::WHITE) ? enemyKing.rank() : 7 - enemyKing.rank();
    Score score;
    Bitboard bb = board.pieces(side, PieceType::ROOK);
    while (bb.any()) {
        Square sq = bb.poplsb();
        int file = sq.file(), rank = sq.rank();
        bool ownOnFile = false, enemyOnFile = false;
        for (int r = 0; r < 8; ++r) {
            if (ownPawns.test(Square((r << 3) | file).value())) ownOnFile = true;
            if (enemyPawns.test(Square((r << 3) | file).value())) enemyOnFile = true;
        }
        if (!ownOnFile && !enemyOnFile) score += kRookW.openFile;
        else if (!ownOnFile && enemyOnFile) score += kRookW.semiOpenFile;
        int relRank = (side == Color::WHITE) ? rank : 7 - rank;
        if (relRank == 6 && enemyKingRelRank >= 6) score += kRookW.seventhRank;
    }
    return score;
}
// Pesos por tipo de peça atacante (não treináveis -- são só a PONDERAÇÃO usada para somar
// "unidades de ataque" antes de indexar a tabela; o ganho/perigo REAL fica todo nos pesos
// treináveis kKingSafetyW.attackUnits[]). Convenção clássica SF: dama pesa mais que torre,
// que pesa mais que menor.
// 🦅 FIX: pawn era 0 aqui mas 1 no Rust tuner (texel_tuner_main.rs KING_ATTACK_WEIGHT) --
// divergência silenciosa confirmada via revisão estrutural do Opus: o tuner indexava
// attackUnits[] num índice diferente do que o motor usa de facto. Alinhado a 1 em ambos.
static constexpr int kKingAttackWeight[6] = { 1, 2, 2, 3, 5, 0 };  // pawn,knight,bishop,rook,queen,king
static Score computeKingSafetyScore(const Board& board, Color side, const AttackInfo& us, const AttackInfo& them) {
    Square ksq = board.kingSq(side);
    Bitboard ring = attacks::kingAttacks(ksq) | Bitboard::fromSquare(ksq);
    int units = 0;
    units += kKingAttackWeight[int(PieceType::KNIGHT)] * (them.byKnight & ring).popcount();
    units += kKingAttackWeight[int(PieceType::BISHOP)] * (them.byBishop & ring).popcount();
    units += kKingAttackWeight[int(PieceType::ROOK)]   * (them.byRook & ring).popcount();
    units += kKingAttackWeight[int(PieceType::QUEEN)]  * (them.byQueen & ring).popcount();
    units += kKingAttackWeight[int(PieceType::PAWN)]   * (them.byPawn & ring).popcount();
    Score score = kKingSafetyW.attackUnits[std::min(units, 49)];

    // Pawn shield: as 3 casas imediatamente em frente ao rei (rank+1 para brancas,
    // rank-1 para pretas), nos ficheiros [file-1, file, file+1].
    int kf = ksq.file(), kr = ksq.rank();
    int shieldRank = (side == Color::WHITE) ? kr + 1 : kr - 1;
    int missing = 0;
    if (shieldRank >= 0 && shieldRank < 8) {
        Bitboard ownPawns = board.pieces(side, PieceType::PAWN);
        for (int df = -1; df <= 1; ++df) {
            int f = kf + df;
            if (f < 0 || f > 7) continue;
            Square sq = Square((shieldRank << 3) | f);
            if (!ownPawns.test(sq.value())) ++missing;
        }
    }
    score += kKingSafetyW.pawnShieldMissing[std::min(missing, 3)];

    // Safe check: casas de onde uma peça inimiga do tipo X DARIA XEQUE ao nosso rei
    // (ataque simétrico a partir da posição do rei), que essa peça realmente alcança, E
    // que são "safe" para o inimigo (não defendidas por nós, ou defendidas só por uma
    // peça mas atacadas 2x pelo inimigo -- mesma ideia do "weak" em computeThreatScore).
    Bitboard occ = board.allOcc;
    Bitboard safeForThem = ~us.all | them.all2;
    Bitboard queenChecks  = (attacks::bishopAttacks(ksq, occ) | attacks::rookAttacks(ksq, occ)) & them.byQueen & safeForThem;
    Bitboard rookChecks   = attacks::rookAttacks(ksq, occ) & them.byRook & safeForThem;
    Bitboard bishopChecks = attacks::bishopAttacks(ksq, occ) & them.byBishop & safeForThem;
    Bitboard knightChecks = attacks::knightAttacks(ksq) & them.byKnight & safeForThem;
    score += kKingSafetyW.safeCheck[0] * queenChecks.popcount();
    score += kKingSafetyW.safeCheck[1] * rookChecks.popcount();
    score += kKingSafetyW.safeCheck[2] * bishopChecks.popcount();
    score += kKingSafetyW.safeCheck[3] * knightChecks.popcount();
    return score;
}
static Score computeMobilityScore(const Board& board, Color side, const AttackInfo& them) {
    Bitboard ownPieces = Bitboard(0ULL);
    for (int pt = 0; pt < 6; ++pt) ownPieces |= board.pieceBB[int(side)][pt];
    Bitboard mobilityArea = ~ownPieces & ~them.byPawn;
    Bitboard occ = board.allOcc;
    Score score;
    Bitboard bb = board.pieces(side, PieceType::KNIGHT);
    while (bb.any()) {
        int cnt = (attacks::knightAttacks(bb.poplsb()) & mobilityArea).popcount();
        score += kMobilityW.knight[std::min(cnt, 8)];
    }
    bb = board.pieces(side, PieceType::BISHOP);
    while (bb.any()) {
        int cnt = (attacks::bishopAttacks(bb.poplsb(), occ) & mobilityArea).popcount();
        score += kMobilityW.bishop[std::min(cnt, 13)];
    }
    bb = board.pieces(side, PieceType::ROOK);
    while (bb.any()) {
        int cnt = (attacks::rookAttacks(bb.poplsb(), occ) & mobilityArea).popcount();
        score += kMobilityW.rook[std::min(cnt, 14)];
    }
    bb = board.pieces(side, PieceType::QUEEN);
    while (bb.any()) {
        Square qs = bb.poplsb();
        int cnt = ((attacks::bishopAttacks(qs, occ) | attacks::rookAttacks(qs, occ)) & mobilityArea).popcount();
        score += kMobilityW.queen[std::min(cnt, 27)];
    }
    return score;
}

// WeakQueen: penaliza a NOSSA PRÓPRIA dama se está numa linha (reta ou diagonal) onde,
// removendo a peça interposta mais próxima, um slider inimigo do tipo compatível
// (bispo/dama numa diagonal, torre/dama numa reta) a atacaria -- vulnerabilidade a um
// pin/ataque descoberto potencial, não um ataque já realizado.
static Score computeWeakQueenScore(const Board& board, Color side) {
    Bitboard ownQueens = board.pieces(side, PieceType::QUEEN);
    if (!ownQueens.any()) return Score{};
    Color enemy = ~side;
    Bitboard enemyDiagSliders = board.pieces(enemy, PieceType::BISHOP) | board.pieces(enemy, PieceType::QUEEN);
    Bitboard enemyOrthoSliders = board.pieces(enemy, PieceType::ROOK) | board.pieces(enemy, PieceType::QUEEN);
    static constexpr int kDirs[8][2] = { {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1} };
    Score score;
    Bitboard qbb = ownQueens;
    while (qbb.any()) {
        Square qsq = qbb.poplsb();
        int qf = qsq.file(), qr = qsq.rank();
        for (int d = 0; d < 8; ++d) {
            bool diagonal = kDirs[d][0] != 0 && kDirs[d][1] != 0;
            Bitboard relevantSliders = diagonal ? enemyDiagSliders : enemyOrthoSliders;
            if (!relevantSliders.any()) continue;
            int f = qf, r = qr;
            Square blocker = SQ_NONE;
            for (int step = 0; step < 7; ++step) {
                f += kDirs[d][0]; r += kDirs[d][1];
                if (f < 0 || f > 7 || r < 0 || r > 7) break;
                Square sq((r << 3) | f);
                if (!board.allOcc.test(sq.value())) continue;
                if (!blocker.isValid()) {
                    blocker = sq;
                    continue;
                }
                if (relevantSliders.test(sq.value())) score += kThreatW.weakQueen;
                break;
            }
        }
    }
    return score;
}

// Conta os 7 termos para `side` atacando o adversário; devolve a soma já pesada (mg,eg).
static Score computeThreatScore(const Board& board, Color side, const AttackInfo& us, const AttackInfo& them) {
    Color enemy = ~side;
    Bitboard enemyAll = Bitboard(0ULL);
    for (int pt = 0; pt < 6; ++pt) enemyAll |= board.pieceBB[int(enemy)][pt];
    Bitboard enemyPawns = board.pieceBB[int(enemy)][int(PieceType::PAWN)];
    Bitboard nonPawnEnemiesReal = enemyAll & ~enemyPawns;

    Bitboard stronglyProtected = them.byPawn | (them.all2 & ~us.all2);
    Bitboard defended = nonPawnEnemiesReal & stronglyProtected;
    Bitboard weak = enemyAll & ~stronglyProtected & us.all;

    Score score;
    // ThreatByMinor: minor ataca (defended|weak), soma por tipo de peça atacada.
    Bitboard minorTargets = (defended | weak) & (us.byKnight | us.byBishop);
    {
        Bitboard bb = minorTargets;
        while (bb.any()) {
            Square sq = bb.poplsb();
            PieceType pt = board.pieceOn(sq);
            if (pt != PieceType::NONE) score += kThreatW.threatByMinor[int(pt)];
        }
    }
    // ThreatByRook: rook ataca só `weak` (não defended).
    {
        Bitboard bb = weak & us.byRook;
        while (bb.any()) {
            Square sq = bb.poplsb();
            PieceType pt = board.pieceOn(sq);
            if (pt != PieceType::NONE) score += kThreatW.threatByRook[int(pt)];
        }
    }
    // ThreatByKing: booleano, rei ataca pelo menos 1 peça weak.
    if ((weak & us.byKing).any()) score += kThreatW.threatByKing;
    // Hanging: weak adicionalmente totalmente indefeso OU não-peão atacado 2x por nós.
    {
        Bitboard hangBase = ~them.all | (nonPawnEnemiesReal & us.all2);
        score += kThreatW.hanging * (weak & hangBase).popcount();
    }
    // WeakQueenProtection: entre as weak, quantas só a dama inimiga defende.
    score += kThreatW.weakQueenProt * (weak & them.byQueen).popcount();
    // RestrictedPiece: casas que o inimigo ocupa/defende, não fortemente protegidas, que
    // nós também atacamos -- restringe a mobilidade das peças inimigas nessas casas.
    {
        Bitboard restricted = them.all & ~stronglyProtected & us.all;
        score += kThreatW.restrictedPiece * restricted.popcount();
    }
    Bitboard safe = ~them.all | us.all;
    // ThreatBySafePawn: peões nossos em casas safe atacando peças inimigas não-peão.
    {
        Bitboard safePawns = board.pieces(side, PieceType::PAWN) & safe;
        Bitboard pawnAtk = (side == Color::WHITE) ? attacks::pawnAttacks<Color::WHITE>(safePawns)
                                                   : attacks::pawnAttacks<Color::BLACK>(safePawns);
        score += kThreatW.threatBySafePawn * (pawnAtk & nonPawnEnemiesReal).popcount();
    }
    // ThreatByPawnPush: peões nossos que poderiam empurrar (1 ou 2 casas, se ainda na
    // rank inicial) para uma casa safe e não atacada por peão inimigo, e dali atacariam
    // uma peça inimiga não-peão -- ameaça "latente", ainda não realizada.
    {
        Bitboard occ = board.allOcc;
        Bitboard ownPawns = board.pieces(side, PieceType::PAWN);
        Bitboard push1, push2;
        if (side == Color::WHITE) {
            push1 = Bitboard(ownPawns.value() << 8) & ~occ;
            Bitboard startRank2 = Bitboard(ownPawns.value() & 0x000000000000FF00ULL);
            push2 = Bitboard((Bitboard(startRank2.value() << 8) & ~occ).value() << 8) & ~occ;
        } else {
            push1 = Bitboard(ownPawns.value() >> 8) & ~occ;
            Bitboard startRank7 = Bitboard(ownPawns.value() & 0x00FF000000000000ULL);
            push2 = Bitboard((Bitboard(startRank7.value() >> 8) & ~occ).value() >> 8) & ~occ;
        }
        Bitboard pushTargets = (push1 | push2) & safe & ~them.byPawn;
        Bitboard pushAtk = (side == Color::WHITE) ? attacks::pawnAttacks<Color::WHITE>(pushTargets)
                                                   : attacks::pawnAttacks<Color::BLACK>(pushTargets);
        score += kThreatW.threatByPawnPush * (pushAtk & nonPawnEnemiesReal).popcount();
    }
    // KnightOnQueen / SliderOnQueen: só fazem sentido com exatamente 1 dama inimiga.
    Bitboard enemyQueens = board.pieces(enemy, PieceType::QUEEN);
    if (enemyQueens.popcount() == 1) {
        Square eqSq = enemyQueens.lsb();
        Bitboard ownPawns = board.pieces(side, PieceType::PAWN);
        Bitboard localSafe = ~ownPawns & ~stronglyProtected;
        bool queenImbalance = board.pieces(side, PieceType::QUEEN).popcount() == 1;
        int mult = queenImbalance ? 2 : 1;
        // KnightOnQueen: temos um cavalo que ataca a casa da dama (via padrão de salto
        // de cavalo a partir dessa casa, simétrico) E essa casa está localSafe.
        {
            Bitboard knightFromQueen = attacks::knightAttacks(eqSq);
            Bitboard cnt = us.byKnight & knightFromQueen & localSafe;
            score += kThreatW.knightOnQueen * cnt.popcount() * mult;
        }
        // SliderOnQueen: bispo/torre nossos atacam a casa da dama através de uma casa
        // que é localSafe E está atacada 2x por nós (caminho duplamente apoiado).
        {
            Bitboard occ = board.allOcc;
            Bitboard sliderFromQueen = attacks::bishopAttacks(eqSq, occ) | attacks::rookAttacks(eqSq, occ);
            Bitboard cnt = (us.byBishop | us.byRook) & sliderFromQueen & localSafe & us.all2;
            score += kThreatW.sliderOnQueen * cnt.popcount() * mult;
        }
    }
    return score;
}

// ─── Eval ─────────────────────────────────────────────────────────────────
static int staticEval(const Board& board) {
    int score;
    if (napoleon::nnue::isLoaded()) {
        score = napoleon::nnue::evaluate(board);
    } else {
        // HCE (sem rede NNUE): PSQT calibrado (já inclui o valor de
        // material implícito, ver comentário acima das tabelas -- NÃO soma material à
        // parte, duplicaria) + termos de threats + mobility, todos calibrados juntos via
        // Texel tuning a partir de posições reais, cada um agora como par {mg,eg}
        // interpolado pela fase do jogo (ver comentário em Score/gamePhase acima).
        // Convenção do motor: sq=0 é a1 (rank1), a tabela está escrita com linha 0 =
        // rank8 — por isso brancas leem sq^56 (inverte o rank), pretas leem sq
        // diretamente (simetria especular completa).
        Score s;
        for (int pt = 0; pt < 6; ++pt) {
            Bitboard wp = board.pieceBB[0][pt];
            while (wp.any()) s += kPsqt[pt][wp.poplsb().value() ^ 56];
            Bitboard bp = board.pieceBB[1][pt];
            while (bp.any()) s -= kPsqt[pt][bp.poplsb().value()];
        }
        AttackInfo whiteAtk = computeAttackInfo(board, Color::WHITE);
        AttackInfo blackAtk = computeAttackInfo(board, Color::BLACK);
        s += computeThreatScore(board, Color::WHITE, whiteAtk, blackAtk);
        s -= computeThreatScore(board, Color::BLACK, blackAtk, whiteAtk);
        s += computeMobilityScore(board, Color::WHITE, blackAtk);
        s -= computeMobilityScore(board, Color::BLACK, whiteAtk);
        s += computeKingSafetyScore(board, Color::WHITE, whiteAtk, blackAtk);
        s -= computeKingSafetyScore(board, Color::BLACK, blackAtk, whiteAtk);
        s += computePawnStructureScore(board, Color::WHITE);
        s -= computePawnStructureScore(board, Color::BLACK);
        s += computeWeakQueenScore(board, Color::WHITE);
        s -= computeWeakQueenScore(board, Color::BLACK);
        s += computeBishopScore(board, Color::WHITE);
        s -= computeBishopScore(board, Color::BLACK);
        s += computeOutpostScore(board, Color::WHITE, whiteAtk);
        s -= computeOutpostScore(board, Color::BLACK, blackAtk);
        s += computeRookScore(board, Color::WHITE);
        s -= computeRookScore(board, Color::BLACK);
        // Interpolação MG/EG pela fase do jogo (material restante) -- ver gamePhase().
        int phase = gamePhase(board);
        int sTapered = (s.mg * phase + s.eg * (MAX_PHASE - phase)) / MAX_PHASE;
        // 🦅 Reescala global do HCE: o PSQT calibrado tem médias bem menores que
        // kPieceValue (Dama≈574 vs 975, Cavalo≈182 vs 325 -- fator ~1.5x médio entre
        // peças). kPieceValue é usado em VÁRIOS sítios da busca somado DIRETAMENTE ao
        // eval (delta pruning, capture futility) e TODAS as margens de poda fixas (RFP,
        // NMP, futility de quietos, SE_MARGIN, razoring) foram pensadas/calibradas
        // implicitamente para a escala "padrão" que kPieceValue representa. Sem
        // reescalar, estas podas ficam muito menos agressivas com o HCE (o eval "parece"
        // sempre mais próximo de alfa/beta do que devia), confirmado como causa real de
        // explosões de nós (depth16→17 do startpos: 2.2M→125M nós, 152s). Multiplicar
        // aqui em vez de criar tabelas alternativas em cada sítio: corrige TODAS as
        // margens de uma vez, mantendo o resto do código (SEE, MVV-LVA, kPieceValue)
        // intocado -- só staticEval() muda de escala.
        static constexpr double HCE_RESCALE = 1.5;
        sTapered = (int)(sTapered * HCE_RESCALE);
        score = board.sideToMove() == Color::WHITE ? sTapered : -sTapered;
        // Tempo: bónus por ser a vez de jogar -- interpolado pela MESMA fase, mas
        // adicionado DEPOIS da conversão de perspetiva (sempre a favor de quem joga
        // agora, não sempre Brancas). Conceito confirmado no Ethereal real
        // (src/evaluate.c), que o usa flat/não-tapered -- mantemos par mg/eg por
        // consistência com o resto do sistema de calibração, sem custo extra.
        int tempoTapered = (kTempoBonus.mg * phase + kTempoBonus.eg * (MAX_PHASE - phase)) / MAX_PHASE;
        score += (int)(tempoTapered * HCE_RESCALE);
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
    bool multiCutSignal = false;  // ver bloco do singular extension: para o LOOP de
                                   // lances depois do lance da TT, não faz return cedo
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
                // a testar MAIS lances neste nó.
                // 🦅 FIX #2: a 1ª correção (return beta+(sScore-beta)*6/10) ainda fazia um
                // return ANTECIPADO com um valor vindo duma busca a profundidade REDUZIDA
                // (singularDepth) -- comparado com o Ethereal real (src/search.c,
                // singularity()), que NUNCA faz return aqui: só sinaliza ao move picker p/
                // parar de testar MAIS lances, e deixa o lance da TT prosseguir pela busca
                // NORMAL (mesma profundidade completa) como faria de qualquer forma. Um
                // return com valor aproximado a partir duma busca reduzida injeta um score
                // instável no pai (janelas de aspiração, outras podas) -- confirmado como
                // causa real duma explosão de nós (depth 16→17: 2.2M→125M nós, 152s) no HCE
                // aqui: o HCE tem mais variância entre profundidades que a NNUE,
                // tornando este corte aproximado MUITO menos fiável. Sem return: o lance da
                // TT é processado abaixo como qualquer outro, só não testamos os restantes.
                multiCutSignal = true;
            } else if (ttScore >= beta || ttScore <= alpha) {
                // Extensão negativa: gate original era só `ttScore >= beta` (Reckless,
                // src/search.rs). Confirmado no Ethereal real (src/search.c, singularity())
                // que TAMBÉM desconfia do lance da TT quando `ttValue <= alpha` (já estava a
                // falhar baixo) -- adicionado esse segundo caso. O lance da TT "parecia" bom
                // o suficiente para cortar OU já estava a falhar, mas a verificação mostrou
                // que NÃO é singular -- desconfia, reduz em vez de assumir que é o melhor.
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
        // Multi-cut (ver bloco do singular extension acima): o lance da TT já foi
        // processado normalmente nesta iteração: agora paramos de testar mais lances,
        // sem ter feito nenhum return antecipado com valor aproximado.
        if (multiCutSignal) break;
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
        // Deadline de explosão: 15x o tempo da última iteração (bem mais generoso que o
        // ITER_GROWTH_GUESS=2.0 usado acima para decidir SE começa -- aqui é só uma rede
        // de segurança contra explosões reais, não o crescimento normal de ID), com piso
        // de 2s para não disparar por ruído de medição em profundidades baixas/triviais.
        static constexpr int64_t ITER_ABORT_FACTOR = 15;
        info.iterDeadlineMs = (lastIterMs > 0)
            ? iterStartMs + std::max((int64_t)2000, lastIterMs * ITER_ABORT_FACTOR)
            : 0;
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

    // Output best move. Salvaguarda absoluta: bestMove pode ficar nulo se a 1ª iteração
    // for interrompida antes de processar qualquer lance e a TT também não tiver nada
    // (motor recém-arrancado, sem histórico) -- "bestmove 0000" é inválido em protocolo
    // UCI e seria uma derrota/erro imediato em jogo real. NUNCA envia 0000 se houver
    // pelo menos um lance legal: usa o primeiro disponível.
    if (bestMove.isNull()) {
        MoveList fallbackList;
        generateMoves(board, fallbackList);
        for (int i = 0; i < fallbackList.count; ++i) {
            if (board.isLegal(fallbackList.moves[i])) { bestMove = fallbackList.moves[i]; break; }
        }
    }
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
