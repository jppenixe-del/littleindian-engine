#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 NAPOLEON — NNUE NET (décodeur NapK9 V9 + forward pass complet, SIMD-friendly)
// ═══════════════════════════════════════════════════════════════════════════
//
// Décode et évalue un réseau NapK9 (format maison, magic "NAPK9LEB").
// Architecture style Stockfish : HalfKAv2_hm accumulator + heads par bucket + PSQT.
//
//   • Feature transformer : 22528 features × L1, accumulator int16 (us + them)
//   • 8 material buckets : bucket = (piece_count - 1) / 4
//   • Heads (bullet/small/big), chacune L1*2 → l1 → l2 → 1, poids int8
//   • PSQT : 22528 × 8, additif au score
//   • OUTPUT_SCALE_CP = 408
//
// Le forward pass est écrit en boucles contiguës pour auto-vectorisation -O3
// (NEON sur ARM, AVX sur x86). Pas d'intrinsics → portable Pixel + serveur.
// ═══════════════════════════════════════════════════════════════════════════

#include "../defs.h"
#include <string>
#include <vector>
#include <cstdint>

class Board;

namespace napoleon::nnue
{

bool load(const std::string& path);   // détecte NapK9 (magic) ou .nnue Stockfish
bool loadEmbedded();                   // 🦅 charge la rede embutida dans le binaire
bool hasEmbeddedNet();                 // true si une rede est embutida
bool hasEmbeddedSmallNet();            // 🦅 true se a small (128) está embebida
void unload();
bool isLoaded();
enum HeadSel { HEAD_BULLET = 0, HEAD_SMALL = 1, HEAD_BIG = 2 };  // 🦅 cabeças 16/32/256
int  evaluate(const Board& board);                 // usa a cabeça big (default)
int  evaluate(const Board& board, int headIdx);    // 🦅 escolhe a cabeça (bullet/small/big)    // score cp, POV side-to-move
int  verifyFinny(const Board& board); // 🦅 debug : diff finny vs complet (0=ok)
// 🦅 Sistema DUAL (big 256 + small 128, ambas embebidas, alternância inteligente)
bool loadSmall(const std::string& path);  // carrega a small de ficheiro
bool loadSmallEmbedded();                  // carrega a small embebida
bool dualLoaded();                         // true se big+small carregadas
void setDualEnabled(bool on);              // ativa alternância (só se dual carregado)
bool dualEnabled();
void setActiveNet(int idx);                // 0=big (precisa), 1=small (rápida)
void setActiveNetNoCount(int idx);          // 🦅 como setActiveNet mas sem contar (PureNNUE)
int  activeNetIdx();                         // 🦅 índice da rede ativa (0=big, 1=small)
int  bigL1();                              // 🦅 L1 da big (0 se não carregada)
int  debugCountThreats(const Board& board); // 🦅 debug: nº features ameaça ativas
void debugPrintThreats(const Board& board);  // 🦅 debug: lista os índices threat
// 🦅 s29: valida o incremental de threats vs completo (pai→filho), bit-a-bit. Para o comando threattest.
bool napkThreatSelfTest(const Board& parent, const Board& child, int L1, bool verbose);
void setThreatsEnabled(bool on);            // 🦅 V9.6 liga/desliga threats em runtime
void resetNetStats();                       // 🦅 contadores de uso big/small (reset)
void getNetStats(uint64_t& big, uint64_t& small);  // 🦅 lê os contadores
void bumpHeadStat(bool small);              // 🦅 PureNNUE: conta head sem trocar a rede
bool threatsEnabled();
int  smallL1();                            // 🦅 L1 da small (0 se não carregada)

// 🦅 V9.5 — Fear/Chaos Head : niveau de volatilité de la position ∈ [0,1].
//   0 = position calme (jouer vite) ; 1 = chaotique/tactique (réfléchir +).
//   Retourne 0.5 (neutre) si le réseau n'a pas de chaos head.
float chaosScore(const Board& board);
bool  hasChaosHead();
// 🦅 s29: verbosidade dos banners de carga (default OFF = silêncio estilo Stockfish).
//   Só imprime no arranque se ligado via setoption NnueVerbose. A info fica sempre
//   disponível via `version` / printNetInfo() a pedido.
void setVerbose(bool on);
bool verbose();
void printNetInfo();   // despeja big+small (L1, threats, chaos, flags) numa só vez, a pedido.

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 ACUMULADOR INCREMENTAL (na pilha do EvalState) — Parte 1: só a API/storage.
//   A ideia (do Maréchal): em vez de reconstruir o acc a cada eval (cópia de L1 + diff de 12
//   bitboards), o acc vive na pilha do EvalState. makeMove→push (add/sub das features que mudam),
//   unmakeMove→pop (volta atrás). O evaluate lê o acc já pronto. NPS sobe ~5-10×.
//   Estas funções são LIGADAS nas Partes 2-3; na Parte 1 só existem (não mudam comportamento).
// ═══════════════════════════════════════════════════════════════════════════
constexpr int NAPK_MAX_L1 = 1536;   // = MAX_L1 (exposto p/ o EvalState dimensionar os buffers)

// Um "slot" de acumulador na pilha (uma perspetiva). int32 (igual ao accW/accB do forward, que
// acumula em int32). NÃO inclui os threats (esses são somados no evaluate, continuam não-incrementais).
struct NapkDelta { int color, type, sq; };

struct NapkAccSlot {
    alignas(32) int32_t accW[NAPK_MAX_L1];   // perspetiva branca (SEM threats)
    alignas(32) int32_t accB[NAPK_MAX_L1];   // perspetiva preta (SEM threats)
    int bucketW = -1, bucketB = -1;          // king-bucket de cada perspetiva
    bool valid = false;                      // slot inicializado (cadeia válida)?
    // 🦅 LAZY UPDATE (estilo SF): o push só guarda os deltas; o acc é materializado
    //   no 1º evaluate que precisar (nós com cutoff antes da eval não pagam nada).
    bool materialized = false;               // accW/accB calculados p/ este ply?
    bool needRefresh = false;                // rei mudou de bucket vs prev → cadeia partida
    const NapkAccSlot* prev = nullptr;       // slot do ply anterior (cadeia)
    NapkDelta adds[2]; int nAdds = 0;        // deltas deste ply (adições)
    NapkDelta rems[2]; int nRems = 0;        // (remoções)

    // 🦅 s29 — THREATS INCREMENTAIS. Acumulador de threats (i16, por perspetiva) + a LISTA
    //   ordenada de threats que o produziu. No filho: diff(listaPai, listaFilho) → só somar/
    //   subtrair as linhas que mudaram (medido: 0-8 de ~68). O caro era somar 68; agora ~4.
    static constexpr int NAPK_MAX_THR = 512;
    alignas(32) int16_t thrAccW[NAPK_MAX_L1];   // acc de threats, perspetiva branca
    alignas(32) int16_t thrAccB[NAPK_MAX_L1];   // perspetiva preta
    int thrListW[NAPK_MAX_THR]; int nThrW = 0;  // lista ORDENADA de threats (branca)
    int thrListB[NAPK_MAX_THR]; int nThrB = 0;  // (preta)
    bool thrValid = false;                      // o thrAcc/thrList deste slot estão prontos?
};

// Parte 2: refresh completo (recalcula o acc do zero p/ a posição atual). Usado na raiz e quando
//   o rei muda de king-bucket. Preenche slot.accW/accB e marca valid.
void napkRefresh(const Board& board, NapkAccSlot& slot);

// Parte 3: push incremental — novo = anterior + deltas (adds/removes das features que mudaram).
//   (declaração agora; corpo na Parte 3, recebe a lista de updates do EvalState.)
//   Parte 1: NÃO implementado ainda (só a estrutura existe).

// Parte 3: o evaluate passa a poder LER um slot pronto (em vez de reconstruir). Ponteiro
//   thread_local para o slot atual; nullptr → fallback p/ o caminho antigo (finny). Datagen/'eval'
//   sem EvalState → nullptr → fallback seguro.
void napkSetCurrentSlot(const NapkAccSlot* slot);   // chamado pelo push/pop do EvalState
void napkSetIncremental(bool on);                   // 🦅 UCI NapkIncremental (A/B fallback)
const NapkAccSlot* napkCurrentSlot();               // o slot atual (nullptr se fora de busca)

// Parte 3: push incremental — calcula `dst` (slot do novo ply) a partir de `src` (slot do ply
//   anterior) aplicando os deltas. addsW/removesW = arrays de (kind, sq) já na perspetiva certa?
//   Não — passamos as peças cruas {color,type,sq} e a função trata as 2 perspetivas + king-bucket
//   refresh (se o rei mudou de bucket, recalcula essa perspetiva do zero via board).
//   nAdds/nRems até 3 (o EvalUpdates limita a 2+2, mas damos folga).
// LAZY: o push regista os deltas e a cadeia (barato); a materialização acontece no evaluate.
void napkLazyPush(const Board& board, const NapkAccSlot& src, NapkAccSlot& dst,
                  const NapkDelta* adds, int nAdds, const NapkDelta* rems, int nRems);

}  // namespace napoleon::nnue

namespace napoleon::nnue {
bool hasChaosHead();
float chaosScore(const Board& board);
}
