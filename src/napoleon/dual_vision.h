#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 NAPOLEON — DUAL VISION (Two-Eye Evaluation)
// ═══════════════════════════════════════════════════════════════════════════
//
// Couche Napoleon ajoutée par-dessus l'évaluation HCE de Sirius.
//
// PRINCIPE :
//   Œil 1 (HCE)  = évaluation handcrafted de Sirius (rapide, fiable, ~3537 ELO)
//   Œil 2 (NNUE) = réseau de neurones custom Napoleon (entraîné sur datagen)
//
//   Les deux yeux regardent la même position. Quand ils sont d'accord, on
//   fait confiance. Quand ils divergent fortement, c'est un signal : la
//   position est subtile (tactique cachée ou jugement positionnel fin).
//
// MODES (réglables via UCI option "DualVision") :
//   0 = OFF        → HCE seul (= Sirius pur)
//   1 = BLEND      → moyenne pondérée des deux yeux
//   2 = VERIFY     → HCE pilote, NNUE corrige seulement sur forte divergence
//
// Le NNUE ne PILOTE jamais aveuglément. Il affine le jugement du HCE.
// ═══════════════════════════════════════════════════════════════════════════

#include "../defs.h"

class Board;

namespace napoleon::dualvision
{

enum class Mode : int
{
    OFF     = 0,  // HCE seul (Sirius pur)
    BLEND   = 1,  // moyenne pondérée
    VERIFY  = 2,  // HCE pilote, NNUE corrige sur divergence
    PRIMARY = 3,  // NapK9/NNUE est l'éval PRINCIPALE (les cp viennent de lui),
                  // mais toute la search Sirius reste autour (pruning, ordering,
                  // valorisation pièces). Façon Stockfish : réseau + search classique.
};

// Configuration globale du Dual Vision (réglable via UCI)
struct Config
{
    Mode mode = Mode::PRIMARY;   // 🦅 FIX: era OFF (bug: combine saía p/ HCE puro e IGNORAVA a NNUE
                                 //   na search — a rede estava "cega"). Agora PRIMARY = NNUE ativa.
    int  blendWeightNNUE = 50;   // poids NNUE en mode BLEND (0..100, %)
    int  divergenceThreshold = 150;  // seuil de divergence (cp) en mode VERIFY
    int  nnueScale = 100;        // amplification du score NNUE (%, 100 = ×1.0).
                                 // Le réseau bullet WDL donne des cp compressés ;
                                 // augmenter (ex: 400) les décompresse vers l'échelle
                                 // Sirius (~100cp/pion) pour que le pruning matche.
    int  primaryHceBase = 15;    // poids HCE de base (%) en mode PRIMARY. Le mélange
                                 // monte automatiquement avec la divergence matérielle.
    int  panicMs = 15000;        // 🦅 sous ce temps restant (ms), on DÉSACTIVE le NapK9
                                 // et on repasse en HCE pur (Sirius) : plus rapide,
                                 // pas de forward pass NNUE → zéro risque de flag.
    bool panicActive = false;    // true quand le temps restant < panicMs (mis à jour
                                 // par le time manager au début de chaque recherche).
    int  minDepth = 4;           // 🦅 NapK9 actif SEULEMENT si depth restante >= minDepth.
                                 // Aux feuilles (depth < minDepth, ~99% des nœuds) → HCE
                                 // rapide. Près de la racine → NapK9 intelligent.
                                 // Garde la vitesse Sirius + l'intelligence NapK9.
                                 // 0 = NapK9 partout (lent), élevé = NapK9 rare (rapide).
    bool nnueLoaded = false;     // true si un réseau .nnue est chargé

    // 🦅 LAZY NNUE (ideia do Maréchal): a small (128) explora todos os nós; a big (256)
    //   só entra quando a small diverge do HCE > lazyMargin (posição que precisa de
    //   precisão). Posições claras ficam na small → + rápido → + profundidade.
    bool lazyNNUE = false;       // OFF por defeito (compat com o modo clássico)
    int  lazyMargin = 90;        // cp de divergência small↔HCE para acordar a big

    // 🦅 PureNNUE (ideia do Maréchal): sem HCE no meio-jogo; cascata de cabeças
    //   16→32→256 por profundidade. HCE só nos finais (material insuficiente, KXvK).
    bool pureNNUE = true;        // 🦅 ON por defeito (objetivo do Maréchal: NNUE pura no meio-jogo)
    // 🔴 LIÇÃO MEDIDA (matches no container, 48 jogos, 1024 treinada): a cascata por PROFUNDIDADE
    //   punha a head bullet/small em 97% dos nós → eval fraca onde decide → o PureNNUE perdia.
    //   big-everywhere vs cascata: +66 elo (LOS ~95%) APESAR de 2.5× menos NPS (29k vs 73k).
    //   Defaults agora 0/0 (big em todo o lado). A head barata só entra em posições já DECIDIDAS
    //   por material (regra estilo SF: simple-eval decide a rede) — ver pureDecidedMaterial.
    int  pureBulletDepth = 0;    // depthLeft < isto → cabeça bullet(16). 0 = nunca (default novo)
    int  pureMidDepth = 0;       // depthLeft < isto → cabeça small(32). 0 = nunca (default novo)
    // ⚠️ 900 e não 500: com 500, linhas táticas com trocas PENDENTES (QxQ antes da recaptura)
    //   disparavam a regra → a head bullet avaliava as posições mais críticas (medido: 38% dos
    //   nós a depth 20). Com 900 só dama LÍQUIDA a mais dispara — posições verdadeiramente decididas.
    int  pureDecidedMaterial = 900;  // |diff material| (cp) ≥ isto → posição decidida → head bullet
    int  pureEndgamePieces = 7;  // 🦅 em PureNNUE, com <= isto peças (final) e perto das folhas
                                 //   (depthLeft < minDepth) → usa a HCE de finais CALIBRADA em vez
                                 //   da NNUE. 0 = desligado (NNUE sempre). Bullet: a HCE de finais
                                 //   é mais fiável/rápida que a NNUE WDL comprimida.

    // 🦅 s29 (à Stockfish): usar a divergência |big - small| como sinal de COMPLEXITY p/ modular
    //   as podas. Posição clara (redes concordam) → podar +; incerta (divergem) → podar -. Custo:
    //   2 forward passes (big+small) nos nós onde se calcula. OFF por defeito (mede-se se compensa).
    bool complexityPruning = false;
    int  complexityMinDepth = 5; // só calcular complexity (2 forwards) em nós com depthLeft >= isto
                                 //   (perto da raiz, onde as podas pesam). Nas folhas seria caro demais.
};

extern Config g_config;

// Charge un réseau .nnue depuis un fichier. Retourne true si succès.
bool loadNetwork(const std::string& path);
bool loadNetworkEmbedded();   // 🦅 rede embutida no binário

// Décharge le réseau (revient en HCE pur).
void unloadNetwork();

// Évaluation NNUE seule (œil 2). POV side-to-move, en centipions.
// Suppose qu'un réseau est chargé (g_config.nnueLoaded == true).
int evaluateNNUE(const Board& board);

// Combine l'œil HCE (déjà calculé) et l'œil NNUE selon le mode courant.
//   hceScore : le score HCE de Sirius (eval::evaluate), POV side-to-move
//   board    : la position
// Retourne le score final POV side-to-move.
// Si mode == OFF ou pas de réseau chargé, retourne hceScore inchangé.
int combine(int hceScore, const Board& board, int depthLeft = -1);

// Helpers UCI
// 🦅 s29: diz se a search corre com a NNUE ativa (DualVision/PureNNUE) ou em HCE puro (Sirius).
//   Usado pela aspiração p/ escolher o caminho: NNUE oscila (eval salta na fronteira HCE↔NNUE)
//   → precisa de média deslizante à Reckless; HCE puro usa o caminho Sirius original (calibrado).
bool isNNUEActive();

// 🦅 s29: complexity NNUE (|big-small|) do último nó avaliado, e setters do modo.
int  lastComplexity();
void resetComplexity();
void setComplexityPruning(bool on);
void setComplexityMinDepth(int d);

void setMode(int mode);
void setBlendWeight(int weight);
void setDivergenceThreshold(int cp);
void setScale(int scale);
void setPrimaryHceBase(int pct);
void setPanicMs(int ms);          // 🦅 seuil de temps (ms) sous lequel NapK9 est OFF
void setPanicActive(bool on);     // 🦅 active/désactive le panic (appelé par time man)
void setMinDepth(int d);          // 🦅 depth restante mini pour activer le NapK9
void setLazyNNUE(bool on);        // 🦅 ativa o lazy small→big
void setLazyMargin(int cp);       // 🦅 margem de divergência small↔HCE para a big
void setPureNNUE(bool on);        // 🦅 modo sem HCE (cascata de cabeças 16/32/256)
void setPureDecidedMaterial(int cp); // 🦅 limiar material p/ a head bullet (estilo SF)
// 🦅 OTIMIZAÇÃO (profile: HCE caro calculado p/ ser DESCARTADO em PureNNUE):
//   true ⇔ o combine devolve NNUE pura sem olhar ao hceScore → o eval salta o HCE.
bool skipHCE(const Board& board, int depthLeft);
void setPureEndgamePieces(int n); // 🦅 PureNNUE: <= n peças + perto das folhas → HCE de finais
void setPureDepths(int bulletDepth, int midDepth);  // 🦅 limiares da cascata
int  evaluateNNUEHead(const Board& board, int headIdx);  // 🦅 avalia com cabeça dada
const char* modeName();

}  // namespace napoleon::dualvision
