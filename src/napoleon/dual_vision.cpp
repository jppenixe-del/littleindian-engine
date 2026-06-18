#include "dual_vision.h"
#include "nnue_net.h"
#include "../board.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace napoleon::dualvision
{

// Configuration globale (instance unique)
Config g_config = {};

// 🦅 s29: última complexity NNUE (|big-small|) calculada, por thread. A search lê-a logo após
//   pedir o eval. -1 = não calculada neste nó (a search trata como 0 / sem sinal).
thread_local int g_lastComplexity = -1;
int  lastComplexity()      { return g_lastComplexity; }
void resetComplexity()     { g_lastComplexity = -1; }

// ───────────────────────────────────────────────────────────────────────────
// Chargement / déchargement du réseau (délègue au module nnue_net)
// ───────────────────────────────────────────────────────────────────────────

bool loadNetwork(const std::string& path)
{
    bool ok = napoleon::nnue::load(path);
    g_config.nnueLoaded = ok;
    return ok;
}

// 🦅 Charge la rede embutida dans le binaire (façon Stockfish, sans fichier).
bool loadNetworkEmbedded()
{
    bool ok = napoleon::nnue::loadEmbedded();
    g_config.nnueLoaded = ok;
    return ok;
}

void unloadNetwork()
{
    napoleon::nnue::unload();
    g_config.nnueLoaded = false;
}

// ───────────────────────────────────────────────────────────────────────────
// Œil 2 : évaluation NNUE (délègue au module nnue_net)
// ───────────────────────────────────────────────────────────────────────────

int evaluateNNUE(const Board& board)
{
    return napoleon::nnue::evaluate(board);
}

// 🦅 Avalia com uma cabeça específica (bullet/small/big) — para o modo PureNNUE.
int evaluateNNUEHead(const Board& board, int headIdx)
{
    return napoleon::nnue::evaluate(board, headIdx);
}

// ───────────────────────────────────────────────────────────────────────────
// Combinaison des deux yeux
// ───────────────────────────────────────────────────────────────────────────

// 🦅 Material insuficiente para dar mate (de QUALQUER lado) → empate teórico.
//   Casos: KvK, KNvK, KBvK, KNNvK (e simétricos), e ainda KminorVKminor (cada lado só
//   com 1 peça menor). Com peões, torres ou damas, há mate possível → NÃO é trivialDraw.
//   Isto evita que a NNUE infle um final morto (+120..+860) e o motor troque para lá.
static bool isInsufficientMatingMaterial(const Board& board)
{
    // Qualquer peão, torre ou dama → há material de mate → não é empate trivial.
    if (board.pieces(PieceType::PAWN).any())  return false;
    if (board.pieces(PieceType::ROOK).any())  return false;
    if (board.pieces(PieceType::QUEEN).any()) return false;

    // Só restam reis, bispos e cavalos. Conta as peças menores por lado.
    auto minors = [&](Color c) -> int {
        return board.pieces(c, PieceType::KNIGHT).popcount()
             + board.pieces(c, PieceType::BISHOP).popcount();
    };
    int w = minors(Color::WHITE);
    int b = minors(Color::BLACK);

    // KvK, K+menor vs K, K vs K+menor → empate. KNN vs K → empate (não forçável).
    // Heurística segura: se NENHUM lado tem >=2 bispos nem (bispo+cavalo), é empate.
    auto canMate = [&](Color c) -> bool {
        int kn = board.pieces(c, PieceType::KNIGHT).popcount();
        int bi = board.pieces(c, PieceType::BISHOP).popcount();
        if (bi >= 2) return true;          // 2 bispos dão mate
        if (bi >= 1 && kn >= 1) return true; // bispo+cavalo dão mate (KBN)
        return false;                       // 0-1 menor, ou só cavalos (KNN não força)
    };
    if (canMate(Color::WHITE) || canMate(Color::BLACK))
        return false;   // alguém pode dar mate → não é empate trivial
    (void)w; (void)b;
    return true;        // ninguém pode dar mate → empate teórico
}

// 🦅 skipHCE: replica EXATAMENTE os ramos do combine que descartam o hceScore.
//   Saltar só quando: rede ON, mode!=OFF, PureNNUE, sem panic, não-insuf-material,
//   e NÃO no ramo "HCE nos finais às folhas".
bool skipHCE(const Board& board, int depthLeft)
{
    if (!g_config.nnueLoaded || g_config.mode == Mode::OFF) return false;
    if (!g_config.pureNNUE) return false;
    if (g_config.panicActive) return false;
    if (isInsufficientMatingMaterial(board)) return false;
    if (depthLeft >= 0 && depthLeft < g_config.minDepth
        && board.allPieces().popcount() <= g_config.pureEndgamePieces)
        return false;
    return true;
}

int combine(int hceScore, const Board& board, int depthLeft)
{
    // Pas de réseau chargé OU mode OFF → HCE pur (= Sirius)
    if (!g_config.nnueLoaded || g_config.mode == Mode::OFF)
        return hceScore;

    // 🦅 EMPATE TEÓRICO DE MATERIAL (fix do jogo dDZlraFi): em finais como KBvK, KNvK,
    //   KNNvK, KvK, a NNUE conta o material (+120..+860) mas é EMPATE — não há mate
    //   possível. Se confiarmos na NNUE, o motor troca para esses finais "a ganhar" e
    //   fica num nulo morto. O HCE (probeEvalFunc → trivialDraw) JÁ sabe que é 0; aqui
    //   respeitamos isso e IGNORAMOS a NNUE quando o lado forte não tem material para
    //   dar mate. (Não toca em KBB/KBN/KQ/KR — esses são ganhos e a NNUE/HCE jogam-nos.)
    if (isInsufficientMatingMaterial(board))
        return hceScore;   // = ~0 do trivialDraw; a NNUE não infla o empate

    // 🦅 PANIC TEMPS : sous panicMs de temps restant, on lâche le NapK9 et on
    // joue en Sirius pur (HCE) — plus rapide, pas de forward pass NNUE, zéro flag.
    if (g_config.panicActive)
        return hceScore;

    // 🦅 MODO PureNNUE (ideia do Maréchal): SEM HCE no meio-jogo. Em vez de usar o HCE
    //   nas folhas (por velocidade), usa as cabeças PEQUENAS da NNUE — quase tão rápidas
    //   como o HCE mas é NNUE consistente (sabe o que treinou). Cascata por profundidade:
    //     folhas (depthLeft < bulletDepth)      → cabeça BULLET(16)  ultra-rápida
    //     intermédio (< midDepth)               → cabeça SMALL(32)
    //     perto da raiz (>= midDepth, ou -1)    → cabeça BIG(L1)     precisa
    //   Os finais já saíram acima (material insuf. → HCE). Aqui é só meio-jogo: NNUE pura.
    if (g_config.pureNNUE)
    {
        int headIdx;
        if (depthLeft < 0)                          headIdx = napoleon::nnue::HEAD_BIG;   // root/datagen
        else if (depthLeft < g_config.pureBulletDepth) headIdx = napoleon::nnue::HEAD_BULLET;
        else if (depthLeft < g_config.pureMidDepth)    headIdx = napoleon::nnue::HEAD_SMALL;
        else                                        headIdx = napoleon::nnue::HEAD_BIG;

        // 🦅 REGRA ESTILO SF (simple-eval decide a rede): se o MATERIAL já decide a posição
        //   (desequilíbrio ≥ pureDecidedMaterial cp), a head bullet chega — precisão extra não
        //   muda o resultado e ganha-se NPS p/ converter. Material conta-se dos bitboards (grátis).
        //   Em posições equilibradas (a maioria) fica a HEAD BIG (a lição dos +66 elo).
        if (headIdx == napoleon::nnue::HEAD_BIG && depthLeft >= 0 && g_config.pureDecidedMaterial > 0)
        {
            static constexpr int MVAL[6] = { 100, 320, 330, 500, 900, 0 };  // P,N,B,R,Q,K
            int diff = 0;
            for (int t = 0; t < 5; ++t)
                diff += MVAL[t] * (board.pieces(Color::WHITE, static_cast<PieceType>(t)).popcount()
                                 - board.pieces(Color::BLACK, static_cast<PieceType>(t)).popcount());
            if (diff >= g_config.pureDecidedMaterial || -diff >= g_config.pureDecidedMaterial)
                headIdx = napoleon::nnue::HEAD_BULLET;
        }

        // 🦅 PASSAGEM A HCE NOS FINAIS (mesma ideia do dual: perto das folhas → HCE):
        //   nos finais (poucas peças), perto das folhas (depthLeft < minDepth), usa a HCE de
        //   finais CALIBRADA em vez da NNUE. Em bullet, a HCE de finais é mais fiável/rápida que
        //   a NNUE comprimida (WDL). Reusa g_config.minDepth (o "setting do dual"). Só nas folhas
        //   (depthLeft >= 0): a chamada de root/datagen (depthLeft<0) mantém-se NNUE.
        if (depthLeft >= 0 && depthLeft < g_config.minDepth
            && board.allPieces().popcount() <= g_config.pureEndgamePieces)
            return hceScore;

        // 🦅 netstats: em PureNNUE a REDE é sempre a big (tem as 3 heads). Usa NoCount (não conta)
        //   p/ definir a rede, e bumpHeadStat conta a HEAD real (bullet/small→small, big→big).
        //   Antes: setActiveNet(0) contava SEMPRE big → estatística errada em PureNNUE.
        napoleon::nnue::setActiveNetNoCount(0);     // rede big (3 heads) — sem contar
        napoleon::nnue::bumpHeadStat(headIdx != napoleon::nnue::HEAD_BIG);
        int nnueScore = evaluateNNUEHead(board, headIdx);
        if (g_config.nnueScale != 100)
            nnueScore = (int)((long)nnueScore * g_config.nnueScale / 100);

        // 🦅 s29 (à Stockfish): COMPLEXITY = |big - small|. Paga o 2º forward (small) só perto da
        //   raiz (depthLeft >= complexityMinDepth) e só se a head usada foi a big (senão não há
        //   "big" p/ comparar). A search lê via lastComplexity() p/ modular as podas.
        g_lastComplexity = -1;
        if (g_config.complexityPruning && depthLeft >= g_config.complexityMinDepth
            && headIdx == napoleon::nnue::HEAD_BIG && napoleon::nnue::dualEnabled())
        {
            napoleon::nnue::setActiveNetNoCount(1);          // small (128)
            int smallScore = evaluateNNUE(board);
            if (g_config.nnueScale != 100)
                smallScore = (int)((long)smallScore * g_config.nnueScale / 100);
            napoleon::nnue::setActiveNetNoCount(0);          // repor a big
            g_lastComplexity = std::abs(nnueScore - smallScore);
        }
        return nnueScore;   // SEM blend com HCE — é o ponto do modo
    }

    // 🦅 SEUIL DE DEPTH : le NapK9 (forward pass coûteux) n'est activé que près
    // de la racine (depthLeft >= minDepth). Aux feuilles (~99% des nœuds), HCE
    // rapide. depthLeft == -1 = appel hors search (root, datagen) → NapK9 actif.
    // Résultat : vitesse Sirius (HCE aux feuilles) + intelligence NapK9 (nœuds clés).
    if (depthLeft >= 0 && depthLeft < g_config.minDepth)
        return hceScore;

    // 🦅 SISTEMA DUAL : escolhe a REDE (big 256 precisa / small 128 rápida).
    //   DOIS modos:
    //   (a) clássico (LazyNNUE OFF): small se |hce|>400 (decidido), senão big.
    //   (b) LAZY (LazyNNUE ON, ideia do Maréchal): a SMALL "abre caminho" em todos
    //       os nós; a BIG só entra quando a small DIVERGE do HCE (posição subtil que
    //       precisa de precisão). Posições claras (small≈hce) ficam na small → mais
    //       rápido → mais profundidade. É o "HCE/big só quando preciso".
    int nnueScore;
    if (napoleon::nnue::dualEnabled())
    {
        if (g_config.lazyNNUE)
        {
            // Passo 1: explora com a SMALL (rápida).
            napoleon::nnue::setActiveNet(1);
            int small = evaluateNNUE(board);
            // Passo 2: a small concorda com o HCE? Se a divergência for pequena, a
            // posição é clara → fica na small. Se grande → precisa da big (precisão).
            if (std::abs(small - hceScore) > g_config.lazyMargin)
            {
                napoleon::nnue::setActiveNet(0);          // zona crítica → BIG
                nnueScore = evaluateNNUE(board);
            }
            else nnueScore = small;                       // posição clara → fica small
        }
        else
        {
            int absH = std::abs(hceScore);
            napoleon::nnue::setActiveNet(absH > 400 ? 1 : 0);   // clássico
            nnueScore = evaluateNNUE(board);
        }
    }
    else
    {
        napoleon::nnue::setActiveNet(0);
        nnueScore = evaluateNNUE(board);
    }

    // Décompression d'échelle : le réseau bullet WDL donne des cp compressés.
    // nnueScale (% ) les ramène vers l'échelle Sirius (~100cp/pion).
    if (g_config.nnueScale != 100)
        nnueScore = (int)((long)nnueScore * g_config.nnueScale / 100);
    int result = hceScore;

    switch (g_config.mode)
    {
        case Mode::PRIMARY:
        {
            // NapK9/NNUE = éval principale (positionnel/tactique), décompressé.
            // Le HCE de Sirius porte le MATÉRIEL EXACT. Le réseau bullet sous-évalue
            // les gros déséquilibres → mélange ADAPTATIF :
            //   • positions équilibrées/positionnelles → surtout NapK9 (il y excelle)
            //   • gros déséquilibre matériel (HCE et NapK9 divergent fort) → plus de HCE
            //
            // On mesure la divergence comme proxy du déséquilibre que le réseau rate.
            int diff = std::abs(hceScore - nnueScore);

            // Poids HCE de base (%) puis montée selon la divergence.
            // baseHce = primaryHceBase (réglable, défaut 15%).
            // À chaque tranche de 200cp de divergence, +12% de HCE, plafonné à 60%.
            int baseHce = g_config.primaryHceBase;          // ex: 15
            int extraHce = (diff / 200) * 12;               // +12% par 200cp d'écart
            int hceW = std::clamp(baseHce + extraHce, baseHce, 60);
            int nnueW = 100 - hceW;

            result = (nnueScore * nnueW + hceScore * hceW) / 100;
            break;
        }

        case Mode::BLEND:
        {
            // Moyenne pondérée : w% NNUE + (100-w)% HCE
            int w = std::clamp(g_config.blendWeightNNUE, 0, 100);
            result = (nnueScore * w + hceScore * (100 - w)) / 100;
            break;
        }

        case Mode::VERIFY:
        {
            // HCE pilote. Si les deux yeux divergent fortement, on tire un peu
            // vers le NNUE (la position est subtile). Sinon HCE inchangé.
            int diff = std::abs(nnueScore - hceScore);
            if (diff >= g_config.divergenceThreshold)
                result = (hceScore * 3 + nnueScore) / 4;
            else
                result = hceScore;
            break;
        }

        case Mode::OFF:
        default:
            result = hceScore;
            break;
    }

    // Sécurité : le score combiné reste dans une plage d'éval (jamais score de mat)
    return std::clamp(result, -3000, 3000);
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers UCI
// ───────────────────────────────────────────────────────────────────────────

// 🦅 s29: a NNUE está a pilotar a eval da search? (DualVision mode!=OFF/PureNNUE, rede carregada,
//   sem panic). Em panic ou OFF, a search corre HCE puro (Sirius) → usa a aspiração original.
bool isNNUEActive()
{
    if (!g_config.nnueLoaded) return false;
    if (g_config.panicActive)  return false;
    if (g_config.mode == Mode::OFF) return false;
    return true;
}

void setMode(int mode)
{
    switch (mode)
    {
        case 1:  g_config.mode = Mode::BLEND;   break;
        case 2:  g_config.mode = Mode::VERIFY;  break;
        case 3:  g_config.mode = Mode::PRIMARY; break;
        default: g_config.mode = Mode::OFF;     break;
    }
}

void setBlendWeight(int weight)
{
    g_config.blendWeightNNUE = std::clamp(weight, 0, 100);
}

void setDivergenceThreshold(int cp)
{
    g_config.divergenceThreshold = std::clamp(cp, 0, 2000);
}

void setScale(int scale)
{
    g_config.nnueScale = std::clamp(scale, 50, 800);
}

void setPrimaryHceBase(int pct)
{
    g_config.primaryHceBase = std::clamp(pct, 0, 60);
}

void setPanicMs(int ms)
{
    g_config.panicMs = std::clamp(ms, 0, 120000);
}

void setPanicActive(bool on)
{
    g_config.panicActive = on;
}

void setMinDepth(int d)
{
    g_config.minDepth = std::clamp(d, 0, 64);
}

void setLazyNNUE(bool on) { g_config.lazyNNUE = on; }
void setLazyMargin(int cp) { g_config.lazyMargin = std::clamp(cp, 0, 600); }
// 🦅 s29: liga/desliga a modulação de podas por complexity (|big-small|) e o seu limiar de depth.
void setComplexityPruning(bool on) { g_config.complexityPruning = on; }
void setComplexityMinDepth(int d)  { g_config.complexityMinDepth = std::clamp(d, 0, 64); }

void setPureNNUE(bool on) {
    g_config.pureNNUE = on;
    // 🦅 FIX: ligar PureNNUE TEM de ativar a NNUE (mode != OFF), senão o combine saía p/ HCE puro
    //   e o setoption não fazia nada (era o bug "PureNNUE on/off dá igual"). Se a desligam, fica
    //   PRIMARY (NNUE ativa, sem cascata) — não voltamos a OFF aqui p/ não cegar a rede sem querer.
    if (g_config.mode == Mode::OFF)
        g_config.mode = Mode::PRIMARY;
}
void setPureEndgamePieces(int n) { g_config.pureEndgamePieces = std::clamp(n, 0, 32); }
void setPureDepths(int bulletDepth, int midDepth) {
    g_config.pureBulletDepth = std::clamp(bulletDepth, 0, 30);
    g_config.pureMidDepth = std::clamp(midDepth, 0, 40);
}

void setPureDecidedMaterial(int cp)
{
    g_config.pureDecidedMaterial = std::clamp(cp, 0, 2000);
}

const char* modeName()
{
    switch (g_config.mode)
    {
        case Mode::BLEND:   return "BLEND";
        case Mode::VERIFY:  return "VERIFY";
        case Mode::PRIMARY: return "PRIMARY";
        case Mode::OFF:
        default:            return "OFF";
    }
}

}  // namespace napoleon::dualvision
