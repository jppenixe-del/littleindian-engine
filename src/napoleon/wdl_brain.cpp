#include "wdl_brain.h"
#include "../board.h"
#include "wdl_model.h"

#include <algorithm>
#include <cmath>

namespace napoleon::wdlbrain
{

Config g_config = {};

WdlProbs probs(const Board& /*board*/, int scoreCp)
{
    // Modelo WDL próprio (ver wdl_model.h) — board reservado para uma
    // futura variante dependente de material, ainda sem calibração.
    napoleon::wdl::Probs w = napoleon::wdl::expectedWDL(scoreCp);
    return { w.win, w.draw, w.loss };
}

double timeFactor(const Board& board, int scoreCp)
{
    if (!g_config.enabled || !g_config.timeModulation)
        return 1.0;

    // Sécurité : sur scores de mat, jouer vite directement (pas besoin de WDL).
    if (std::abs(scoreCp) >= 20000)
        return 0.40;

    WdlProbs p = probs(board, scoreCp);

    // Criticité = à quel point la partie est "en jeu".
    //   draw élevé + win/loss proches → critique (chaque coup compte) → +temps
    //   win très élevé OU loss très élevé → décidé → -temps (jouer vite)
    //
    // decisiveness = |win - loss| dans [0..1]. Proche de 0 = équilibré (critique).
    double decisiveness = std::abs(p.win - p.loss);

    // Position TRÈS décidée (gagnée ou perdue clairement) → jouer vite.
    if (decisiveness > 0.85)
        return 0.45;   // ~2x plus rapide : pas besoin de calculer un résultat acquis
    if (decisiveness > 0.65)
        return 0.65;

    // Position équilibrée et vivante (beaucoup de draw + win/loss serrés) → réfléchir.
    if (decisiveness < 0.25 && p.draw < 0.70)
        return 1.45;   // ~1.5x : moment critique, vaut le temps
    if (decisiveness < 0.40)
        return 1.20;

    return 1.0;  // situation normale
}

int fearAdjustment(const Board& board, int scoreCp)
{
    if (!g_config.enabled || g_config.fearStrength <= 0)
        return 0;

    WdlProbs p = probs(board, scoreCp);

    // Si la proba de perte est notable, on applique une petite prudence :
    // un léger malus qui pousse le moteur à préférer des lignes plus sûres
    // (le malus est symétrique au score, donc il décourage les positions
    //  où loss est élevé même si le score brut paraît ok — anti-blunder).
    //
    // strength 0..100 → échelle le malus max (~ -40cp à pleine peur).
    double s = static_cast<double>(g_config.fearStrength) / 100.0;

    // lossPressure : 0 si loss faible, monte si loss > 0.30
    double lossPressure = std::max(0.0, p.loss - 0.30) / 0.70;  // 0..1
    int malus = static_cast<int>(-40.0 * s * lossPressure);

    // À l'inverse, si on est très confiant (win élevé), petit bonus pour
    // encourager à convertir (pousser l'avantage plutôt que mollir).
    double winConfidence = std::max(0.0, p.win - 0.70) / 0.30;   // 0..1
    int bonus = static_cast<int>(15.0 * s * winConfidence);

    return malus + bonus;
}

void setEnabled(bool on)        { g_config.enabled = on; }
void setFearStrength(int s)     { g_config.fearStrength = std::clamp(s, 0, 100); }
void setTimeModulation(bool on) { g_config.timeModulation = on; }

}  // namespace napoleon::wdlbrain
