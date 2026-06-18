#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 NAPOLEON — WDL BRAIN (fear, gestion du temps, prise de décision)
// ═══════════════════════════════════════════════════════════════════════════
//
// Utilise le modèle WDL de Sirius (expectedWDL : winProb / drawProb / lossProb)
// pour moduler le COMPORTEMENT du moteur au-delà du simple score :
//
//   • FEAR        : si proba de perte élevée → préférer la solidité (anti-blunder)
//   • TEMPS       : position critique (WDL serré) → réfléchir plus ;
//                   position décidée (gagnée/perdue) → jouer vite (anti-flag)
//   • DÉCISION    : comparer les coups par leur valeur WDL, pas juste le score cp
//
// Le WDL brain est OPT-IN (option UCI WdlBrain). OFF par défaut = Sirius pur.
// ═══════════════════════════════════════════════════════════════════════════

#include "../defs.h"

class Board;

namespace napoleon::wdlbrain
{

struct Config
{
    bool enabled = false;        // OFF par défaut (Sirius pur)
    int  fearStrength = 50;      // 0..100 : intensité de la prudence
    bool timeModulation = true;  // module le temps selon la criticité WDL
};

extern Config g_config;

// Probabilités WDL pour la position courante, vue du side-to-move.
struct WdlProbs
{
    double win;
    double draw;
    double loss;
};

// Calcule W/D/L (0..1) à partir du board et du score (cp, POV side-to-move).
WdlProbs probs(const Board& board, int scoreCp);

// FACTEUR DE TEMPS [0.3 .. 1.6] : combien moduler le temps de réflexion.
//   ~1.0  = normal
//   <1.0  = position décidée (gagnée OU perdue) → jouer plus vite
//   >1.0  = position critique/équilibrée → réfléchir plus
// timeModulation doit être ON, sinon retourne 1.0.
double timeFactor(const Board& board, int scoreCp);

// AJUSTEMENT FEAR : petit malus/bonus (cp) à appliquer pour préférer la
// solidité quand la proba de perte est élevée. Retourne 0 si fear OFF.
// Positif = on est confiant (peut pousser), négatif = prudence.
int fearAdjustment(const Board& board, int scoreCp);

// Helpers UCI
void setEnabled(bool on);
void setFearStrength(int s);
void setTimeModulation(bool on);

}  // namespace napoleon::wdlbrain
