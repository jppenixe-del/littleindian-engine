#include "wdl_model.h"
#include <cmath>

namespace napoleon::wdl {

namespace {
// Mesma escala usada no treino (bullet_integration/napk9_train*.rs:
// eval_scale = SCALE = 408, blend wdl_weight*resultado + (1-wdl_weight)*
// sigmoid(cp/408)). A rede já prevê E = sigmoid(score/408) como "score
// esperado" (= P(vitória) + 0.5·P(empate)) — não se inventa uma constante
// extra aqui, respeita-se exatamente a fórmula do treino.
constexpr double SCALE = 408.0;
}  // namespace

Probs expectedWDL(int scoreCp) {
    double e = 1.0 / (1.0 + std::exp(-double(scoreCp) / SCALE));  // E ∈ [0,1]

    // Decomposição em W/D/L a partir de E, sem constantes extra: win = E²,
    // loss = (1-E)², draw = 1 - win - loss = 2E(1-E). Em E=0.5 dá 25/50/25
    // (máximo de empate); nos extremos tende para 100% decisivo. É a forma
    // mais simples consistente com E sem assumir mais nada sobre a posição.
    double win  = e * e;
    double loss = (1.0 - e) * (1.0 - e);
    double draw = 2.0 * e * (1.0 - e);
    return { win, draw, loss };
}

}  // namespace napoleon::wdl
