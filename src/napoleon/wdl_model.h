#pragma once

// Modelo WDL próprio do littleindian — respeita exatamente a fórmula usada
// no treino da rede (bullet_integration/napk9_train*.rs): a rede prevê
// E = sigmoid(score_cp / 408) como "score esperado" (= P(vitória) +
// 0.5·P(empate)). A decomposição em W/D/L (win=E², loss=(1-E)², draw=
// 2E(1-E)) não usa nenhuma constante extra — só E, que já está ancorado
// no treino.

namespace napoleon::wdl {

struct Probs {
    double win;
    double draw;
    double loss;
};

// scoreCp: eval em centipawns, do ponto de vista de quem tem a vez de jogar.
Probs expectedWDL(int scoreCp);

}  // namespace napoleon::wdl
