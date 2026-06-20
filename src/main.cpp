#include <cstdlib>
#include "defs.h"
#include "attacks.h"
#include "uci.h"
#include "tt.h"
#include "napoleon/nnue_net.h"
#include "napoleon/banner.h"

int main() {
    napoleon::printBanner();
    attacks::init();
    zobrist::init();
    gTT.resize(TT::DEFAULT_MB);
    // Load embedded net if available
    if (napoleon::nnue::hasEmbeddedNet())
        napoleon::nnue::loadEmbedded();
    // 🦅 LITTLEINDIAN_NO_NNUE=1: arranca já em modo HCE (sem rede), p/ testes manuais
    // sem ter de mandar "setoption name EvalFile value none" em todo "go". Não muda o
    // default real do motor (SPRT/produção continuam sem a env var, NNUE como sempre).
    if (std::getenv("LITTLEINDIAN_NO_NNUE"))
        napoleon::nnue::unload();
    uci::loop();
    return 0;
}
