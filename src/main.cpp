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
    uci::loop();
    return 0;
}
