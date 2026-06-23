#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include "defs.h"
#include "attacks.h"
#include "uci.h"
#include "tt.h"
#include "napoleon/nnue_net.h"
#include "napoleon/banner.h"

int main(int argc, char* argv[]) {
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
    // 🦅 Suporte a "./littleindian <comando> [<comando> ...]" como argumentos de linha de
    // comando (convenção de frameworks de teste como o OpenBench: invocam `./binary bench`
    // ou `./binary "setoption name X value Y" bench quit` -- cada elemento da lista de
    // argumentos é 1 comando UCI completo, mesmo contendo espaços). "quit" é adicionado no
    // fim se não vier já incluído, para o loop terminar como esperado.
    if (argc > 1) {
        std::ostringstream cmds;
        bool has_quit = false;
        for (int i = 1; i < argc; ++i) {
            cmds << argv[i] << '\n';
            if (std::string(argv[i]) == "quit") has_quit = true;
        }
        if (!has_quit) cmds << "quit\n";
        static std::istringstream fake_stdin(cmds.str());
        std::cin.rdbuf(fake_stdin.rdbuf());
    }
    uci::loop();
    return 0;
}
