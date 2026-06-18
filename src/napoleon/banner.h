#pragma once
// Banner ASCII de arranque (estilo Lc0). Só aparece em terminal interativo
// (isatty) — uma GUI/Lichess que pilota o motor por pipe nunca o vê, por
// isso não interfere com o protocolo UCI.
#include <cstdio>

#if defined(_WIN32)
  #include <io.h>
  #define ISATTY _isatty
  #define FILENO _fileno
#else
  #include <unistd.h>
  #define ISATTY isatty
  #define FILENO fileno
#endif

namespace napoleon {

inline void printBanner() {
    if (!ISATTY(FILENO(stdout)))
        return;

    const char* O   = "\033[1;33m";   // pena: ocre/dourado
    const char* R   = "\033[1;31m";   // pena: vermelho
    const char* W   = "\033[1;37m";   // nome: branco vivo
    const char* DIM = "\033[2;37m";   // tagline: cinzento
    const char* Z   = "\033[0m";      // reset

    std::printf("\n");
    std::printf("              %s,%s\n",            O, Z);
    std::printf("             %s/|%s\\%s\n",         O, R, Z);
    std::printf("            %s//|%s\\\\%s\n",       O, R, Z);
    std::printf("           %s///|%s\\\\\\%s\n",     O, R, Z);
    std::printf("          %s////|%s\\\\\\\\%s\n",   O, R, Z);
    std::printf("              %s|||%s\n",           R, Z);
    std::printf("              %s|||%s\n",           R, Z);
    std::printf("               %s|%s\n",            R, Z);
    std::printf("\n");
    std::printf("   %slittleindian%s  %s·%s  %sUCI chess engine, NNUE-native (NapK9)%s\n",
                W, Z, DIM, Z, DIM, Z);
    std::printf("\n");
    std::fflush(stdout);
}

} // namespace napoleon
