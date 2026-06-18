// ═══════════════════════════════════════════════════════════════════════════
// 🦅 Redes NapK9 EMBUTIDAS no binário (façon Stockfish via incbin)
// ═══════════════════════════════════════════════════════════════════════════
//   Sistema DUAL: duas redes embutidas no executável.
//     BIG   (-DEMBEDDED_NET_PATH)       : rede grande (256), precisa.
//     SMALL (-DEMBEDDED_SMALL_NET_PATH) : rede pequena (128), rápida.
//   Cada uma é incluída via .incbin se o respetivo -D for passado na compilação.
// ═══════════════════════════════════════════════════════════════════════════
#include <cstddef>
#include <cstdint>

// ───────────────────────── REDE BIG (principal) ─────────────────────────────
#if defined(EMBEDDED_NET_PATH)
asm(
    ".section .rodata\n"
    ".global g_embeddedNetStart\n"
    ".balign 64\n"
    "g_embeddedNetStart:\n"
    ".incbin \"" EMBEDDED_NET_PATH "\"\n"
    ".global g_embeddedNetEnd\n"
    ".balign 1\n"
    "g_embeddedNetEnd:\n"
    ".previous\n"
);
extern const unsigned char g_embeddedNetStart[];
extern const unsigned char g_embeddedNetEnd[];
namespace napoleon::nnue {
const uint8_t* embeddedNetData()  { return (const uint8_t*)g_embeddedNetStart; }
size_t         embeddedNetSize()  { return (size_t)(g_embeddedNetEnd - g_embeddedNetStart); }
bool           hasEmbeddedNet()   { return embeddedNetSize() > 0; }
}
#else
namespace napoleon::nnue {
const uint8_t* embeddedNetData()  { return nullptr; }
size_t         embeddedNetSize()  { return 0; }
bool           hasEmbeddedNet()   { return false; }
}
#endif

// ───────────────────────── REDE SMALL (dual) ────────────────────────────────
#if defined(EMBEDDED_SMALL_NET_PATH)
asm(
    ".section .rodata\n"
    ".global g_embeddedSmallStart\n"
    ".balign 64\n"
    "g_embeddedSmallStart:\n"
    ".incbin \"" EMBEDDED_SMALL_NET_PATH "\"\n"
    ".global g_embeddedSmallEnd\n"
    ".balign 1\n"
    "g_embeddedSmallEnd:\n"
    ".previous\n"
);
extern const unsigned char g_embeddedSmallStart[];
extern const unsigned char g_embeddedSmallEnd[];
namespace napoleon::nnue {
const uint8_t* embeddedSmallNetData() { return (const uint8_t*)g_embeddedSmallStart; }
size_t         embeddedSmallNetSize() { return (size_t)(g_embeddedSmallEnd - g_embeddedSmallStart); }
bool           hasEmbeddedSmallNet()  { return embeddedSmallNetSize() > 0; }
}
#else
namespace napoleon::nnue {
const uint8_t* embeddedSmallNetData() { return nullptr; }
size_t         embeddedSmallNetSize() { return 0; }
bool           hasEmbeddedSmallNet()  { return false; }
}
#endif
