#include "nnue_net.h"
#include <atomic>
#include "../board.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <memory>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace napoleon::nnue
{

// 🦅 Rede embutida (définie dans embedded_net.cpp via .incbin).
const uint8_t* embeddedNetData();
size_t         embeddedNetSize();
bool           hasEmbeddedNet();
// 🦅 Sistema DUAL : segunda rede (small) embutida.
const uint8_t* embeddedSmallNetData();
size_t         embeddedSmallNetSize();
bool           hasEmbeddedSmallNet();

// 🦅 MAX_L1: tamanho MÁXIMO de acumulador que o motor suporta. Os buffers do acc são fixos a este
//   valor (stack). A big pode ter L1 até MAX_L1. Antes era 1024 hardcoded → a 1536 (escala sykora)
//   não cabia (overflow). Subido p/ 1536. Se um dia fores além, muda SÓ aqui. (custo: stack;
//   1536 int32 × ~9 buffers ≈ 55KB de stack por thread — OK.)
static constexpr int   MAX_L1            = 1536;
static constexpr int   TOTAL_FEATURES   = 22528;
// 🦅 V10 FULL THREATS (à la Reckless/SF): (lado da vítima 2) × (relação ataque/defesa 2) ×
//   (atacante 6, incl. rei) × (vítima 6, incl. rei) × (casa da vítima 64) = 9216.
//   side: 0 = vítima MINHA, 1 = vítima DELES (na perspetiva). rel: 0 = ATAQUE (atacante de cor
//   oposta à vítima), 1 = DEFESA (mesma cor). Pares impossíveis ficam a peso 0 no treino.
//   ⚠️ CONTRATO DE CALIBRAÇÃO: o gerador Rust (napk9_v10_features.rs) TEM de produzir os MESMOS
//   índices — validar com training/calibra_threats.py SEMPRE que se mexe num dos lados.
static constexpr int THREAT_FEATURES_FULL = 2 * 2 * 6 * 6 * 64;  // 9216
static inline int makeThreatFull(int side, int rel, int att, int vic, int sq) {
    return (((side * 2 + rel) * 6 + att) * 6 + vic) * 64 + sq;
}
static constexpr int   FEATURES_PER_KB  = 704;
static constexpr int   MATERIAL_BUCKETS = 8;
// 🦅 OUTPUT_SCALE_CP: out (≈cp_SF/400 do treino) × isto = score em unidades-Sirius. O Sirius usa
//   NormalizeToPawnValue=102 (102 = 1 peão). P/ alinhar EXATO: 400 × 102/100 = 408 (era 410, ~0.5%
//   off). Assim os scores da NNUE batem com as margens de poda do Sirius (futility/LMR em 102/peão).
static constexpr float OUTPUT_SCALE_CP  = 408.0f;

static constexpr int BULLET_L1 = 16, BULLET_L2 = 32;
static constexpr int SMALL_L1  = 32, SMALL_L2  = 32;
// 🦅 SFNNv13: a big afunila o acc GRANDE (L1×2) p/ densas PEQUENAS fixas, SEM camada L1→L1 gorda.
//   1ª densa: L1×2 → BIG_DL1(32) ; 2ª: 32 → BIG_DL2(32) ; 3ª: 32 → 1×8. (SF usa 31/32; usamos 32/32.)
static constexpr int BIG_DL1   = 32, BIG_DL2   = 32;

struct Head
{
    std::vector<int8_t> l1_w[MATERIAL_BUCKETS];
    std::vector<float>  l1_b[MATERIAL_BUCKETS];
    std::vector<int8_t> l2_w[MATERIAL_BUCKETS];
    std::vector<float>  l2_b[MATERIAL_BUCKETS];
    std::vector<int8_t> l3_w[MATERIAL_BUCKETS];
    float               l3_b[MATERIAL_BUCKETS] = {};
    float l1_scale[MATERIAL_BUCKETS] = {1,1,1,1,1,1,1,1};
    float l2_scale[MATERIAL_BUCKETS] = {1,1,1,1,1,1,1,1};
    float l3_scale[MATERIAL_BUCKETS] = {1,1,1,1,1,1,1,1};
    int l1_out = 0, l2_out = 0;
};

// 🦅 s29: silêncio por defeito (estilo Stockfish). Banners só com setoption NnueVerbose=true.
static bool g_nnueVerbose = false;
void setVerbose(bool on) { g_nnueVerbose = on; }
bool verbose() { return g_nnueVerbose; }

struct Network
{
    int L1 = 0;
    std::vector<int16_t> accWeight;
    std::vector<int16_t> accBias;
    std::vector<int32_t> psqt;
    float qa = 255.0f, qb = 64.0f;
    float accScale = 1.0f / 127.0f;
    Head bullet, small, big;
    bool has3heads = false;
    bool loaded = false;

    // 🦅 V9.6 — THREAT INPUTS (ameaças). Bloco de features NÃO-incremental somado
    //   FRESH a cada eval, ANTES da ClippedReLU (passa pela mesma ativação/L1).
    //   Layout por perspetiva: 5 vítimas {P,N,B,R,Q} × 64 casas × 2 direções
    //     dir 0 = a MINHA peça nesta casa está atacada (defesa/pendurada)
    //     dir 1 = EU ataco a peça adversária nesta casa (iniciativa)
    //   = 640 features. threatWeight = [640 × L1] int16 (mesma escala QA que accWeight).
    //   Se ausente/zero → contribuição zero → eval IDÊNTICO ao V9.5 (retrocompat total).
    bool hasThreats = false;
    std::vector<int16_t> threatWeight;   // [THREAT_FEATURES * L1], int16
    bool fullThreats = false;            // 🦅 V10: true → tensor 9216 (flags bit 8); false → 640
    uint32_t flags = 0; uint32_t version = 0; size_t bytes = 0; size_t nThreats = 0;   // 🦅 info p/ printNetInfo

    // 🦅 V9.5 — Chaos/Fear Head (GLOBALE, pas par bucket). Format export Python :
    //   chaos_l1 (L1*2 → 32), chaos_l2 (32 → 16), chaos_l3 (16 → 1).
    //   weights ×QB (int16), bias ×QA×QB (int32). Sortie sigmoid → [0,1].
    bool hasChaos = false;
    std::vector<int16_t> ch_l1_w, ch_l2_w, ch_l3_w;   // poids transposés
    std::vector<int32_t> ch_l1_b, ch_l2_b, ch_l3_b;   // biais
    int ch_l1_out = 32, ch_l2_out = 16;               // tailles (chaos_l1→32, chaos_l2→16)
};

// 🦅 SISTEMA DUAL (façon Stockfish) : DUAS redes independentes.
//   g_netBig   = rede grande (ex: siriux_256, L1=256) — PRECISA, posições críticas.
//   g_netSmall = rede pequena (ex: siriux_128, L1=128) — RÁPIDA, posições decididas.
//   Cada uma tem o seu accumulador (L1 diferente → incompatíveis).
//   g_net aponta para a rede ATIVA no momento da avaliação (alternada por margem).
static Network g_netBig;
static Network g_netSmall;
// 🦅 FIX s29 (CORE DUMP multi-thread): g_netPtr era global mutável. setActiveNet() muda-o ANTES
//   de cada evaluate; com N threads, a thread A punha small e a B punha big NO MESMO instante →
//   corrida → a thread A lia g_net.L1 da big (1024) mas com buffers da small (640×128) →
//   threatWeight[idx*1024] num vector de 640*128 → heap-overflow → SIGSEGV. thread_local dá a
//   cada thread o SEU ponteiro de rede ativa (idem g_netIdx). É a causa RAIZ do crash com Threads>1.
static thread_local Network* g_netPtr = &g_netBig;   // rede ativa (POR THREAD)
#define g_net (*g_netPtr)               // todas as referências usam a rede ativa
static bool g_dualLoaded = false;       // true se ambas as redes carregaram

namespace
{
    int64_t lebOne(const uint8_t* buf, size_t& pos, size_t end)
    {
        uint64_t result = 0; int shift = 0;
        while (pos < end)
        {
            uint8_t b = buf[pos++];
            result |= (uint64_t)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
            if (shift > 63) return 0;
        }
        return (int64_t)(result >> 1) ^ -(int64_t)(result & 1);
    }
    void lebI16(const uint8_t* buf, size_t bufSize, int16_t* out, size_t count)
    {
        size_t pos = 0;
        for (size_t i = 0; i < count && pos < bufSize; ++i)
            out[i] = (int16_t)std::clamp<int64_t>(lebOne(buf, pos, bufSize), -32768, 32767);
    }
    void lebI32(const uint8_t* buf, size_t bufSize, int32_t* out, size_t count)
    {
        size_t pos = 0;
        for (size_t i = 0; i < count && pos < bufSize; ++i)
            out[i] = (int32_t)std::clamp<int64_t>(lebOne(buf, pos, bufSize), -2147483648LL, 2147483747LL);
    }

    const char LEB_MAGIC[17] = {'C','O','M','P','R','E','S','S','E','D','_','L','E','B','1','2','8'};

    const uint8_t* readChunk(const std::vector<uint8_t>& d, size_t& pos, uint32_t& outSize)
    {
        if (pos + 17 + 4 > d.size()) return nullptr;
        if (std::memcmp(d.data() + pos, LEB_MAGIC, 17) != 0) return nullptr;
        pos += 17;
        std::memcpy(&outSize, d.data() + pos, 4);
        pos += 4;
        if (pos + outSize > d.size()) return nullptr;
        const uint8_t* p = d.data() + pos;
        pos += outSize;
        return p;
    }

    int detectBigL2(int l1)
    {
        if (l1 <= 128) return 16;
        if (l1 <= 256) return 32;
        if (l1 <= 512) return 32;
        if (l1 <= 768) return 48;
        if (l1 <= 1024) return 64;
        return 96;
    }
    int materialBucket(int pc) { return std::clamp((pc - 1) / 4, 0, 7); }
}

static bool loadBattalion(Head& h, const std::vector<uint8_t>& buf, size_t& pos,
                          int l1_in, int l1_out, int l2_out, float qa, float qb)
{
    h.l1_out = l1_out; h.l2_out = l2_out;
    float biasDequant = 1.0f / (qa * qb);

    for (int bk = 0; bk < MATERIAL_BUCKETS; ++bk)
    {
        {
            uint32_t cw; const uint8_t* w = readChunk(buf, pos, cw); if (!w) return false;
            std::vector<int16_t> tmp((size_t)l1_in * l1_out);
            lebI16(w, cw, tmp.data(), tmp.size());
            h.l1_w[bk].assign(tmp.size() + 32, 0);   // 🦅 FIX s29: +32 margem AVX (loadu 32B na última row)
            for (size_t i = 0; i < tmp.size(); ++i) h.l1_w[bk][i] = (int8_t)std::clamp<int>(tmp[i], -127, 127);
            uint32_t cb; const uint8_t* b = readChunk(buf, pos, cb); if (!b) return false;
            std::vector<int32_t> bt(l1_out); lebI32(b, cb, bt.data(), l1_out);
            h.l1_b[bk].resize(l1_out);
            for (int i = 0; i < l1_out; ++i) h.l1_b[bk][i] = (float)bt[i] * biasDequant;
            h.l1_scale[bk] = 1.0f / qb;
        }
        {
            uint32_t cw; const uint8_t* w = readChunk(buf, pos, cw); if (!w) return false;
            std::vector<int16_t> tmp((size_t)l1_out * l2_out);
            lebI16(w, cw, tmp.data(), tmp.size());
            h.l2_w[bk].assign(tmp.size() + 32, 0);   // 🦅 FIX s29: +32 margem AVX
            for (size_t i = 0; i < tmp.size(); ++i) h.l2_w[bk][i] = (int8_t)std::clamp<int>(tmp[i], -127, 127);
            uint32_t cb; const uint8_t* b = readChunk(buf, pos, cb); if (!b) return false;
            std::vector<int32_t> bt(l2_out); lebI32(b, cb, bt.data(), l2_out);
            h.l2_b[bk].resize(l2_out);
            for (int i = 0; i < l2_out; ++i) h.l2_b[bk][i] = (float)bt[i] * biasDequant;
            h.l2_scale[bk] = 1.0f / qb;
        }
        {
            uint32_t cw; const uint8_t* w = readChunk(buf, pos, cw); if (!w) return false;
            std::vector<int16_t> tmp(l2_out); lebI16(w, cw, tmp.data(), l2_out);
            h.l3_w[bk].assign(l2_out + 32, 0);   // 🦅 margem AVX
            for (int i = 0; i < l2_out; ++i) h.l3_w[bk][i] = (int8_t)std::clamp<int>(tmp[i], -127, 127);
            uint32_t cb; const uint8_t* b = readChunk(buf, pos, cb); if (!b) return false;
            int32_t bt = 0; lebI32(b, cb, &bt, 1);
            h.l3_b[bk] = (float)bt * biasDequant;
            h.l3_scale[bk] = 1.0f / qb;
        }
    }
    return true;
}

static bool loadNapK9(const std::vector<uint8_t>& buf)
{
    if (buf.size() < 36) return false;

    uint32_t version, L1, nBuckets, nFeatures, qa, qb, flags;
    std::memcpy(&version,    buf.data() + 8,  4);
    std::memcpy(&L1,         buf.data() + 12, 4);
    std::memcpy(&nBuckets,   buf.data() + 16, 4);
    std::memcpy(&nFeatures,  buf.data() + 20, 4);
    std::memcpy(&qa,         buf.data() + 24, 4);
    std::memcpy(&qb,         buf.data() + 28, 4);
    std::memcpy(&flags,      buf.data() + 32, 4);

    if (nFeatures != (uint32_t)TOTAL_FEATURES) return false;
    if (nBuckets != (uint32_t)MATERIAL_BUCKETS) return false;
    // 🦅 PROTEÇÃO: os buffers de acumulador são fixos a MAX_L1 (stack). Uma rede com L1 > MAX_L1
    //   causaria overflow nos loops "for i < L1". Rejeita com mensagem clara em vez de corromper.
    if ((int)L1 > MAX_L1) {
        std::fprintf(stderr, "🔴 [NapK9] L1=%u > MAX_L1=%d — rede grande demais p/ este motor. "
                             "Recompila com MAX_L1 maior em nnue_net.cpp.\n", L1, MAX_L1);
        return false;
    }

    g_net.L1 = (int)L1; g_net.qa = (float)qa; g_net.qb = (float)qb;
    g_net.flags = flags; g_net.version = version; g_net.bytes = buf.size();   // 🦅 info p/ printNetInfo
    // 🦅 SFNNv13: a big já não usa L2 variável (detectBigL2). Afunila L1×2 → BIG_DL1 → BIG_DL2 → 1×8.

    if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NapK9] v=%u L1=%u buckets=%u QA=%u QB=%u flags=%u\n",
                 version, L1, nBuckets, qa, qb, flags);

    size_t pos = 36;

    { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
      g_net.accBias.resize(L1); lebI16(c, cs, g_net.accBias.data(), L1); }

    { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
      g_net.accWeight.assign((size_t)TOTAL_FEATURES * L1 + 32, 0);   // 🦅 FIX s29: margem AVX
      lebI16(c, cs, g_net.accWeight.data(), g_net.accWeight.size()); }

    if (flags & 1) {
        uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
        g_net.psqt.assign((size_t)TOTAL_FEATURES * MATERIAL_BUCKETS + 32, 0);   // 🦅 margem AVX
        lebI32(c, cs, g_net.psqt.data(), g_net.psqt.size());
    }

    if (flags & 2) {
        g_net.has3heads = true;
        if (!loadBattalion(g_net.bullet, buf, pos, L1*2, BULLET_L1, BULLET_L2, g_net.qa, g_net.qb)) return false;
        if (!loadBattalion(g_net.small,  buf, pos, L1*2, SMALL_L1,  SMALL_L2,  g_net.qa, g_net.qb)) return false;
        if (!loadBattalion(g_net.big,    buf, pos, L1*2, BIG_DL1,   BIG_DL2,   g_net.qa, g_net.qb)) return false;
    }

    // 🦅 V9.5 — Chaos/Fear Head (flag bit2). GLOBALE : chaos_l1(L1*2→32),
    // chaos_l2(32→16), chaos_l3(16→1). Format : pour chaque couche, chunk
    // weight (int16, ×QB transposé) puis chunk bias (int32, ×QA×QB).
    if (flags & 4) {
        const int L1x2 = (int)L1 * 2;
        const int o1 = g_net.ch_l1_out;   // 32
        const int o2 = g_net.ch_l2_out;   // 16
        // chaos_l1 : weight [L1x2 * o1], bias [o1]
        { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
          g_net.ch_l1_w.assign((size_t)L1x2 * o1 + 32, 0); lebI16(c, cs, g_net.ch_l1_w.data(), (size_t)L1x2 * o1); }
        { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
          g_net.ch_l1_b.resize(o1); lebI32(c, cs, g_net.ch_l1_b.data(), o1); }
        // chaos_l2 : weight [o1 * o2], bias [o2]
        { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
          g_net.ch_l2_w.assign((size_t)o1 * o2 + 32, 0); lebI16(c, cs, g_net.ch_l2_w.data(), (size_t)o1 * o2); }
        { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
          g_net.ch_l2_b.resize(o2); lebI32(c, cs, g_net.ch_l2_b.data(), o2); }
        // chaos_l3 : weight [o2 * 1], bias [1]
        { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
          g_net.ch_l3_w.assign(o2 + 32, 0); lebI16(c, cs, g_net.ch_l3_w.data(), o2); }
        { uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs); if (!c) return false;
          g_net.ch_l3_b.resize(1); lebI32(c, cs, g_net.ch_l3_b.data(), 1); }
        g_net.hasChaos = !g_net.ch_l1_w.empty() && !g_net.ch_l3_w.empty();
        if (g_net.hasChaos)
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NapK9] Chaos/Fear Head chargée 🌪️\n");
    }

    // 🦅 V9.6 — THREAT INPUTS (bloco OPCIONAL, no fim do ficheiro). Um chunk extra
    //   com [THREAT_FEATURES(640) × L1] int16. Ausente nos ficheiros V9.5 → readChunk
    //   devolve nullptr e ficamos com hasThreats=false (eval idêntico ao antigo).
    {
        uint32_t cs; const uint8_t* c = readChunk(buf, pos, cs);
        if (c)
        {
            // 🦅 V10: flags bit 8 → tensor FULL (9216); senão o clássico (640).
            g_net.fullThreats = (flags & 8) != 0;
            const size_t nThr = g_net.fullThreats ? (size_t)THREAT_FEATURES_FULL : 640;
            const size_t need = nThr * L1;
            // 🦅 FIX s29 (core dump multi-thread): +16 de margem AVX. addThreatRowI16 lê 16 int16
            //   por loadu; na ÚLTIMA row a leitura podia tocar 1 elemento além do fim do vector
            //   (resize exato) → heap-overflow que o multi-thread expunha (SIGSEGV). A margem
            //   garante que qualquer loadu de 16 na última row fica in-bounds.
            g_net.threatWeight.assign(need + 16, 0);
            lebI16(c, cs, g_net.threatWeight.data(), need);
            // só ativa se houver algum peso não-nulo (zero → contribuição zero de qualquer forma).
            bool nonZero = false;
            for (int16_t v : g_net.threatWeight) if (v != 0) { nonZero = true; break; }
            g_net.hasThreats = nonZero;
            if (g_net.hasThreats)
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NapK9] Threat inputs chargés ⚔️ (%zu × %d)%s\n", nThr, L1,
                             g_net.fullThreats ? " FULL V10 🔥" : " (640 clássico)");
            else
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NapK9] ⚪ SEM threat inputs (chunk a zeros) — rede rápida, eval só peças\n");
        }
        else
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NapK9] ⚪ SEM threat inputs (chunk ausente) — rede rápida, eval só peças\n");
    }

    g_net.loaded = !g_net.accWeight.empty() && g_net.has3heads;
    return g_net.loaded;
}

namespace
{
    constexpr int BUCKET_MAP[64] = {
         0, 1, 2, 3, 3, 2, 1, 0,    4, 5, 6, 7, 7, 6, 5, 4,
         8, 9,10,11,11,10, 9, 8,   12,13,14,15,15,14,13,12,
        16,17,18,19,19,18,17,16,   20,21,22,23,23,22,21,20,
        24,25,26,27,27,26,25,24,   28,29,30,31,31,30,29,28,
    };

    inline int engCode(int pieceType, int color) { return (color == 0) ? (pieceType + 1) : (pieceType + 7); }

    inline int pieceKindW(int pc)
    {
        if (pc == 12) return 0;
        switch (pc) {
            case 5: return 1; case 4: return 2; case 3: return 3; case 2: return 4; case 1: return 5;
            case 11: return 6; case 10: return 7; case 9: return 8; case 8: return 9; case 7: return 10;
            default: return -1;
        }
    }
    inline int pieceKindB(int pc)
    {
        if (pc == 6) return 0;
        switch (pc) {
            case 11: return 1; case 10: return 2; case 9: return 3; case 8: return 4; case 7: return 5;
            case 5: return 6; case 4: return 7; case 3: return 8; case 2: return 9; case 1: return 10;
            default: return -1;
        }
    }

    inline int makeFeat(int bucket, int kind, int sq) { return bucket * 704 + kind * 64 + sq; }

    // 🦅 V9.6 — THREAT INPUTS : índices do bloco de ameaças (640 por perspetiva).
    //   vítima v ∈ {0..4} = {P,N,B,R,Q}, casa s ∈ {0..63}, direção d ∈ {0,1}.
    //   idx = (d*5 + v)*64 + s   →   [0, 640)
    static constexpr int THREAT_VICTIMS  = 5;     // P,N,B,R,Q (rei nunca é "vítima" de ameaça)
    static constexpr int THREAT_DIRS     = 2;     // 0 = somos atacados, 1 = nós atacamos
    static constexpr int THREAT_FEATURES = THREAT_VICTIMS * 64 * THREAT_DIRS;  // 640
    inline int makeThreat(int dir, int victim, int sq) { return (dir * THREAT_VICTIMS + victim) * 64 + sq; }

    // Recolhe as features de ameaça ATIVAS para uma perspetiva (persp: 0=branca, 1=preta).
    //   Para cada peça {P,N,B,R,Q}:
    //     - se é NOSSA e está atacada pelo adversário → dir 0 (defesa)
    //     - se é DELES e nós a atacamos              → dir 1 (iniciativa)
    //   A casa é vista na perspetiva (branca normal, preta espelhada ^56), igual às
    //   features peça-casa. 'out' recebe os índices [0,640); devolve quantos.
    // 🦅 OTIMIZADO (profile: attackersTo era 5.2M chamadas = 57×/eval = ~25% do tempo):
    //   mapas "casas atacadas por cada cor" calculados 1× por eval + AND com as peças.
    inline void computeAttackMaps(const Board& board, Bitboard attacked[2])
    {
        const Bitboard occ = board.allPieces();
        for (int c = 0; c < 2; ++c)
        {
            const Color col = static_cast<Color>(c);
            Bitboard a = (c == 0)
                ? attacks::pawnAttacks<Color::WHITE>(board.pieces(col, PieceType::PAWN))
                : attacks::pawnAttacks<Color::BLACK>(board.pieces(col, PieceType::PAWN));
            Bitboard kn = board.pieces(col, PieceType::KNIGHT);
            while (kn.any()) a |= attacks::knightAttacks(kn.poplsb());
            Bitboard bq = board.pieces(col, PieceType::BISHOP) | board.pieces(col, PieceType::QUEEN);
            while (bq.any()) a |= attacks::bishopAttacks(bq.poplsb(), occ);
            Bitboard rq = board.pieces(col, PieceType::ROOK) | board.pieces(col, PieceType::QUEEN);
            while (rq.any()) a |= attacks::rookAttacks(rq.poplsb(), occ);
            a |= attacks::kingAttacks(board.kingSq(col));
            attacked[c] = a;
        }
    }

    inline int gatherThreats(const Board& board, int persp, int* out, const Bitboard attacked[2])
    {
        const Color me   = (persp == 0) ? Color::WHITE : Color::BLACK;
        const Color them = ~me;
        const Bitboard atkThem = attacked[static_cast<int>(them)];
        const Bitboard atkMe   = attacked[static_cast<int>(me)];
        int n = 0;
        const PieceType vt[5] = { PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                                  PieceType::ROOK, PieceType::QUEEN };
        for (int v = 0; v < 5; ++v)
        {
            Bitboard mine = board.pieces(me, vt[v]) & atkThem;
            while (mine.any())
            {
                Square sq = mine.poplsb();
                int sval = (persp == 0) ? sq.value() : (sq.value() ^ 56);
                out[n++] = makeThreat(0, v, sval);
            }
            Bitboard theirs = board.pieces(them, vt[v]) & atkMe;
            while (theirs.any())
            {
                Square sq = theirs.poplsb();
                int sval = (persp == 0) ? sq.value() : (sq.value() ^ 56);
                out[n++] = makeThreat(1, v, sval);
            }
        }
        return n;
    }

    // ═══ 🦅 V10 FULL THREATS ═══════════════════════════════════════════════
    // Mapas de ataque POR TIPO: attackedBy[cor][tipo]. 12 mapas, 1× por eval, servem as 2 persp.
    inline void computeAttackMapsByType(const Board& board, Bitboard attackedBy[2][6])
    {
        const Bitboard occ = board.allPieces();
        for (int c = 0; c < 2; ++c)
        {
            const Color col = static_cast<Color>(c);
            attackedBy[c][0] = (c == 0)
                ? attacks::pawnAttacks<Color::WHITE>(board.pieces(col, PieceType::PAWN))
                : attacks::pawnAttacks<Color::BLACK>(board.pieces(col, PieceType::PAWN));
            Bitboard bb = board.pieces(col, PieceType::KNIGHT); Bitboard a(0);
            while (bb.any()) a |= attacks::knightAttacks(bb.poplsb());
            attackedBy[c][1] = a;
            bb = board.pieces(col, PieceType::BISHOP); a = Bitboard(0);
            while (bb.any()) a |= attacks::bishopAttacks(bb.poplsb(), occ);
            attackedBy[c][2] = a;
            bb = board.pieces(col, PieceType::ROOK); a = Bitboard(0);
            while (bb.any()) a |= attacks::rookAttacks(bb.poplsb(), occ);
            attackedBy[c][3] = a;
            bb = board.pieces(col, PieceType::QUEEN); a = Bitboard(0);
            while (bb.any()) { Square qs = bb.poplsb();
                a |= attacks::bishopAttacks(qs, occ) | attacks::rookAttacks(qs, occ); }
            attackedBy[c][4] = a;
            attackedBy[c][5] = attacks::kingAttacks(board.kingSq(col));
        }
    }

    // gatherThreatsFull — espelho EXATO do gather_threats_full (Rust). Ver contrato no topo.
    inline int gatherThreatsFull(const Board& board, int persp, int* out,
                                 const Bitboard attackedBy[2][6])
    {
        const int meI = persp;          // persp 0 = brancas, 1 = pretas
        const int themI = meI ^ 1;
        int n = 0;
        for (int vc = 0; vc < 2; ++vc)             // cor da vítima: 0 = minha, 1 = deles
        {
            const int vColI = (vc == 0) ? meI : themI;
            const Color vCol = static_cast<Color>(vColI);
            for (int vt = 0; vt < 6; ++vt)         // tipo da vítima
            {
                Bitboard victims = board.pieces(vCol, static_cast<PieceType>(vt));
                while (victims.any())
                {
                    Square sq = victims.poplsb();
                    const int sqP = (persp == 0) ? sq.value() : (sq.value() ^ 56);
                    const Bitboard sqBB = Bitboard::fromSquare(sq);
                    for (int at = 0; at < 6; ++at) // tipo do atacante
                    {
                        if ((attackedBy[vColI ^ 1][at] & sqBB).any())
                            out[n++] = makeThreatFull(vc, 0, at, vt, sqP);   // rel 0 = ataque
                        if ((attackedBy[vColI][at] & sqBB).any())
                            out[n++] = makeThreatFull(vc, 1, at, vt, sqP);   // rel 1 = defesa
                    }
                }
            }
        }
        return n;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 FINNY TABLES (cache d'accumulateurs par king-bucket, façon Stockfish)
// ═══════════════════════════════════════════════════════════════════════════
//   Au lieu de recalculer un accumulateur (perspective) depuis zéro à chaque
//   éval, on garde un CACHE indexé par (perspective, king-bucket). Chaque entrée
//   stocke l'accumulateur déjà calculé + les bitboards (par couleur×type) qui
//   l'ont produit. Pour évaluer une position :
//     - on prend l'entrée du bucket courant,
//     - on calcule le DIFF de bitboards (ce qui a changé depuis le cache),
//     - on applique add/remove seulement pour les cases qui diffèrent,
//     - on rafraîchit l'entrée du cache avec le nouvel état.
//   La plupart des positions consécutives partagent le même bucket avec peu de
//   différences → le diff est petit → gros gain. Et c'est LOCAL à evaluate()
//   (aucun hook make/unmake), donc SÛR. Une table par thread (recherche //).
//
//   Init "depuis biais" : une entrée fraîche = accumulateur égal aux biais et
//   bitboards à 0 (aucune pièce) ; le premier diff ajoute alors toutes les
//   pièces (équivalent au calcul complet), puis les évals suivantes profitent.
// ═══════════════════════════════════════════════════════════════════════════
struct FinnyEntry
{
    bool init = false;
    alignas(32) int32_t acc[MAX_L1];
    uint64_t bb[2][6];   // bitboards [couleur][type] ayant produit acc
};
// [perspective 0=blanche/1=noire][bucket 0..31]
struct FinnyTable
{
    int L1 = 0;
    FinnyEntry e[2][32];
};
// ⚠️ ~390KB por FinnyTable × 2 (big/small) ≈ 780KB direto na TLS — somado ao
//   tl_plyArr abaixo (~6.3MB), já é demasiado para a TLS estática de uma
//   thread nova (SIGSEGV logo na 1ª evaluate() ao ligar Lazy SMP). Mesmo
//   tratamento: só um ponteiro pequeno na TLS, os dados ficam no heap.
struct FinnyTablePair { FinnyTable t[2]; };
static thread_local std::unique_ptr<FinnyTablePair> tl_finnyHolder;
static inline FinnyTable& finnyFor(int idx) {
    if (!tl_finnyHolder) tl_finnyHolder = std::make_unique<FinnyTablePair>();
    return tl_finnyHolder->t[idx];
}
static thread_local int g_netIdx = 0;             // índice da rede ativa (0=big,1=small) POR THREAD
// 🦅 Parte 3: ponteiro thread_local p/ o slot incremental atual (array napkAcc[2] do EvalState).
//   Declarado AQUI (antes do evaluate) p/ o evaluate o ver. nullptr → fallback (finny).
static thread_local const NapkAccSlot* g_napkCurrentSlot = nullptr;
// 🦅 default false: search.cpp ligou o push/pop ao make/unmakeMove (2026-06-18), ainda por
//   validar/SPRT — UCI NapkIncremental liga explicitamente (true) p/ testar; "setoption" abaixo
//   espelha este default.
static bool g_napkIncremental = false;
static void napkMaterialize(const Board& board, NapkAccSlot* slot);   // fwd (def. mais abaixo)
static void napkMaterializeThreats(const Board& board, NapkAccSlot* slot);   // 🦅 s29 fwd (threats incrementais)
#define tl_finny finnyFor(g_netIdx)

// Calcule l'accumulateur d'UNE perspective dans `out`, en partant du cache
// `fe` du bucket `bkt` : applique le diff de bitboards. persp=0 → blanche
// (kind = pieceKindW, case telle quelle), persp=1 → noire (pieceKindB, sq^56).
static void finnyResolve(FinnyEntry& fe, const Board& board, int L1,
                         int persp, int bkt, int32_t* out)
{
    if (!fe.init)
    {
        for (int i = 0; i < L1; ++i) fe.acc[i] = g_net.accBias[i];
        for (int c = 0; c < 2; ++c) for (int t = 0; t < 6; ++t) fe.bb[c][t] = 0;
        fe.init = true;
    }

    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
        {
            uint64_t oldbb = fe.bb[c][t];
            uint64_t newbb = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t)).value();
            if (oldbb == newbb) continue;

            int pc = engCode(t, c);
            int kind = (persp == 0) ? pieceKindW(pc) : pieceKindB(pc);
            if (kind != -1)
            {
                uint64_t removed = oldbb & ~newbb;
                uint64_t added   = newbb & ~oldbb;
                uint64_t r = removed;
                while (r) {
                    int sq = std::countr_zero(r); r &= r - 1;
                    int fsq = (persp == 0) ? sq : (sq ^ 56);
                    const int16_t* w = &g_net.accWeight[(size_t)makeFeat(bkt, kind, fsq) * L1];
                    for (int i = 0; i < L1; ++i) fe.acc[i] -= w[i];
                }
                uint64_t ad = added;
                while (ad) {
                    int sq = std::countr_zero(ad); ad &= ad - 1;
                    int fsq = (persp == 0) ? sq : (sq ^ 56);
                    const int16_t* w = &g_net.accWeight[(size_t)makeFeat(bkt, kind, fsq) * L1];
                    for (int i = 0; i < L1; ++i) fe.acc[i] += w[i];
                }
            }
            fe.bb[c][t] = newbb;
        }

    for (int i = 0; i < L1; ++i) out[i] = fe.acc[i];
}

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 PILE INCRÉMENTALE PAR PLY (étape B — le vrai incrémental façon Stockfish)
// ═══════════════════════════════════════════════════════════════════════════
//   Une pile d'accumulateurs indexée par gamePly. Chaque entrée chaîne sur la
//   précédente : entre deux ply consécutifs, seul le coup joué change (2-4
//   features). On calcule donc pile[ply] = pile[ply-1] + diff(1 coup) → ultra
//   rapide. Le diff se fait par comparaison de bitboards (capture roque,
//   promotion, en-passant nativement).
//
//   KING-BUCKET : si le roi d'une perspective a changé de bucket entre ply-1 et
//   ply, TOUTES ses features changent → on ne peut pas chaîner cette perspective
//   → on retombe sur finnyResolve (cache par bucket) pour CETTE perspective.
//   L'autre perspective peut quand même chaîner. C'est exactement la logique SF.
//
//   Validité : chaque entrée mémorise le ply, les buckets et les bitboards. Si
//   pile[ply-1] ne correspond pas (saut d'arbre, thread, désynchro) → finny.
//   verifyStack/verifyFinny prouvent l'égalité avec le calcul complet.
// ═══════════════════════════════════════════════════════════════════════════
struct PlyAcc
{
    bool validW = false, validB = false;  // perspective blanche/noire à jour ?
    int  ply = -1;
    int  bucketW = -1, bucketB = -1;
    alignas(32) int32_t accW[MAX_L1];
    alignas(32) int32_t accB[MAX_L1];
    uint64_t bb[2][6];   // bitboards de la position à ce ply
};
struct PlyStack
{
    int L1 = 0;
    PlyAcc s[260];       // MAX_PLY large (gamePly peut monter en recherche)
};
// ⚠️ ~12.4KB por PlyAcc × 260 plies × 2 redes ≈ 6.3MB — o maior bloco TLS do
//   ficheiro, e o que efetivamente rebentou ao ligar Lazy SMP (é o caminho
//   plyResolve, usado sempre que NapkIncremental está OFF, o default). Mesmo
//   tratamento: ponteiro pequeno na TLS, ~6.3MB no heap por thread.
struct PlyStackPair { PlyStack p[2]; };
static thread_local std::unique_ptr<PlyStackPair> tl_plyHolder;
static inline PlyStack& plyFor(int idx) {
    if (!tl_plyHolder) tl_plyHolder = std::make_unique<PlyStackPair>();
    return tl_plyHolder->p[idx];
}
#define tl_ply plyFor(g_netIdx)

// Applique le diff de bitboards (old→new) sur une perspective, en partant d'un
// accu source `src` vers `dst`. persp 0=blanche, 1=noire.
static inline void applyDiff(const int32_t* src, int32_t* dst, int L1, int persp, int bkt,
                             const uint64_t oldbb[2][6], const Board& board)
{
    for (int i = 0; i < L1; ++i) dst[i] = src[i];
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
        {
            uint64_t ob = oldbb[c][t];
            uint64_t nb = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t)).value();
            if (ob == nb) continue;
            int pc = engCode(t, c);
            int kind = (persp == 0) ? pieceKindW(pc) : pieceKindB(pc);
            if (kind == -1) continue;
            uint64_t removed = ob & ~nb, added = nb & ~ob;
            uint64_t r = removed;
            while (r) { int sq = std::countr_zero(r); r &= r - 1;
                int fsq = (persp == 0) ? sq : (sq ^ 56);
                const int16_t* w = &g_net.accWeight[(size_t)makeFeat(bkt, kind, fsq) * L1];
                for (int i = 0; i < L1; ++i) dst[i] -= w[i];
            }
            uint64_t ad = added;
            while (ad) { int sq = std::countr_zero(ad); ad &= ad - 1;
                int fsq = (persp == 0) ? sq : (sq ^ 56);
                const int16_t* w = &g_net.accWeight[(size_t)makeFeat(bkt, kind, fsq) * L1];
                for (int i = 0; i < L1; ++i) dst[i] += w[i];
            }
        }
}

// Résout les accumulateurs de la position courante via la PILE incrémentale.
// Remplit accWout/accBout. Utilise pile[ply-1] si chaînable, sinon finny.
static void plyResolve(const Board& board, int L1, int b_w, int b_b,
                       int32_t* accWout, int32_t* accBout)
{
    if (tl_ply.L1 != L1)
    {
        tl_ply.L1 = L1;
        for (auto& e : tl_ply.s) { e.validW = e.validB = false; e.ply = -1; }
    }

    int ply = board.gamePly();
    if (ply < 0 || ply >= 260)   // hors borne → finny direct
    {
        finnyResolve(tl_finny.e[0][b_w], board, L1, 0, b_w, accWout);
        finnyResolve(tl_finny.e[1][b_b], board, L1, 1, b_b, accBout);
        return;
    }

    PlyAcc& cur = tl_ply.s[ply];
    // Entrée précédente chaînable ? (ply-1 valide et correspond)
    bool canChain = (ply > 0 && tl_ply.s[ply-1].ply == ply-1);
    PlyAcc& prev = tl_ply.s[(ply > 0) ? ply-1 : 0];

    // Perspective BLANCHE
    if (canChain && prev.validW && prev.bucketW == b_w)
    {
        applyDiff(prev.accW, cur.accW, L1, 0, b_w, prev.bb, board);
    }
    else
    {
        finnyResolve(tl_finny.e[0][b_w], board, L1, 0, b_w, cur.accW);
    }
    // Perspective NOIRE
    if (canChain && prev.validB && prev.bucketB == b_b)
    {
        applyDiff(prev.accB, cur.accB, L1, 1, b_b, prev.bb, board);
    }
    else
    {
        finnyResolve(tl_finny.e[1][b_b], board, L1, 1, b_b, cur.accB);
    }

    // Mémoriser l'état de ce ply pour chaîner le suivant
    cur.ply = ply;
    cur.bucketW = b_w; cur.bucketB = b_b;
    cur.validW = cur.validB = true;
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
            cur.bb[c][t] = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t)).value();

    for (int i = 0; i < L1; ++i) { accWout[i] = cur.accW[i]; accBout[i] = cur.accB[i]; }
}

// 🦅 VÉRIFICATION : compare l'accu finny au calcul complet (0 = parfait).
int verifyFinny(const Board& board)
{
    if (!g_net.loaded) return -1;
    const int L1 = g_net.L1;
    int kw = board.kingSq(Color::WHITE).value();
    int kb = board.kingSq(Color::BLACK).value();
    int b_w = BUCKET_MAP[kw];
    int b_b = BUCKET_MAP[kb ^ 56];

    if (tl_finny.L1 != L1) {
        tl_finny.L1 = L1;
        for (int p = 0; p < 2; ++p) for (int b = 0; b < 32; ++b) tl_finny.e[p][b].init = false;
    }
    alignas(32) int32_t fW[MAX_L1], fB[MAX_L1];
    // Teste le VRAI chemin de l'éval (pile incrémentale + finny en filet).
    plyResolve(board, L1, b_w, b_b, fW, fB);

    alignas(32) int32_t cW[MAX_L1], cB[MAX_L1];
    for (int i = 0; i < L1; ++i) { cW[i] = g_net.accBias[i]; cB[i] = g_net.accBias[i]; }
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
        {
            Bitboard bb = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t));
            while (bb.any())
            {
                int sq = bb.poplsb().value();
                int pc = engCode(t, c);
                int kWk = pieceKindW(pc);
                if (kWk != -1) {
                    const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_w, kWk, sq) * L1];
                    for (int i = 0; i < L1; ++i) cW[i] += w[i];
                }
                int kBk = pieceKindB(pc);
                if (kBk != -1) {
                    const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_b, kBk, sq ^ 56) * L1];
                    for (int i = 0; i < L1; ++i) cB[i] += w[i];
                }
            }
        }
    int diff = 0;
    for (int i = 0; i < L1; ++i) { if (fW[i] != cW[i]) ++diff; if (fB[i] != cB[i]) ++diff; }
    return diff;
}

// 🦅 SISTEMA DUAL (façon Stockfish) : seleção da REDE ativa no evaluate.
//   g_useDual=false → usa sempre a big (comportamento normal, uma rede).
//   g_useDual=true  → o combine escolhe big (precisa) ou small (rápida) por margem.
// A seleção é feita via setActiveNet() ANTES de cada evaluate (define g_netPtr+g_netIdx).
static bool g_useDual = false;   // ativado por setoption DualNet (e só se ambas carregadas)

// 🦅 V9.6 — override runtime dos threats (setoption UseThreats). Default ON.
//   Permite A/B na MESMA rede: ON = com threats, OFF = ignora o bloco threat.
static bool g_threatsEnabled = true;
void setThreatsEnabled(bool on) { g_threatsEnabled = on; }
bool threatsEnabled() { return g_threatsEnabled; }

void setDualEnabled(bool on) { g_useDual = on && g_dualLoaded; }
int bigL1()   { return g_netBig.L1; }    // 🦅 tamanho L1 da big (ex 256)
int smallL1() { return g_netSmall.L1; }  // 🦅 tamanho L1 da small (ex 128)

// 🦅 s29: info das redes ativas A PEDIDO (estilo Stockfish — silêncio no arranque, fala se
//   perguntado via `version`/`netstats`). Uma rede por linha: L1, flags, threats, chaos, bytes.
static void printOneNet(const char* etiqueta, const Network& n)
{
    if (!n.loaded) { std::printf("info string %s: (nao carregada)\n", etiqueta); return; }
    const char* thr = !n.hasThreats ? "sem threats"
                    : n.fullThreats ? "threats FULL V10 (9216)" : "threats 640";
    std::printf("info string %s: L1=%d v=%u flags=%u | %s | chaos=%s | %.1f KB\n",
                etiqueta, n.L1, n.version, n.flags, thr,
                n.hasChaos ? "sim" : "nao", n.bytes / 1024.0);
}
void printNetInfo()
{
    printOneNet("BIG ", g_netBig);
    printOneNet("SMALL", g_netSmall);
}

// 🦅 DEBUG : conta as features de ameaça ativas (soma das 2 perspetivas). Só para testes.
int debugCountThreats(const Board& board)
{
    int thr[THREAT_FEATURES];
    Bitboard attacked[2];
    computeAttackMaps(board, attacked);
    int n = gatherThreats(board, 0, thr, attacked);
    n += gatherThreats(board, 1, thr, attacked);
    return n;
}

void debugPrintThreats(const Board& board)
{
    int thr[512];
    Bitboard attackedBy[2][6];
    computeAttackMapsByType(board, attackedBy);
    for (int p = 0; p < 2; ++p) {
        int n = gatherThreatsFull(board, p, thr, attackedBy);
        std::sort(thr, thr + n);
        std::printf("persp%d:", p);
        for (int k = 0; k < n; ++k) std::printf(" %d", thr[k]);
        std::printf("\n");
    }
}

bool dualEnabled() { return g_useDual; }

// 🦅 Contadores de uso por rede (Pergunta 1 do Maréchal: saber quem decidiu).
//   Reset em resetNetStats() (chamado no ucinewgame). atomic para o Lazy SMP.
static std::atomic<uint64_t> g_useBigCount{0};
static std::atomic<uint64_t> g_useSmallCount{0};
void resetNetStats() { g_useBigCount = 0; g_useSmallCount = 0; }
void getNetStats(uint64_t& big, uint64_t& small) { big = g_useBigCount.load(); small = g_useSmallCount.load(); }
// 🦅 PureNNUE: incrementa o contador da head usada SEM trocar a rede ativa (lá é sempre a big
//   com heads diferentes). small=true → conta "small" (bullet/small head); senão "big".
void bumpHeadStat(bool small) {
    if (small) g_useSmallCount.fetch_add(1, std::memory_order_relaxed);
    else       g_useBigCount.fetch_add(1, std::memory_order_relaxed);
}

// Escolhe a rede ativa: 0=big (precisa), 1=small (rápida). Só troca se dual ativo
// e a small estiver carregada; senão fica na big.
void setActiveNet(int idx)
{
    if (g_useDual && idx == 1 && g_netSmall.loaded) { g_netPtr = &g_netSmall; g_netIdx = 1; g_useSmallCount.fetch_add(1, std::memory_order_relaxed); }
    else                                            { g_netPtr = &g_netBig;   g_netIdx = 0; g_useBigCount.fetch_add(1, std::memory_order_relaxed); }
}

// 🦅 Igual ao setActiveNet mas SEM mexer nos contadores (p/ o PureNNUE, que conta a head à parte
//   com bumpHeadStat — senão a big era contada 2×).
void setActiveNetNoCount(int idx)
{
    if (g_useDual && idx == 1 && g_netSmall.loaded) { g_netPtr = &g_netSmall; g_netIdx = 1; }
    else                                            { g_netPtr = &g_netBig;   g_netIdx = 0; }
}

int activeNetIdx() { return g_netIdx; }   // 🦅 índice da rede ativa (0=big, 1=small)

// 🦅 Cabeças (heads) dentro de cada rede: HEAD_BULLET(16)/HEAD_SMALL(32)/HEAD_BIG(L1).
//   (enum HeadSel declarado no .h.) A ideia do Maréchal: usar a bullet/small nas folhas
//   (NNUE pura, rápida, SEM HCE) e a big perto da raiz.

int evaluate(const Board& board) { return evaluate(board, HEAD_BIG); }

// 🦅 NPS-fix (s29): soma de uma row de threat (int16[L1]) ao acumulador (int32[L1]).
//   Era um loop ESCALAR — ~70 threats × 2 persp × L1 por eval = ~80% do tempo com V10
//   (medido: threats ON 49k nps vs OFF 234k). AVX2: 16 colunas por iteração.
static inline void addThreatRow(int32_t* acc, const int16_t* w, int L1)
{
#if defined(__AVX2__)
    for (int i = 0; i < L1; i += 16) {
        const __m256i wv = _mm256_loadu_si256((const __m256i*)&w[i]);
        const __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(wv));
        const __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(wv, 1));
        _mm256_storeu_si256((__m256i*)&acc[i],
            _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)&acc[i]), lo));
        _mm256_storeu_si256((__m256i*)&acc[i + 8],
            _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)&acc[i + 8]), hi));
    }
#else
    for (int i = 0; i < L1; ++i) acc[i] += (int32_t)w[i];
#endif
}

static inline void addThreatRowI16(int16_t* acc, const int16_t* w, int L1)
{
#if defined(__AVX2__)
    for (int i = 0; i < L1; i += 16)
        _mm256_storeu_si256((__m256i*)&acc[i],
            _mm256_add_epi16(_mm256_loadu_si256((const __m256i*)&acc[i]),
                             _mm256_loadu_si256((const __m256i*)&w[i])));
#else
    for (int i = 0; i < L1; ++i) acc[i] += w[i];
#endif
}

// 🦅 s29 — SUBTRAÇÃO de linha (p/ os threats que SAÍRAM no incremental).
static inline void subThreatRowI16(int16_t* acc, const int16_t* w, int L1)
{
#if defined(__AVX2__)
    for (int i = 0; i < L1; i += 16)
        _mm256_storeu_si256((__m256i*)&acc[i],
            _mm256_sub_epi16(_mm256_loadu_si256((const __m256i*)&acc[i]),
                             _mm256_loadu_si256((const __m256i*)&w[i])));
#else
    for (int i = 0; i < L1; ++i) acc[i] -= w[i];
#endif
}

// 🦅 s29 — recomputa a lista de threats de uma perspetiva, ORDENADA (p/ o diff incremental).
//   Reusa gatherThreatsFull (barato, ~2% do custo) e ordena. out[] tem de ter NAPK_MAX_THR.
static inline int threatListSorted(const Board& board, int persp, int* out,
                                   const Bitboard attackedBy[2][6])
{
    int n = gatherThreatsFull(board, persp, out, attackedBy);
    std::sort(out, out + n);
    return n;
}

// 🦅 s29 — DIFF ordenado: dadas duas listas ordenadas (pai, filho), produz adds (no filho, não no
//   pai) e rems (no pai, não no filho). É um merge-walk O(n). Devolve nº de adds e rems.
static inline void threatDiff(const int* parent, int np, const int* child, int nc,
                              int* adds, int& nAdds, int* rems, int& nRems)
{
    nAdds = 0; nRems = 0;
    int i = 0, j = 0;
    while (i < np && j < nc)
    {
        if (parent[i] == child[j]) { ++i; ++j; }
        else if (parent[i] < child[j]) { rems[nRems++] = parent[i++]; }   // no pai, não no filho
        else { adds[nAdds++] = child[j++]; }                              // no filho, não no pai
    }
    while (i < np) rems[nRems++] = parent[i++];
    while (j < nc) adds[nAdds++] = child[j++];
}

// 🦅 s29 — constrói o thrAcc de uma perspetiva DO ZERO (modo completo, a referência de validação).
//   Soma as linhas de todos os threats da lista. Usado na raiz/refresh e p/ validar o incremental.
static inline void threatAccFull(int16_t* thrAcc, const int* list, int n, int L1)
{
    for (int i = 0; i < L1; ++i) thrAcc[i] = 0;
    for (int k = 0; k < n; ++k)
        addThreatRowI16(thrAcc, &g_net.threatWeight[(size_t)list[k] * L1], L1);
}

// 🦅 s29 — constrói o thrAcc INCREMENTALMENTE a partir do pai: copia o acc do pai e aplica só o
//   delta (adds somam, rems subtraem). É o ganho: ~4 linhas em vez de ~68.
static inline void threatAccIncremental(int16_t* dst, const int16_t* parentAcc,
                                        const int* adds, int nAdds,
                                        const int* rems, int nRems, int L1)
{
    std::memcpy(dst, parentAcc, sizeof(int16_t) * L1);
    for (int k = 0; k < nAdds; ++k)
        addThreatRowI16(dst, &g_net.threatWeight[(size_t)adds[k] * L1], L1);
    for (int k = 0; k < nRems; ++k)
        subThreatRowI16(dst, &g_net.threatWeight[(size_t)rems[k] * L1], L1);
}

// 🦅 s29 — VALIDAÇÃO bit-a-bit (modo NAPK_VALIDATE): compara o thrAcc incremental com o completo.
//   Se diferem, imprime e aborta — apanha qualquer erro do incremental ANTES de corromper jogos.
//   Off em produção (custo zero). Devolve true se OK.
static inline bool threatAccValidate(const int16_t* inc, const int* list, int n, int L1,
                                     const char* tag)
{
    alignas(32) int16_t ref[NAPK_MAX_L1];
    threatAccFull(ref, list, n, L1);
    for (int i = 0; i < L1; ++i)
        if (inc[i] != ref[i])
        {
            fprintf(stderr, "[NAPK_VALIDATE] threat acc MISMATCH (%s) @i=%d inc=%d ref=%d n=%d\n",
                    tag, i, inc[i], ref[i], n);
            return false;
        }
    return true;
}

// 🦅 s29 — SELF-TEST do incremental de threats. Dado pai (board p/ a posição anterior) e filho
//   (board atual), constrói o thrAcc do filho de DUAS formas — completa (referência) e incremental
//   (acc do pai + diff) — e compara bit-a-bit. Devolve true se idênticos. Usado pelo comando UCI
//   'threattest' p/ provar a correção SEM ligar à busca (zero risco p/ jogos).
bool napkThreatSelfTest(const Board& parent, const Board& child, int L1, bool verbose)
{
    Bitboard amP[2][6], amC[2][6];
    computeAttackMapsByType(parent, amP);
    computeAttackMapsByType(child, amC);
    int lp[512], lc[512];
    bool allOk = true;
    for (int persp = 0; persp < 2; ++persp)
    {
        int np = threatListSorted(parent, persp, lp, amP);
        int nc = threatListSorted(child,  persp, lc, amC);
        // referência: acc completo do filho
        alignas(32) int16_t accFull[NAPK_MAX_L1];
        threatAccFull(accFull, lc, nc, L1);
        // incremental: acc completo do PAI + diff(pai,filho)
        alignas(32) int16_t accParent[NAPK_MAX_L1];
        threatAccFull(accParent, lp, np, L1);
        int adds[512], rems[512]; int nA, nR;
        threatDiff(lp, np, lc, nc, adds, nA, rems, nR);
        alignas(32) int16_t accInc[NAPK_MAX_L1];
        threatAccIncremental(accInc, accParent, adds, nA, rems, nR, L1);
        bool ok = threatAccValidate(accInc, lc, nc, L1, persp == 0 ? "W" : "B");
        if (verbose)
            std::printf("persp%d: pai=%d filho=%d  +%d -%d mudaram  → %s\n",
                        persp, np, nc, nA, nR, ok ? "OK (bit-a-bit)" : "MISMATCH!");
        allOk = allOk && ok;
    }
    return allOk;
}
static inline void fuseI16intoI32(int32_t* acc32, const int16_t* acc16, int L1)
{
#if defined(__AVX2__)
    for (int i = 0; i < L1; i += 16) {
        const __m256i v = _mm256_loadu_si256((const __m256i*)&acc16[i]);
        _mm256_storeu_si256((__m256i*)&acc32[i],
            _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)&acc32[i]),
                             _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))));
        _mm256_storeu_si256((__m256i*)&acc32[i + 8],
            _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)&acc32[i + 8]),
                             _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))));
    }
#else
    for (int i = 0; i < L1; ++i) acc32[i] += (int32_t)acc16[i];
#endif
}



// 🦅 Computa só a parte ESPECÍFICA de uma cabeça (densas l1→l2→l3) sobre um
//   `concat`/`bucket` já prontos (a parte cara — acumulador+threats — é
//   PARTILHADA pelas 3 cabeças, calculada uma só vez por quem chama). Extraído
//   de evaluate() sem mudar nada da matemática/SIMD, só para o reaproveitar em
//   evaluateAllHeads() (medir convergência bullet/small/big sem pagar 3×
//   acumulador). `concat` não é const por causa do hack de debug ZERO_CONCAT,
//   que evaluate() já aplica antes de chamar — chamadas múltiplas (uma por
//   cabeça) sobre o MESMO concat não voltam a zerar nada.
static float runHead(const Head& h, int bucket, const uint8_t* concat, int l1_in)
{
    int l1_out = h.l1_out, l2_out = h.l2_out;
    float s1 = h.l1_scale[bucket], s2 = h.l2_scale[bucket], s3 = h.l3_scale[bucket];
    float dequant1 = s1 / 127.0f;
    float dequant2 = s2;
    float dequant3 = s3;

    alignas(32) float a1[MAX_L1];
    const int8_t*  l1w_base = h.l1_w[bucket].data();
    const float*   l1b_base = h.l1_b[bucket].data();
    for (int o = 0; o < l1_out; ++o)
    {
        const int8_t* w = &l1w_base[(size_t)o * l1_in];
        int64_t sum = 0;
        int i = 0;

        #if defined(__AVX2__)
        __m256i sum_v = _mm256_setzero_si256();
        for (; i <= l1_in - 32; i += 32) {
            __m256i c_v = _mm256_loadu_si256((const __m256i*)&concat[i]);
            __m256i w_v = _mm256_loadu_si256((const __m256i*)&w[i]);
            #if defined(__AVX512VNNI__) || defined(__AVXVNNI__)
            sum_v = _mm256_dpbusd_epi32(sum_v, c_v, w_v);
            #else
            __m256i madd = _mm256_maddubs_epi16(c_v, w_v);
            __m256i low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(madd));
            __m256i high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(madd, 1));
            sum_v = _mm256_add_epi32(sum_v, low);
            sum_v = _mm256_add_epi32(sum_v, high);
            #endif
        }
        int32_t buffer[8];
        _mm256_storeu_si256((__m256i*)buffer, sum_v);
        sum += buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
        #elif defined(__ARM_NEON)
        int32x4_t sum_v = vdupq_n_s32(0);
        for (; i <= l1_in - 16; i += 16) {
            uint8x16_t c_v = vld1q_u8(&concat[i]);
            int8x16_t w_v = vld1q_s8(&w[i]);
            int16x8_t low16 = vmovl_s8(vget_low_s8(w_v));
            uint16x8_t c_low16 = vmovl_u8(vget_low_u8(c_v));
            int16x8_t high16 = vmovl_s8(vget_high_s8(w_v));
            uint16x8_t c_high16 = vmovl_u8(vget_high_u8(c_v));
            sum_v = vmlal_s16(sum_v, vget_low_s16(low16), vreinterpret_s16_u16(vget_low_u16(c_low16)));
            sum_v = vmlal_s16(sum_v, vget_high_s16(low16), vreinterpret_s16_u16(vget_high_u16(c_low16)));
            sum_v = vmlal_s16(sum_v, vget_low_s16(high16), vreinterpret_s16_u16(vget_low_u16(c_high16)));
            sum_v = vmlal_s16(sum_v, vget_high_s16(high16), vreinterpret_s16_u16(vget_high_u16(c_high16)));
        }
        sum += vgetq_lane_s32(sum_v, 0) + vgetq_lane_s32(sum_v, 1) + vgetq_lane_s32(sum_v, 2) + vgetq_lane_s32(sum_v, 3);
        #endif

        for (; i < l1_in; ++i) sum += (int)concat[i] * (int)w[i];
        a1[o] = std::clamp(l1b_base[o] + (float)sum * dequant1, 0.0f, 1.0f);
    }

    alignas(32) float a2[256];
    for (int o = 0; o < l2_out; ++o)
    {
        const int8_t* w = &h.l2_w[bucket][(size_t)o * l1_out];
        float dot = 0.0f;
        int i = 0;
        #if defined(__AVX2__)
        __m256 dot_v = _mm256_setzero_ps();
        for (; i <= l1_out - 8; i += 8) {
            __m128i w_bytes = _mm_loadl_epi64((const __m128i*)&w[i]);
            __m256i w_ints = _mm256_cvtepi8_epi32(w_bytes);
            __m256 w_floats = _mm256_cvtepi32_ps(w_ints);
            __m256 a1_v = _mm256_loadu_ps(&a1[i]);
            dot_v = _mm256_fmadd_ps(w_floats, a1_v, dot_v);
        }
        float buf_f[8]; _mm256_storeu_ps(buf_f, dot_v);
        dot += buf_f[0] + buf_f[1] + buf_f[2] + buf_f[3] + buf_f[4] + buf_f[5] + buf_f[6] + buf_f[7];
        #endif
        for (; i < l1_out; ++i) dot += (float)w[i] * a1[i];
        a2[o] = std::clamp(h.l2_b[bucket][o] + dot * dequant2, 0.0f, 1.0f);
    }

    float out = h.l3_b[bucket];
    {
        const int8_t* w = &h.l3_w[bucket][0];
        float dot3 = 0.0f;
        int i = 0;
        #if defined(__AVX2__)
        __m256 dot_v3 = _mm256_setzero_ps();
        for (; i <= l2_out - 8; i += 8) {
            __m128i w_bytes = _mm_loadl_epi64((const __m128i*)&w[i]);
            __m256i w_ints = _mm256_cvtepi8_epi32(w_bytes);
            __m256 w_floats = _mm256_cvtepi32_ps(w_ints);
            __m256 a2_v = _mm256_loadu_ps(&a2[i]);
            dot_v3 = _mm256_fmadd_ps(w_floats, a2_v, dot_v3);
        }
        float buf_f3[8]; _mm256_storeu_ps(buf_f3, dot_v3);
        dot3 += buf_f3[0] + buf_f3[1] + buf_f3[2] + buf_f3[3] + buf_f3[4] + buf_f3[5] + buf_f3[6] + buf_f3[7];
        #endif
        for (; i < l2_out; ++i) dot3 += (float)w[i] * a2[i];
        out += dot3 * dequant3;
    }
    return out;
}

int evaluate(const Board& board, int headIdx)
{
    if (!g_net.loaded) return 0;

    const int L1 = g_net.L1;
    const int stm = static_cast<int>(board.sideToMove());

    int kw = board.kingSq(Color::WHITE).value();
    int kb = board.kingSq(Color::BLACK).value();
    int b_w = BUCKET_MAP[kw];
    int b_b = BUCKET_MAP[kb ^ 56];

    // 🦅 PILE INCRÉMENTALE PAR PLY (étape B) : chaîne sur le ply précédent
    // (diff d'1 coup) si possible, sinon retombe sur les finny tables (par bucket).
    if (tl_finny.L1 != L1)   // (ré)init finny si le réseau a changé de taille
    {
        tl_finny.L1 = L1;
        for (int p = 0; p < 2; ++p) for (int b = 0; b < 32; ++b) tl_finny.e[p][b].init = false;
    }
    alignas(32) int32_t accW[MAX_L1], accB[MAX_L1];
    // 🦅 Parte 3: se há um slot incremental válido (busca com EvalState) → COPIA dele (RÁPIDO,
    //   1 cópia em vez de reconstruir). O slot é o da rede ATIVA (g_netIdx). Fallback: plyResolve
    //   (datagen/'eval' sem EvalState, ou slot inválido).
    NapkAccSlot* slot = (g_napkIncremental && g_napkCurrentSlot)
        ? const_cast<NapkAccSlot*>(&g_napkCurrentSlot[g_netIdx]) : nullptr;
    if (slot && slot->valid && slot->bucketW == b_w && slot->bucketB == b_b) {
        napkMaterialize(board, slot);           // 🦅 LAZY: calcula a cadeia só quando é preciso
        std::memcpy(accW, slot->accW, sizeof(int32_t) * L1);
        std::memcpy(accB, slot->accB, sizeof(int32_t) * L1);
        // 🦅 s29: materializar TAMBÉM os threats do slot (incremental se o pai os tiver). A lista do
        //   pai está guardada no slot (não precisa do board do pai) → o diff só usa listas + board atual.
        if (g_napkIncremental && g_net.fullThreats)
            napkMaterializeThreats(board, slot);
    } else {
        plyResolve(board, L1, b_w, b_b, accW, accB);
    }
    //   ANTES da ClippedReLU (assim passam pela mesma ativação + L1, sem tocar no
    //   caminho crítico). Não-incremental: recalculado a cada eval. Se a rede não tem
    //   pesos de threat (hasThreats=false), salta tudo → eval idêntico ao V9.5.
    //   g_threatsEnabled permite DESLIGAR em runtime (setoption UseThreats false) para
    //   medir o efeito dos threats na MESMA rede (teste A/B sem treinar 2 redes).
    if (g_net.hasThreats && g_threatsEnabled)
    {
        alignas(32) int16_t thrW[MAX_L1]; alignas(32) int16_t thrB[MAX_L1];
        // 🦅 s29: se o slot tem o thrAcc incremental pronto (busca, fullThreats), COPIA-O (rápido:
        //   já só somou as ~4 linhas que mudaram, não as ~68). Senão, recalcula tudo (datagen/eval
        //   sem slot, ou rede 640). É o ganho do Eixo 2.
        bool usedSlotThreats = false;
        if (g_napkIncremental && g_net.fullThreats && slot && slot->thrValid
            && slot->bucketW == b_w && slot->bucketB == b_b)
        {
            std::memcpy(thrW, slot->thrAccW, sizeof(int16_t) * L1);
            std::memcpy(thrB, slot->thrAccB, sizeof(int16_t) * L1);
            usedSlotThreats = true;
        }
        else
            for (int i = 0; i < L1; ++i) { thrW[i] = 0; thrB[i] = 0; }
        if (!usedSlotThreats && g_net.fullThreats)
        {
            int thr[512];
            Bitboard attackedBy[2][6];
            computeAttackMapsByType(board, attackedBy);
            int nW = gatherThreatsFull(board, 0, thr, attackedBy);
            for (int k = 0; k < nW; ++k) {
                if (k + 2 < nW) __builtin_prefetch(&g_net.threatWeight[(size_t)thr[k + 2] * L1]);
                addThreatRowI16(thrW, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
            }
            int nB = gatherThreatsFull(board, 1, thr, attackedBy);
            for (int k = 0; k < nB; ++k) {
                if (k + 2 < nB) __builtin_prefetch(&g_net.threatWeight[(size_t)thr[k + 2] * L1]);
                addThreatRowI16(thrB, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
            }
        }
        else if (!usedSlotThreats)
        {
            int thr[THREAT_FEATURES];
            Bitboard attacked[2];
            computeAttackMaps(board, attacked);
            int nW = gatherThreats(board, 0, thr, attacked);
            for (int k = 0; k < nW; ++k) addThreatRowI16(thrW, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
            int nB = gatherThreats(board, 1, thr, attacked);
            for (int k = 0; k < nB; ++k) addThreatRowI16(thrB, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
        }
        fuseI16intoI32(accW, thrW, L1);
        fuseI16intoI32(accB, thrB, L1);
    }

    // Compteurs matériels (légers) pour le bucket et la HCE.
    int pieceCount = 0;
    int white_queens = 0, black_queens = 0;
    int white_rooks = 0, black_rooks = 0;
    int white_minors = 0, black_minors = 0;
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
        {
            int n = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t)).popcount();
            pieceCount += n;
            if (c == 0) {
                if (t == 4) white_queens += n;
                else if (t == 3) white_rooks += n;
                else if (t == 1 || t == 2) white_minors += n;
            } else {
                if (t == 4) black_queens += n;
                else if (t == 3) black_rooks += n;
                else if (t == 1 || t == 2) black_minors += n;
            }
        }

    int bucket = materialBucket(pieceCount);

    const int32_t* accUs   = (stm == 0) ? accW : accB;
    const int32_t* accThem = (stm == 0) ? accB : accW;
    const float QA = g_net.qa;

    // 🦅 OPTIM MAJEURE : concat en SIMD int (était float scalaire = 1220ns goulot !).
    // clipped relu [0,QA] puis scale [0,127] via mul+shift avec ARRONDI EXACT :
    // round(x*127/QA). Pour QA=255 : (x*32640 + 32768) >> 16 (prouvé == float lround).
    // Fallback scalaire identique pour QA != 255 ou reliquat.
    // 🦅 FIX s29 (core dump multi-thread): concat era [2048] = 2×1024 EXATO p/ L1=1024 → o loop
    //   AVX (blocos de 32) tocava concat[2048] no limite; com L1=1536 (MAX) transbordava já a 1
    //   thread. O multi-thread só EXPUNHA o over-read (stack mais apertada → memória não mapeada
    //   → SIGSEGV). Dimensionar p/ 2×MAX_L1 + 32 de margem AVX. Custa stack mas elimina o crash.
    alignas(32) uint8_t concat[2 * NAPK_MAX_L1 + 32];
    const int QAi = (int)QA;
    {
        // Coefficient de mise à l'échelle arrondie : round(127/QA * 2^16).
        const int scaleMul = (int)std::lround(127.0 / QAi * 65536.0);
        const int32_t* srcs[2] = { accUs, accThem };
        for (int half = 0; half < 2; ++half)
        {
            const int32_t* a = srcs[half];
            uint8_t* d = concat + half * L1;
            int i = 0;
            #if defined(__AVX2__)
            const __m256i zero = _mm256_setzero_si256();
            const __m256i qav  = _mm256_set1_epi32(QAi);
            const __m256i mulv = _mm256_set1_epi32(scaleMul);
            const __m256i half16 = _mm256_set1_epi32(32768);
            for (; i <= L1 - 8; i += 8) {
                __m256i v = _mm256_loadu_si256((const __m256i*)&a[i]);
                v = _mm256_max_epi32(v, zero);           // max(0,acc)
                v = _mm256_min_epi32(v, qav);            // min(QA,acc) → [0,QA]
                v = _mm256_mullo_epi32(v, mulv);         // ×(127/QA*2^16)
                v = _mm256_add_epi32(v, half16);         // +0.5 (arrondi)
                v = _mm256_srli_epi32(v, 16);            // >>16 → [0,127]
                __m128i lo = _mm256_castsi256_si128(v);
                __m128i hi = _mm256_extracti128_si256(v, 1);
                __m128i p16 = _mm_packus_epi32(lo, hi);  // 8 int32 → 8 uint16
                __m128i p8  = _mm_packus_epi16(p16, p16);// → 8 uint8
                _mm_storel_epi64((__m128i*)&d[i], p8);
            }
            #elif defined(__ARM_NEON)
            const int32x4_t zero = vdupq_n_s32(0);
            const int32x4_t qav  = vdupq_n_s32(QAi);
            const int32x4_t mulv = vdupq_n_s32(scaleMul);
            const int32x4_t half16 = vdupq_n_s32(32768);
            for (; i <= L1 - 4; i += 4) {
                int32x4_t v = vld1q_s32(&a[i]);
                v = vmaxq_s32(v, zero);
                v = vminq_s32(v, qav);
                v = vmulq_s32(v, mulv);
                v = vaddq_s32(v, half16);
                v = vshrq_n_s32(v, 16);
                d[i+0]=(uint8_t)vgetq_lane_s32(v,0); d[i+1]=(uint8_t)vgetq_lane_s32(v,1);
                d[i+2]=(uint8_t)vgetq_lane_s32(v,2); d[i+3]=(uint8_t)vgetq_lane_s32(v,3);
            }
            #endif
            for (; i < L1; ++i) {
                int x = a[i]; if (x < 0) x = 0; if (x > QAi) x = QAi;
                d[i] = (uint8_t)((x * scaleMul + 32768) >> 16);
            }
        }
    }

    // 🦅 Escolhe a cabeça conforme headIdx (bullet/small/big). Se a rede não tem as 3
    //   cabeças, ou o índice é inválido, cai na big (seguro). As cabeças bullet(16) e
    //   small(32) são muito mais baratas que a big — ideais para as folhas em modo
    //   PureNNUE (substituem o HCE: NNUE pura mas quase tão rápida).
    const Head& h = (g_net.has3heads && headIdx == HEAD_BULLET) ? g_net.bullet
                  : (g_net.has3heads && headIdx == HEAD_SMALL)  ? g_net.small
                  :                                               g_net.big;
    int l1_in = L1 * 2;
    if (getenv("ZERO_CONCAT")) for(int i=0;i<l1_in;++i) concat[i]=0;

    float out = runHead(h, bucket, concat, l1_in);

    float psqtBias = 0.0f;
    if (!g_net.psqt.empty())
    {
        int64_t psW = 0, psB = 0;
        for (int c = 0; c < 2; ++c)
            for (int t = 0; t < 6; ++t)
            {
                Bitboard bb = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t));
                while (bb.any())
                {
                    int sq = bb.poplsb().value();
                    int pc = engCode(t, c);
                    int kWk = pieceKindW(pc);
                    if (kWk != -1) psW += g_net.psqt[(size_t)makeFeat(b_w, kWk, sq) * MATERIAL_BUCKETS + bucket];
                    int kBk = pieceKindB(pc);
                    if (kBk != -1) psB += g_net.psqt[(size_t)makeFeat(b_b, kBk, sq ^ 56) * MATERIAL_BUCKETS + bucket];
                }
            }
        float psqtUs   = (stm == 0) ? (float)psW : (float)psB;
        float psqtThem = (stm == 0) ? (float)psB : (float)psW;
        psqtBias = ((psqtUs - psqtThem) / g_net.qa) / 2.0f;
    }

    int score = (int)std::lround((out + psqtBias) * OUTPUT_SCALE_CP);

    // 🚨 REGULADOR CIRÚRGICO ANTI-KING WALK
    int whiteKingPenalty = 0;
    if (black_queens == 0 && (black_rooks > 0 || black_minors >= 2)) {
        int rank = kw / 8; // Fila indexada a 0 (0=Fila 1, 1=Fila 2, 2=Fila 3...)
        if (rank >= 2) {   // Se o Rei Branco subir para a Fila 3 ou superior numa transição perigosa
            whiteKingPenalty = 150 + (black_rooks * 50) + (black_minors * 25);
        }
    }

    int blackKingPenalty = 0;
    if (white_queens == 0 && (white_rooks > 0 || white_minors >= 2)) {
        int rank = kb / 8;
        if (rank <= 5) {   // Se o Rei Preto descer para a Fila 6 ou inferior numa transição perigosa
            blackKingPenalty = 150 + (white_rooks * 50) + (white_minors * 25);
        }
    }

    // Aplicação estrita baseada na perspetiva do jogador da vez (stm)
    if (stm == 0) {
        score += (blackKingPenalty - whiteKingPenalty);
    } else {
        score += (whiteKingPenalty - blackKingPenalty);
    }

    return std::clamp(score, -3000, 3000);
}

// 🦅 Resolve o acumulador+threats+concat UMA SÓ VEZ (a parte cara, igual à de
//   evaluate()) e aplica as 3 cabeças (bullet/small/big) sobre o MESMO concat
//   — custo extra é só 3× as densas pequenas (16-32 neurónios), não 3×
//   acumulador. Para medir convergência entre cabeças (UCI "headconverge"),
//   não para ganhar NPS em jogo (esse ganho não existe por esta via — ver
//   a conversa que motivou isto: a parte cara é partilhada de qualquer forma).
void evaluateAllHeads(const Board& board, int& bulletScore, int& smallScore, int& bigScore)
{
    if (!g_net.loaded) { bulletScore = smallScore = bigScore = 0; return; }

    const int L1 = g_net.L1;
    const int stm = static_cast<int>(board.sideToMove());

    int kw = board.kingSq(Color::WHITE).value();
    int kb = board.kingSq(Color::BLACK).value();
    int b_w = BUCKET_MAP[kw];
    int b_b = BUCKET_MAP[kb ^ 56];

    if (tl_finny.L1 != L1) {
        tl_finny.L1 = L1;
        for (int p = 0; p < 2; ++p) for (int b = 0; b < 32; ++b) tl_finny.e[p][b].init = false;
    }
    alignas(32) int32_t accW[MAX_L1], accB[MAX_L1];
    NapkAccSlot* slot = (g_napkIncremental && g_napkCurrentSlot)
        ? const_cast<NapkAccSlot*>(&g_napkCurrentSlot[g_netIdx]) : nullptr;
    if (slot && slot->valid && slot->bucketW == b_w && slot->bucketB == b_b) {
        napkMaterialize(board, slot);
        std::memcpy(accW, slot->accW, sizeof(int32_t) * L1);
        std::memcpy(accB, slot->accB, sizeof(int32_t) * L1);
        if (g_napkIncremental && g_net.fullThreats)
            napkMaterializeThreats(board, slot);
    } else {
        plyResolve(board, L1, b_w, b_b, accW, accB);
    }
    if (g_net.hasThreats && g_threatsEnabled)
    {
        alignas(32) int16_t thrW[MAX_L1]; alignas(32) int16_t thrB[MAX_L1];
        bool usedSlotThreats = false;
        if (g_napkIncremental && g_net.fullThreats && slot && slot->thrValid
            && slot->bucketW == b_w && slot->bucketB == b_b)
        {
            std::memcpy(thrW, slot->thrAccW, sizeof(int16_t) * L1);
            std::memcpy(thrB, slot->thrAccB, sizeof(int16_t) * L1);
            usedSlotThreats = true;
        }
        else
            for (int i = 0; i < L1; ++i) { thrW[i] = 0; thrB[i] = 0; }
        if (!usedSlotThreats && g_net.fullThreats)
        {
            int thr[512];
            Bitboard attackedBy[2][6];
            computeAttackMapsByType(board, attackedBy);
            int nW = gatherThreatsFull(board, 0, thr, attackedBy);
            for (int k = 0; k < nW; ++k) addThreatRowI16(thrW, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
            int nB = gatherThreatsFull(board, 1, thr, attackedBy);
            for (int k = 0; k < nB; ++k) addThreatRowI16(thrB, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
        }
        else if (!usedSlotThreats)
        {
            int thr[THREAT_FEATURES];
            Bitboard attacked[2];
            computeAttackMaps(board, attacked);
            int nW = gatherThreats(board, 0, thr, attacked);
            for (int k = 0; k < nW; ++k) addThreatRowI16(thrW, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
            int nB = gatherThreats(board, 1, thr, attacked);
            for (int k = 0; k < nB; ++k) addThreatRowI16(thrB, &g_net.threatWeight[(size_t)thr[k] * L1], L1);
        }
        fuseI16intoI32(accW, thrW, L1);
        fuseI16intoI32(accB, thrB, L1);
    }

    int pieceCount = 0;
    int white_queens = 0, black_queens = 0, white_rooks = 0, black_rooks = 0;
    int white_minors = 0, black_minors = 0;
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t) {
            int n = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t)).popcount();
            pieceCount += n;
            if (c == 0) {
                if (t == 4) white_queens += n;
                else if (t == 3) white_rooks += n;
                else if (t == 1 || t == 2) white_minors += n;
            } else {
                if (t == 4) black_queens += n;
                else if (t == 3) black_rooks += n;
                else if (t == 1 || t == 2) black_minors += n;
            }
        }
    int bucket = materialBucket(pieceCount);

    const int32_t* accUs   = (stm == 0) ? accW : accB;
    const int32_t* accThem = (stm == 0) ? accB : accW;
    const float QA = g_net.qa;
    alignas(32) uint8_t concat[2 * NAPK_MAX_L1 + 32];
    const int QAi = (int)QA;
    {
        const int scaleMul = (int)std::lround(127.0 / QAi * 65536.0);
        const int32_t* srcs[2] = { accUs, accThem };
        for (int half = 0; half < 2; ++half) {
            const int32_t* a = srcs[half];
            uint8_t* d = concat + half * L1;
            for (int i = 0; i < L1; ++i) {
                int x = a[i]; if (x < 0) x = 0; if (x > QAi) x = QAi;
                d[i] = (uint8_t)((x * scaleMul + 32768) >> 16);
            }
        }
    }
    int l1_in = L1 * 2;

    float psqtBias = 0.0f;
    if (!g_net.psqt.empty()) {
        int64_t psW = 0, psB = 0;
        for (int c = 0; c < 2; ++c)
            for (int t = 0; t < 6; ++t) {
                Bitboard bb = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t));
                while (bb.any()) {
                    int sq = bb.poplsb().value();
                    int pc = engCode(t, c);
                    int kWk = pieceKindW(pc);
                    if (kWk != -1) psW += g_net.psqt[(size_t)makeFeat(b_w, kWk, sq) * MATERIAL_BUCKETS + bucket];
                    int kBk = pieceKindB(pc);
                    if (kBk != -1) psB += g_net.psqt[(size_t)makeFeat(b_b, kBk, sq ^ 56) * MATERIAL_BUCKETS + bucket];
                }
            }
        float psqtUs   = (stm == 0) ? (float)psW : (float)psB;
        float psqtThem = (stm == 0) ? (float)psB : (float)psW;
        psqtBias = ((psqtUs - psqtThem) / g_net.qa) / 2.0f;
    }

    int whiteKingPenalty = 0;
    if (black_queens == 0 && (black_rooks > 0 || black_minors >= 2)) {
        int rank = kw / 8;
        if (rank >= 2) whiteKingPenalty = 150 + (black_rooks * 50) + (black_minors * 25);
    }
    int blackKingPenalty = 0;
    if (white_queens == 0 && (white_rooks > 0 || white_minors >= 2)) {
        int rank = kb / 8;
        if (rank <= 5) blackKingPenalty = 150 + (white_rooks * 50) + (white_minors * 25);
    }

    auto finalize = [&](float out) -> int {
        int score = (int)std::lround((out + psqtBias) * OUTPUT_SCALE_CP);
        if (stm == 0) score += (blackKingPenalty - whiteKingPenalty);
        else          score += (whiteKingPenalty - blackKingPenalty);
        return std::clamp(score, -3000, 3000);
    };

    const Head& hBullet = g_net.has3heads ? g_net.bullet : g_net.big;
    const Head& hSmall  = g_net.has3heads ? g_net.small  : g_net.big;
    bulletScore = finalize(runHead(hBullet, bucket, concat, l1_in));
    smallScore  = finalize(runHead(hSmall,  bucket, concat, l1_in));
    bigScore    = finalize(runHead(g_net.big, bucket, concat, l1_in));
}
static bool loadFromBytes(const std::vector<uint8_t>& buf)
{
    long size = (long)buf.size();
    if (size >= 8 && std::memcmp(buf.data(), "NAPK9LEB", 8) == 0)
    {
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NNUE] Format NapK9 détecté.\n");
        bool ok = loadNapK9(buf);
if (g_nnueVerbose) if (ok) std::fprintf(stderr, "🦅 [NNUE] NapK9 chargé (L1=%d).\n", g_net.L1);
        return ok;
    }
    return false;
}

// 🦅 SISTEMA DUAL : carrega bytes para uma rede ESPECÍFICA (big ou small).
static bool loadIntoNet(Network* target, int idx, const std::vector<uint8_t>& buf)
{
    Network* saved = g_netPtr; int savedIdx = g_netIdx;
    g_netPtr = target; g_netIdx = idx;          // redireciona g_net para o alvo
    bool ok = loadFromBytes(buf);
    g_netPtr = saved; g_netIdx = savedIdx;       // restaura
    return ok;
}

bool load(const std::string& path)
{
    // 🦅 V9.6 FIX: NÃO chamar unload() (que apagava a small e punha g_dualLoaded=false).
    //   O EvalFile recarrega SÓ a big; a small (já carregada pelo auto-load) tem de
    //   sobreviver, senão o CombatMode/DualNet perde o dual quando o bot envia EvalFile
    //   por último. Limpamos apenas a big e preservamos a small.
    bool hadSmall = g_netSmall.loaded;
    g_netBig = Network{};
    g_netPtr = &g_netBig; g_netIdx = 0;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0) { std::fclose(f); return false; }
    std::vector<uint8_t> buf((size_t)size);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) { std::fclose(f); return false; }
    std::fclose(f);
    // Carrega como rede BIG (rede principal/normal). g_netPtr já aponta para big.
    g_netPtr = &g_netBig; g_netIdx = 0;
    bool ok = loadFromBytes(buf);
    // re-sincroniza o estado dual: continua dual se a big carregou E a small sobreviveu.
    g_dualLoaded = ok && g_netBig.loaded && hadSmall && g_netSmall.loaded;
    return ok;
}

// 🦅 Carrega a rede SMALL (a 128) de um ficheiro. Usada além da big.
bool loadSmall(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0) { std::fclose(f); return false; }
    std::vector<uint8_t> buf((size_t)size);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) { std::fclose(f); return false; }
    std::fclose(f);
    bool ok = loadIntoNet(&g_netSmall, 1, buf);
    g_dualLoaded = ok && g_netBig.loaded;
if (g_nnueVerbose) if (ok) std::fprintf(stderr, "🦅 [NNUE] Small net carregada (L1=%d) — sistema DUAL ativo.\n", g_netSmall.L1);
    return ok;
}

// 🦅 Charge la rede EMBUTIDA dans le binaire (façon Stockfish). Aucun fichier
// externe requis. Retourne false si le binaire n'a pas de rede embutida.
bool loadEmbedded()
{
    if (!hasEmbeddedNet()) return false;
    unload();
    const uint8_t* data = embeddedNetData();
    size_t size = embeddedNetSize();
    std::vector<uint8_t> buf(data, data + size);
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NNUE] Rede embutida (%zu octets).\n", size);
    g_netPtr = &g_netBig; g_netIdx = 0;
    return loadFromBytes(buf);
}

// 🦅 Carrega a small net EMBEBIDA (2º símbolo incbin). Sistema dual autónomo.
bool loadSmallEmbedded()
{
    if (!hasEmbeddedSmallNet()) return false;
    const uint8_t* data = embeddedSmallNetData();
    size_t size = embeddedSmallNetSize();
    std::vector<uint8_t> buf(data, data + size);
if (g_nnueVerbose) std::fprintf(stderr, "🦅 [NNUE] Small net embutida (%zu octets).\n", size);
    bool ok = loadIntoNet(&g_netSmall, 1, buf);
    g_dualLoaded = ok && g_netBig.loaded;
    if (ok && g_nnueVerbose) std::fprintf(stderr, "🦅 [NNUE] Sistema DUAL ativo (big L1=%d + small L1=%d).\n",
                          g_netBig.L1, g_netSmall.L1);
    return ok;
}

void unload() { g_netBig = Network{}; g_netSmall = Network{}; g_dualLoaded = false;
                g_netPtr = &g_netBig; g_netIdx = 0; }
bool isLoaded() { return g_netBig.loaded; }
bool dualLoaded() { return g_dualLoaded; }

// ═══════════════════════════════════════════════════════════════════════════
// 🦅 ACUMULADOR INCREMENTAL — Parte 1: stubs (ainda NÃO ligados; corpos nas Partes 2-3).
//   Existem só p/ o build compilar com a estrutura nova. NÃO mudam comportamento: o
//   g_napkCurrentSlot fica nullptr → o evaluate continua a usar o caminho antigo (finny).
// ═══════════════════════════════════════════════════════════════════════════

void napkSetCurrentSlot(const NapkAccSlot* slot) { g_napkCurrentSlot = slot; }
void napkSetIncremental(bool on) { g_napkIncremental = on; }
bool napkIncrementalEnabled() { return g_napkIncremental; }
const NapkAccSlot* napkCurrentSlot() { return g_napkCurrentSlot; }
// 🦅 LAZY UPDATE (estilo Stockfish): o push regista SÓ os deltas e a cadeia (barato, ~50 bytes).
//   O acc só é calculado quando um evaluate o pede (napkMaterialize). Nós onde a busca corta antes
//   da eval (TT hit, beta cutoff cedo) NÃO pagam a cópia de L1.
void napkLazyPush(const Board& board, const NapkAccSlot& src, NapkAccSlot& dst,
                  const NapkDelta* adds, int nAdds, const NapkDelta* rems, int nRems) {
    if (!g_net.loaded) { dst.valid = false; return; }
    int kw = board.kingSq(Color::WHITE).value();
    int kb = board.kingSq(Color::BLACK).value();
    dst.bucketW = BUCKET_MAP[kw];
    dst.bucketB = BUCKET_MAP[kb ^ 56];
    dst.prev = &src;
    dst.needRefresh = (!src.valid) || (dst.bucketW != src.bucketW) || (dst.bucketB != src.bucketB);
    dst.materialized = false;
    dst.thrValid = false;   // 🦅 s29: slot novo → threats por materializar
    dst.valid = true;
    dst.nAdds = nAdds; dst.nRems = nRems;
    for (int i = 0; i < nAdds && i < 2; ++i) dst.adds[i] = adds[i];
    for (int i = 0; i < nRems && i < 2; ++i) dst.rems[i] = rems[i];
}

// Aplica os deltas de `cur` sobre o acc do prev (materializado) → cur->acc. Buckets iguais.
static void napkApplyDeltas(NapkAccSlot* cur) {
    const int L1 = g_net.L1;
    const NapkAccSlot* p = cur->prev;
    std::memcpy(cur->accW, p->accW, sizeof(int32_t) * L1);
    std::memcpy(cur->accB, p->accB, sizeof(int32_t) * L1);
    const int b_w = cur->bucketW, b_b = cur->bucketB;
    for (int a = 0; a < cur->nAdds; ++a) {
        int pc = engCode(cur->adds[a].type, cur->adds[a].color);
        int kw = pieceKindW(pc), kb = pieceKindB(pc);
        if (kw != -1) { const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_w, kw, cur->adds[a].sq) * L1];
                        for (int i = 0; i < L1; ++i) cur->accW[i] += w[i]; }
        if (kb != -1) { const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_b, kb, cur->adds[a].sq ^ 56) * L1];
                        for (int i = 0; i < L1; ++i) cur->accB[i] += w[i]; }
    }
    for (int r = 0; r < cur->nRems; ++r) {
        int pc = engCode(cur->rems[r].type, cur->rems[r].color);
        int kw = pieceKindW(pc), kb = pieceKindB(pc);
        if (kw != -1) { const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_w, kw, cur->rems[r].sq) * L1];
                        for (int i = 0; i < L1; ++i) cur->accW[i] -= w[i]; }
        if (kb != -1) { const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_b, kb, cur->rems[r].sq ^ 56) * L1];
                        for (int i = 0; i < L1; ++i) cur->accB[i] -= w[i]; }
    }
    cur->materialized = true;
}

// 🦅 s29 — materializa o thrAcc (acumulador de threats) de um slot. Se o pai tem thrAcc válido e
//   o bucket não mudou, usa o INCREMENTAL (acc do pai + diff das listas). Senão, calcula do zero.
//   board = posição ATUAL do slot. Opera na rede ativa (g_net). Com NAPK_VALIDATE valida bit-a-bit.
static void napkMaterializeThreats(const Board& board, NapkAccSlot* slot) {
    if (slot->thrValid) return;
    if (!g_net.loaded || !g_net.fullThreats) { slot->thrValid = false; return; }
    const int L1 = g_net.L1;

    // recomputar a lista de threats da posição atual (barato, ~2% do custo)
    Bitboard am[2][6];
    computeAttackMapsByType(board, am);
    int lc_w[NapkAccSlot::NAPK_MAX_THR], lc_b[NapkAccSlot::NAPK_MAX_THR];
    int nc_w = threatListSorted(board, 0, lc_w, am);
    int nc_b = threatListSorted(board, 1, lc_b, am);

    const NapkAccSlot* p = slot->prev;
    bool canInc = p && p->thrValid && !slot->needRefresh
                  && slot->bucketW == (p ? p->bucketW : -2)   // bucket não mudou
                  && slot->bucketB == (p ? p->bucketB : -2);
    if (canInc)
    {
        // incremental: thrAcc = thrAcc_pai + diff(lista_pai, lista_filho)
        int adds[NapkAccSlot::NAPK_MAX_THR], rems[NapkAccSlot::NAPK_MAX_THR]; int nA, nR;
        threatDiff(p->thrListW, p->nThrW, lc_w, nc_w, adds, nA, rems, nR);
        threatAccIncremental(slot->thrAccW, p->thrAccW, adds, nA, rems, nR, L1);
        threatDiff(p->thrListB, p->nThrB, lc_b, nc_b, adds, nA, rems, nR);
        threatAccIncremental(slot->thrAccB, p->thrAccB, adds, nA, rems, nR, L1);
    }
    else
    {
        // do zero (raiz, refresh de bucket, ou pai sem threats)
        threatAccFull(slot->thrAccW, lc_w, nc_w, L1);
        threatAccFull(slot->thrAccB, lc_b, nc_b, L1);
    }

#ifdef NAPK_VALIDATE
    if (!threatAccValidate(slot->thrAccW, lc_w, nc_w, L1, "matW")
        || !threatAccValidate(slot->thrAccB, lc_b, nc_b, L1, "matB"))
    {
        fprintf(stderr, "[NAPK_VALIDATE] napkMaterializeThreats MISMATCH (inc=%d)\n", (int)canInc);
        std::abort();
    }
#endif

    // guardar a lista no slot p/ o filho fazer o diff
    std::memcpy(slot->thrListW, lc_w, sizeof(int) * nc_w); slot->nThrW = nc_w;
    std::memcpy(slot->thrListB, lc_b, sizeof(int) * nc_b); slot->nThrB = nc_b;
    slot->thrValid = true;
}

// 🦅 Materializa o slot p/ o board ATUAL: anda p/ trás até um materializado, aplica os deltas pelo
//   caminho (materializa todos — cache). Se há needRefresh na cadeia → full refresh do atual (raro).
static void napkMaterialize(const Board& board, NapkAccSlot* slot) {
    if (slot->materialized) return;
    NapkAccSlot* chain[256]; int n = 0;
    NapkAccSlot* s = slot;
    bool needFull = false;
    while (s && !s->materialized) {
        if (s->needRefresh || n >= 255) { needFull = true; break; }
        chain[n++] = s;
        s = const_cast<NapkAccSlot*>(s->prev);
    }
    if (needFull || !s) {
        // 🦅 estilo SF (AccumulatorCaches): refresh do CACHE do king-bucket (finny), não do zero.
        //   O finnyResolve faz o diff entre os bitboards guardados e o board atual (poucas peças
        //   mudaram desde a última visita ao bucket) e ATUALIZA o cache. tl_finny indexa a rede ativa.
        if (tl_finny.L1 != g_net.L1) {
            tl_finny.L1 = g_net.L1;
            for (int p = 0; p < 2; ++p) for (int b = 0; b < 32; ++b) tl_finny.e[p][b].init = false;
        }
        finnyResolve(tl_finny.e[0][slot->bucketW], board, g_net.L1, 0, slot->bucketW, slot->accW);
        finnyResolve(tl_finny.e[1][slot->bucketB], board, g_net.L1, 1, slot->bucketB, slot->accB);
        slot->materialized = true;
        slot->needRefresh = false;
        return;
    }
    for (int i = n - 1; i >= 0; --i) napkApplyDeltas(chain[i]);
}

// 🦅 Parte 2: refresh COMPLETO — calcula o acc do zero p/ a posição atual (as 2 perspetivas).
//   Usado no init (raiz) e quando o rei muda de king-bucket (Parte 3). Replica o cálculo do
//   finnyResolve !init: acc = bias + soma das features de todas as peças. SEM threats (somados
//   no forward). Opera na rede ATIVA (g_net via g_netPtr/g_netIdx).
void napkRefresh(const Board& board, NapkAccSlot& slot) {
    if (!g_net.loaded) { slot.valid = false; return; }
    const int L1 = g_net.L1;
    int kw = board.kingSq(Color::WHITE).value();
    int kb = board.kingSq(Color::BLACK).value();
    int b_w = BUCKET_MAP[kw];
    int b_b = BUCKET_MAP[kb ^ 56];
    slot.bucketW = b_w;
    slot.bucketB = b_b;

    // bias inicial nas 2 perspetivas
    for (int i = 0; i < L1; ++i) { slot.accW[i] = g_net.accBias[i]; slot.accB[i] = g_net.accBias[i]; }

    // somar as features de cada peça (perspetiva branca: sq tal-qual; preta: sq^56)
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
        {
            uint64_t bbv = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t)).value();
            int pc = engCode(t, c);
            int kindW = pieceKindW(pc);
            int kindB = pieceKindB(pc);
            uint64_t bits = bbv;
            while (bits) {
                int sq = std::countr_zero(bits); bits &= bits - 1;
                if (kindW != -1) {
                    const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_w, kindW, sq) * L1];
                    for (int i = 0; i < L1; ++i) slot.accW[i] += w[i];
                }
                if (kindB != -1) {
                    const int16_t* w = &g_net.accWeight[(size_t)makeFeat(b_b, kindB, sq ^ 56) * L1];
                    for (int i = 0; i < L1; ++i) slot.accB[i] += w[i];
                }
            }
        }
    slot.valid = true;
    slot.thrValid = false;   // 🦅 s29: refresh invalida os threats (recalculados no forward)
    slot.materialized = true;
    slot.needRefresh = false;
}

}  // namespace napoleon::nnue

namespace napoleon::nnue {
bool hasChaosHead() { return g_net.hasChaos; }

float chaosScore(const Board& board)
{
    if (!g_net.loaded || !g_net.hasChaos) return 0.5f;

    const int L1 = g_net.L1;
    const int stm = static_cast<int>(board.sideToMove());

    int kw = board.kingSq(Color::WHITE).value();
    int kb = board.kingSq(Color::BLACK).value();
    int b_w = BUCKET_MAP[kw];
    int b_b = BUCKET_MAP[kb ^ 56];

    // Recalcule l'accumulator (même logique que evaluate()).
    std::vector<int32_t> accW(L1), accB(L1);
    for (int i = 0; i < L1; ++i) { accW[i] = g_net.accBias[i]; accB[i] = g_net.accBias[i]; }

    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 6; ++t)
        {
            Bitboard bb = board.pieces(static_cast<Color>(c), static_cast<PieceType>(t));
            while (bb.any())
            {
                int sq = bb.poplsb().value();
                int pc = engCode(t, c);
                int kWk = pieceKindW(pc);
                if (kWk != -1)
                {
                    int f = makeFeat(b_w, kWk, sq);
                    const int16_t* w = &g_net.accWeight[(size_t)f * L1];
                    for (int i = 0; i < L1; ++i) accW[i] += w[i];
                }
                int kBk = pieceKindB(pc);
                if (kBk != -1)
                {
                    int f = makeFeat(b_b, kBk, sq ^ 56);
                    const int16_t* w = &g_net.accWeight[(size_t)f * L1];
                    for (int i = 0; i < L1; ++i) accB[i] += w[i];
                }
            }
        }

    const int32_t* accUs   = (stm == 0) ? accW.data() : accB.data();
    const int32_t* accThem = (stm == 0) ? accB.data() : accW.data();
    const float QA = g_net.qa, QB = g_net.qb;

    // concat [0,127] comme pour les battalions
    std::vector<uint8_t> concat(L1 * 2);
    for (int i = 0; i < L1; ++i)
    {
        float u = std::clamp((float)accUs[i]   / QA, 0.0f, 1.0f);
        float v = std::clamp((float)accThem[i] / QA, 0.0f, 1.0f);
        concat[i]      = static_cast<uint8_t>(std::lround(u * 127.0f));
        concat[L1 + i] = static_cast<uint8_t>(std::lround(v * 127.0f));
    }

    const int l1_in = L1 * 2;
    const int o1 = g_net.ch_l1_out;   // 32
    const int o2 = g_net.ch_l2_out;   // 16

    // Déquant : weights ×QB, bias ×QA×QB. couche1 entrée ×127 (concat) →
    // dequant1 = 1/(127*QB) ; le bias est en ×QA×QB → /(QA*QB).
    const float dequant1 = 1.0f / (127.0f * QB);
    const float biasScale1 = 1.0f / (QA * QB);

    // chaos_l1 : [o1] = crelu( concat · W + b )
    std::vector<float> a1(o1);
    for (int o = 0; o < o1; ++o)
    {
        const int16_t* w = &g_net.ch_l1_w[(size_t)o * l1_in];
        int64_t sum = 0;
        for (int i = 0; i < l1_in; ++i) sum += (int)concat[i] * (int)w[i];
        float val = (float)g_net.ch_l1_b[o] * biasScale1 + (float)sum * dequant1;
        a1[o] = std::clamp(val, 0.0f, 1.0f);
    }

    // chaos_l2 : [o2] = crelu( a1 · W + b ). a1 ∈ [0,1], weights ×QB → /QB.
    const float dequant2 = 1.0f / QB;
    const float biasScale2 = 1.0f / (QA * QB);
    std::vector<float> a2(o2);
    for (int o = 0; o < o2; ++o)
    {
        const int16_t* w = &g_net.ch_l2_w[(size_t)o * o1];
        float sum = (float)g_net.ch_l2_b[o] * biasScale2;
        for (int i = 0; i < o1; ++i) sum += a1[i] * (float)w[i] / QB;
        a2[o] = std::clamp(sum, 0.0f, 1.0f);
    }

    // chaos_l3 : [1] = a2 · W + b  (logit), puis sigmoid.
    float logit = (float)g_net.ch_l3_b[0] / (QA * QB);
    for (int i = 0; i < o2; ++i) logit += a2[i] * (float)g_net.ch_l3_w[i] / QB;

    float chaos = 1.0f / (1.0f + std::exp(-logit));
    return std::clamp(chaos, 0.0f, 1.0f);
}
}
