// 🦅 sf17_small_interpret.cpp -- standalone validation tool, NOT part of the engine.
//
// Reads a REAL official Stockfish 17.1 .nnue file (classic HalfKAv2_hm only, no
// FullThreats -- that was added later, in current master) and computes an eval for a
// given FEN. Same purpose as tools/sf_nnue_interpret.cpp (validate littleindian's own
// dense-layer/SIMD infrastructure against a real, externally-produced network) but for
// the SMALL net (L1=128, fast, used by SF as a cheap pre-filter) so we get a clean,
// isolated throughput number uncomplicated by FullThreats.
//
// Format read directly from Stockfish 17.1's own source (GPL-3.0, tag sf_17.1,
// github.com/official-stockfish/Stockfish): nnue_common.h, network.cpp,
// nnue_feature_transformer.h, nnue_architecture.h, features/half_ka_v2_hm.{h,cpp},
// layers/affine_transform*.h, layers/{clipped_relu,sqr_clipped_relu}.h. No SF code
// copied -- independent reimplementation of the documented file format. Differs from
// sf_nnue_interpret.cpp in real ways verified against THIS version's source, not
// assumed from the other one: version constant, no threat chunks at all, a "scale
// weights by 2 after reading" step this version has (and the other doesn't -- it bakes
// the x2 into the file directly), and SqrClippedReLU/ClippedReLU using the plain
// WeightScaleBits (no +1 override like current master's first layer).
//
// Usage: sf17_small_interpret <path-to.nnue> <L1> <L2> "<fen>" [fen2] ...
//   (L1/L2 let this same binary handle both the small (128,15) and big (3072,15) nets)

#include "../src/board.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

using i8 = int8_t; using u8 = uint8_t;
using i16 = int16_t;
using i32 = int32_t; using u32 = uint32_t;
using i64 = int64_t;

static constexpr int HALFKA_DIMS = 22528;
static constexpr int PSQT_BUCKETS = 8;
static constexpr int LAYER_STACKS = 8;
static constexpr int WEIGHT_SCALE_BITS = 6;
static constexpr int OUTPUT_SCALE = 16;

template<typename IntType>
static void readLeb128(std::ifstream& f, IntType* out, size_t count) {
    char magic[17];
    f.read(magic, 17);
    u32 bytesLeft;
    f.read(reinterpret_cast<char*>(&bytesLeft), 4);
    std::vector<u8> buf(8192);
    size_t bufPos = buf.size();
    for (size_t idx = 0; idx < count; idx++) {
        IntType result = 0;
        int shift = 0;
        while (true) {
            if (bufPos == buf.size()) {
                size_t toRead = std::min(static_cast<size_t>(bytesLeft), buf.size());
                f.read(reinterpret_cast<char*>(buf.data()), toRead);
                bufPos = 0;
            }
            u8 byte = buf[bufPos++];
            bytesLeft--;
            result = static_cast<IntType>(result | (IntType)((byte & 0x7f) << (shift % 32)));
            shift += 7;
            if ((byte & 0x80) == 0) {
                if (!(shift >= 32 || (byte & 0x40) == 0))
                    result = static_cast<IntType>(result | ~((IntType(1) << shift) - 1));
                break;
            }
        }
        out[idx] = result;
    }
}

template<typename IntType>
static void readRaw(std::ifstream& f, IntType* out, size_t count) {
    f.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
}

enum { SF_PAWN = 1, SF_KNIGHT = 2, SF_BISHOP = 3, SF_ROOK = 4, SF_QUEEN = 5, SF_KING = 6 };
static int sfPiece(int color, int sfPieceType) { return color * 8 + sfPieceType; }

enum { PS_NONE = 0, PS_W_PAWN = 0, PS_B_PAWN = 64, PS_W_KNIGHT = 128, PS_B_KNIGHT = 192,
       PS_W_BISHOP = 256, PS_B_BISHOP = 320, PS_W_ROOK = 384, PS_B_ROOK = 448,
       PS_W_QUEEN = 512, PS_B_QUEEN = 576, PS_KING = 640, PS_NB = 704 };

static int pieceSquareIndex(int perspective, int piece) {
    static const int W[16] = {PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
                               PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE};
    static const int B[16] = {PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
                               PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE};
    return perspective == 0 ? W[piece] : B[piece];
}
static int kingBucket(int sq) {
    static const int KB[64] = {
        28,29,30,31,31,30,29,28, 24,25,26,27,27,26,25,24, 20,21,22,23,23,22,21,20, 16,17,18,19,19,18,17,16,
        12,13,14,15,15,14,13,12, 8, 9, 10,11,11,10, 9, 8,  4, 5, 6, 7, 7, 6, 5, 4,  0, 1, 2, 3, 3, 2, 1, 0,
    };
    return KB[sq] * PS_NB;
}
static int orientTbl(int sq) {
    static const int OT[64] = {
        7,7,7,7,0,0,0,0, 7,7,7,7,0,0,0,0, 7,7,7,7,0,0,0,0, 7,7,7,7,0,0,0,0,
        7,7,7,7,0,0,0,0, 7,7,7,7,0,0,0,0, 7,7,7,7,0,0,0,0, 7,7,7,7,0,0,0,0,
    };
    return OT[sq];
}
static int halfKaMakeIndex(int perspective, int s, int pc, int ksq) {
    int flip = 56 * perspective;
    return (s ^ orientTbl(ksq) ^ flip) + pieceSquareIndex(perspective, pc) + kingBucket(ksq ^ flip);
}

struct LayerStack {
    std::vector<i32> fc0_bias; std::vector<i8> fc0_w;  // [L2+1] , [L2+1][L1]
    std::vector<i32> fc1_bias; std::vector<i8> fc1_w;  // [L3]   , [L3][L2*2]
    i32 fc2_bias[1] = {0};     std::vector<i8> fc2_w;  // [1]    , [1][L3]
};

struct Sf17Network {
    int L1 = 0, L2 = 0, L3 = 0;
    std::vector<i16> ftBiases, ftWeights;   // [L1] , [22528][L1]
    std::vector<i32> psqtWeights;           // [22528][8]
    LayerStack stacks[LAYER_STACKS];

    bool load(const std::string& path, int l1, int l2, int l3) {
        L1 = l1; L2 = l2; L3 = l3;
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

        u32 version, hashValue, descLen;
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&hashValue), 4);
        f.read(reinterpret_cast<char*>(&descLen), 4);
        f.seekg(descLen, std::ios::cur);
        if (version != 0x7AF32F20u) {
            std::fprintf(stderr, "bad version 0x%X (esperava 0x7AF32F20 -- SF 17.1)\n", version);
            return false;
        }

        u32 ftHash; f.read(reinterpret_cast<char*>(&ftHash), 4);
        if (getenv("SF17_DEBUG")) std::fprintf(stderr, "after header+fthash: pos=%lld\n", (long long)f.tellg());

        ftBiases.resize(L1);
        readLeb128<i16>(f, ftBiases.data(), L1);
        if (getenv("SF17_DEBUG")) std::fprintf(stderr, "after ftBiases: pos=%lld (bias[0..2]=%d,%d,%d)\n",
                                               (long long)f.tellg(), ftBiases[0], ftBiases[1], ftBiases[2]);
        ftWeights.resize(static_cast<size_t>(HALFKA_DIMS) * L1);
        readLeb128<i16>(f, ftWeights.data(), ftWeights.size());
        if (getenv("SF17_DEBUG")) std::fprintf(stderr, "after ftWeights: pos=%lld\n", (long long)f.tellg());
        psqtWeights.resize(static_cast<size_t>(HALFKA_DIMS) * PSQT_BUCKETS);
        readLeb128<i32>(f, psqtWeights.data(), psqtWeights.size());
        if (getenv("SF17_DEBUG")) std::fprintf(stderr, "after psqtWeights: pos=%lld\n", (long long)f.tellg());

        // 🦅 scale_weights(true): esta versão guarda os pesos/biases SEM o x2 da
        // permutação SIMD -- duplica-os depois de ler (ver nnue_feature_transformer.h
        // desta versão, scale_weights()). A versão "current master" já vem pré-duplicada
        // no ficheiro, por isso o outro interpretador (sf_nnue_interpret.cpp) NÃO faz isto.
        for (auto& b : ftBiases) b = static_cast<i16>(b * 2);
        for (auto& w : ftWeights) w = static_cast<i16>(w * 2);

        for (int b = 0; b < LAYER_STACKS; b++) {
            u32 archHash; f.read(reinterpret_cast<char*>(&archHash), 4);
            LayerStack& s = stacks[b];
            int fc0Total = L2 + 1;

            s.fc0_bias.resize(fc0Total);
            readRaw<i32>(f, s.fc0_bias.data(), fc0Total);
            s.fc0_w.resize(static_cast<size_t>(fc0Total) * L1);
            readRaw<i8>(f, s.fc0_w.data(), s.fc0_w.size());

            // 🦅 AffineTransform PADDED a múltiplos de 32 (MaxSimdWidth) -- L2*2=30 para
            //   L2=15 não é múltiplo de 32, fica armazenado com stride 32 (2 colunas de
            //   padding por linha, sempre zero do lado da entrada). Faltava isto -- 64
            //   bytes/bucket em falta (32*32-32*30), exatamente os 512 bytes (×8) que
            //   sobravam no ficheiro. Sem isto cada bucket >0 lia desalinhado.
            int paddedL2x2 = ((L2 * 2 + 31) / 32) * 32;
            s.fc1_bias.resize(L3);
            readRaw<i32>(f, s.fc1_bias.data(), L3);
            s.fc1_w.resize(static_cast<size_t>(L3) * paddedL2x2);
            readRaw<i8>(f, s.fc1_w.data(), s.fc1_w.size());

            readRaw<i32>(f, s.fc2_bias, 1);
            s.fc2_w.resize(L3);
            readRaw<i8>(f, s.fc2_w.data(), s.fc2_w.size());
            if (getenv("SF17_DEBUG") && b < 2)
                std::fprintf(stderr, "after bucket %d: pos=%lld fc0_bias[0..2]=%d,%d,%d\n",
                             b, (long long)f.tellg(), s.fc0_bias[0], s.fc0_bias[1], s.fc0_bias[2]);
        }
        if (getenv("SF17_DEBUG")) {
            f.clear();
            auto cur = f.tellg();
            f.seekg(0, std::ios::end);
            std::fprintf(stderr, "final pos=%lld, file size=%lld\n", (long long)cur, (long long)f.tellg());
        }
        return !f.fail();
    }
};

static std::pair<int,int> sf17Evaluate(const Sf17Network& net, const Board& board, double* usSecOut = nullptr) {
    auto t0 = std::chrono::high_resolution_clock::now();
    const int L1 = net.L1, L2 = net.L2, L3 = net.L3;

    int stm = board.sideToMove() == Color::WHITE ? 0 : 1;
    int perspectives[2] = {stm, 1 - stm};

    static thread_local std::vector<i32> accumulation[2];
    accumulation[0].assign(L1, 0); accumulation[1].assign(L1, 0);
    i32 psqtAcc[2][PSQT_BUCKETS];

    for (int p = 0; p < 2; p++) {
        int ksq = board.kingSq(p == 0 ? Color::WHITE : Color::BLACK).value();
        for (int j = 0; j < L1; j++) accumulation[p][j] = net.ftBiases[j];
        for (int k = 0; k < PSQT_BUCKETS; k++) psqtAcc[p][k] = 0;
        for (int sq = 0; sq < 64; sq++) {
            PieceType pt = board.pieceOn(sq);
            if (pt == PieceType::NONE) continue;
            int color = board.colorOn(sq) == Color::WHITE ? 0 : 1;
            int sfPt = static_cast<int>(pt) + 1;
            int pc = sfPiece(color, sfPt);
            int idx = halfKaMakeIndex(p, sq, pc, ksq);
            const i16* w = &net.ftWeights[static_cast<size_t>(idx) * L1];
            for (int j = 0; j < L1; j++) accumulation[p][j] += w[j];
            const i32* pw = &net.psqtWeights[static_cast<size_t>(idx) * PSQT_BUCKETS];
            for (int k = 0; k < PSQT_BUCKETS; k++) psqtAcc[p][k] += pw[k];
        }
    }

    int pieceCount = (board.pieces(Color::WHITE, PieceType::PAWN) | board.pieces(Color::WHITE, PieceType::KNIGHT) |
                       board.pieces(Color::WHITE, PieceType::BISHOP) | board.pieces(Color::WHITE, PieceType::ROOK) |
                       board.pieces(Color::WHITE, PieceType::QUEEN) | board.pieces(Color::WHITE, PieceType::KING) |
                       board.pieces(Color::BLACK, PieceType::PAWN) | board.pieces(Color::BLACK, PieceType::KNIGHT) |
                       board.pieces(Color::BLACK, PieceType::BISHOP) | board.pieces(Color::BLACK, PieceType::ROOK) |
                       board.pieces(Color::BLACK, PieceType::QUEEN) | board.pieces(Color::BLACK, PieceType::KING)).popcount();
    int bucket = std::clamp((pieceCount - 1) / 4, 0, 7);

    i32 psqt = (psqtAcc[perspectives[0]][bucket] - psqtAcc[perspectives[1]][bucket]) / 2;

    static thread_local std::vector<u8> tf;
    tf.assign(L1, 0);
    for (int p = 0; p < 2; p++) {
        int offset = (L1 / 2) * p;
        for (int j = 0; j < L1 / 2; j++) {
            i32 sum0 = std::clamp(accumulation[perspectives[p]][j], 0, 254);
            i32 sum1 = std::clamp(accumulation[perspectives[p]][j + L1 / 2], 0, 254);
            tf[offset + j] = static_cast<u8>(static_cast<unsigned>(sum0 * sum1) / 512);
        }
    }

    const LayerStack& s = net.stacks[bucket];
    int fc0Total = L2 + 1;
    std::vector<i32> fc0_out(fc0Total);
    for (int o = 0; o < fc0Total; o++) {
        i64 sum = s.fc0_bias[o];
        const i8* row = &s.fc0_w[static_cast<size_t>(o) * L1];
        for (int i = 0; i < L1; i++) sum += static_cast<i32>(row[i]) * static_cast<i32>(tf[i]);
        fc0_out[o] = static_cast<i32>(sum);
    }
    if (getenv("SF17_DEBUG")) {
        std::fprintf(stderr, "bucket=%d tf[0..4]=%d,%d,%d,%d fc0_bias[0..4]=%d,%d,%d,%d fc0_out[0..4]=%d,%d,%d,%d fc0_out[skip=%d]=%d\n",
                     bucket, tf[0],tf[1],tf[2],tf[3], s.fc0_bias[0],s.fc0_bias[1],s.fc0_bias[2],s.fc0_bias[3],
                     fc0_out[0],fc0_out[1],fc0_out[2],fc0_out[3], L2, fc0_out[L2]);
    }

    std::vector<u8> concat(L2 * 2, 0);
    for (int i = 0; i < L2; i++) {
        i64 v = static_cast<i64>(fc0_out[i]) * fc0_out[i];
        i64 sq = v >> (2 * WEIGHT_SCALE_BITS + 7);
        concat[i] = static_cast<u8>(std::min<i64>(127, sq));
    }
    for (int i = 0; i < L2; i++) {
        i32 v = fc0_out[i] >> WEIGHT_SCALE_BITS;
        concat[L2 + i] = static_cast<u8>(std::clamp(v, 0, 127));
    }

    int paddedL2x2 = ((L2 * 2 + 31) / 32) * 32;
    std::vector<i32> fc1_out(L3);
    for (int o = 0; o < L3; o++) {
        i64 sum = s.fc1_bias[o];
        const i8* row = &s.fc1_w[static_cast<size_t>(o) * paddedL2x2];   // stride = padded, não L2*2
        for (int i = 0; i < L2 * 2; i++) sum += static_cast<i32>(row[i]) * static_cast<i32>(concat[i]);
        fc1_out[o] = static_cast<i32>(sum);
    }
    std::vector<u8> ac1(L3);
    for (int i = 0; i < L3; i++) ac1[i] = static_cast<u8>(std::clamp(fc1_out[i] >> WEIGHT_SCALE_BITS, 0, 127));

    i64 fc2_sum = s.fc2_bias[0];
    for (int i = 0; i < L3; i++) fc2_sum += static_cast<i32>(s.fc2_w[i]) * static_cast<i32>(ac1[i]);

    i64 fwdOut = (static_cast<i64>(fc0_out[L2]) * (600LL * OUTPUT_SCALE)) / (127LL * (1LL << WEIGHT_SCALE_BITS));
    i64 positionalRaw = fc2_sum + fwdOut;

    int psqtFinal = psqt / OUTPUT_SCALE;
    int positionalFinal = static_cast<int>(positionalRaw / OUTPUT_SCALE);

    if (usSecOut) {
        auto t1 = std::chrono::high_resolution_clock::now();
        *usSecOut = std::chrono::duration<double>(t1 - t0).count();
    }
    return {psqtFinal, positionalFinal};
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <net.nnue> <L1> <L2> \"<fen>\" [fen2] ...\n", argv[0]);
        std::fprintf(stderr, "  small net: L1=128 L2=15 ; big net: L1=3072 L2=15\n");
        return 1;
    }
    int l1 = std::atoi(argv[2]), l2 = std::atoi(argv[3]);
    Sf17Network net;
    if (!net.load(argv[1], l1, l2, 32)) return 1;
    std::fprintf(stderr, "net loaded ok (L1=%d L2=%d L3=32)\n", l1, l2);

    for (int i = 4; i < argc; i++) {
        Board b;
        if (!b.setFen(argv[i])) { std::fprintf(stderr, "bad fen: %s\n", argv[i]); continue; }
        double usSec = 0;
        auto [psqt, positional] = sf17Evaluate(net, b, &usSec);
        std::printf("fen: %s\n  psqt=%d positional=%d total(internal,stm-pov)=%d  (%.2f us)\n",
                    argv[i], psqt, positional, psqt + positional, usSec * 1e6);
    }

    if (argc > 4) {
        Board b;
        b.setFen(argv[argc - 1]);
        auto t0 = std::chrono::high_resolution_clock::now();
        const int N = 200000;
        volatile int sink = 0;
        for (int i = 0; i < N; i++) {
            auto [p, q] = sf17Evaluate(net, b);
            sink += p + q;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("throughput: %d evals in %.3fs = %.0f evals/sec (full from-scratch, no SIMD)\n",
                    N, secs, N / secs);
    }
    return 0;
}
