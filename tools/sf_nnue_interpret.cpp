// 🦅 sf_nnue_interpret.cpp -- standalone validation tool, NOT part of the engine.
//
// Reads a REAL official Stockfish .nnue file (current master format: HalfKAv2_hm +
// FullThreats input, L1=1024 -> FC0(31+1 skip) -> [sqr-clipped-relu(31) concat
// clipped-relu(31)] -> FC1(32) -> clipped-relu -> FC2(1)) and computes an eval for a
// given FEN using ONLY the HalfKAv2_hm feature set (FullThreats deliberately
// skipped -- its own feature-index scheme is far more involved; this tool's purpose
// is to validate littleindian's own SIMD/dense-layer infrastructure against a real,
// externally-produced network, not to fully replicate SF's threats).
//
// Format read directly from Stockfish's own source (GPL-3.0,
// github.com/official-stockfish/Stockfish): nnue_common.h, network.cpp,
// nnue_feature_transformer.h, nnue_architecture.h, features/half_ka_v2_hm.{h,cpp},
// layers/affine_transform*.h, layers/{clipped_relu,sqr_clipped_relu}.h. No SF code
// copied -- this is an independent reimplementation of the documented file format.
//
// Usage: sf_nnue_interpret <path-to.nnue> "<fen>" [fen2] [fen3] ...

#include "../src/board.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <string>

using i8 = int8_t; using u8 = uint8_t;
using i16 = int16_t;
using i32 = int32_t; using u32 = uint32_t;
using i64 = int64_t;

static constexpr int      L1               = 1024;
static constexpr int      L2               = 31;
static constexpr int      L3               = 32;
static constexpr int      PSQT_BUCKETS     = 8;
static constexpr int      LAYER_STACKS     = 8;
static constexpr int      HALFKA_DIMS      = 22528;
static constexpr int      THREAT_DIMS      = 60720;
static constexpr int      WEIGHT_SCALE_BITS = 6;
static constexpr int      FT_MAX_VAL       = 255;
static constexpr int      HIDDEN_ONE_VAL   = 128;
static constexpr int      OUTPUT_SCALE     = 16;

// ───────────────────────── LEB128 (matches SF's read_leb_128_detail exactly) ─────
template<typename IntType>
static void readLeb128(std::ifstream& f, IntType* out, size_t count) {
    char magic[17];
    f.read(magic, 17);  // "COMPRESSED_LEB128" -- trusted, not re-checked
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

static void skipLeb128Section(std::ifstream& f) {
    char magic[17];
    f.read(magic, 17);
    u32 bytesLeft;
    f.read(reinterpret_cast<char*>(&bytesLeft), 4);
    f.seekg(bytesLeft, std::ios::cur);
}

template<typename IntType>
static void readRaw(std::ifstream& f, IntType* out, size_t count) {
    f.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
}

// ───────────────────────── HalfKAv2_hm feature indexing ──────────────────────────
// Piece encoding matches SF's types.h: PieceType 1=PAWN..6=KING, Piece = color*8+pt.
enum { SF_NO_PIECE_TYPE = 0, SF_PAWN = 1, SF_KNIGHT = 2, SF_BISHOP = 3, SF_ROOK = 4, SF_QUEEN = 5, SF_KING = 6 };

static int sfPiece(int color, int sfPieceType) { return color * 8 + sfPieceType; }

enum { PS_NONE = 0, PS_W_PAWN = 0, PS_B_PAWN = 64, PS_W_KNIGHT = 128, PS_B_KNIGHT = 192,
       PS_W_BISHOP = 256, PS_B_BISHOP = 320, PS_W_ROOK = 384, PS_B_ROOK = 448,
       PS_W_QUEEN = 512, PS_B_QUEEN = 576, PS_KING = 640, PS_NB = 704 };

// PieceSquareIndex[perspective][sfPiece] -- mirrors half_ka_v2_hm.h exactly.
static int pieceSquareIndex(int perspective, int piece) {
    static const int W[16] = {PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
                               PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE};
    static const int B[16] = {PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
                               PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE};
    return perspective == 0 ? W[piece] : B[piece];
}

static int kingBucket(int sq) {
    static const int KB[64] = {
        28,29,30,31,31,30,29,28,
        24,25,26,27,27,26,25,24,
        20,21,22,23,23,22,21,20,
        16,17,18,19,19,18,17,16,
        12,13,14,15,15,14,13,12,
        8, 9, 10,11,11,10, 9, 8,
        4, 5, 6, 7, 7, 6, 5, 4,
        0, 1, 2, 3, 3, 2, 1, 0,
    };
    return KB[sq] * PS_NB;
}
static int orientTbl(int sq) {
    // SQ_H1=7, SQ_A1=0 in SF's Square enum (a1=0..h8=63, file+rank*8, same as ours).
    static const int OT[64] = {
        7,7,7,7,0,0,0,0,  7,7,7,7,0,0,0,0,  7,7,7,7,0,0,0,0,  7,7,7,7,0,0,0,0,
        7,7,7,7,0,0,0,0,  7,7,7,7,0,0,0,0,  7,7,7,7,0,0,0,0,  7,7,7,7,0,0,0,0,
    };
    return OT[sq];
}

// make_index: perspective(0=white,1=black), s=piece square (absolute), pc=SF piece code, ksq=own king sq (absolute)
static int halfKaMakeIndex(int perspective, int s, int pc, int ksq) {
    int flip = 56 * perspective;
    return (s ^ orientTbl(ksq) ^ flip) + pieceSquareIndex(perspective, pc) + kingBucket(ksq ^ flip);
}

// ───────────────────────── Network data ───────────────────────────────────────────
struct LayerStack {
    i32 fc0_bias[L2 + 1];
    std::vector<i8> fc0_w;  // [L2+1][1024] row-major
    i32 fc1_bias[L3];
    std::vector<i8> fc1_w;  // [32][64] row-major (64 = ceil(62,32))
    i32 fc2_bias[1];
    std::vector<i8> fc2_w;  // [1][32] row-major
};

struct SfNetwork {
    std::vector<i16> ftBiases;    // [1024]
    std::vector<i16> ftWeights;   // [22528][1024] row-major
    std::vector<i32> psqtWeights; // [22528][8] row-major
    LayerStack stacks[LAYER_STACKS];

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

        u32 version, hashValue, descLen;
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&hashValue), 4);
        f.read(reinterpret_cast<char*>(&descLen), 4);
        f.seekg(descLen, std::ios::cur);
        if (version != 0x6A448AFAu) { std::fprintf(stderr, "bad version\n"); return false; }

        u32 ftHash; f.read(reinterpret_cast<char*>(&ftHash), 4);  // FeatureTransformer section header

        ftBiases.resize(L1);
        readLeb128<i16>(f, ftBiases.data(), L1);

        f.seekg(static_cast<std::streamoff>(THREAT_DIMS) * L1, std::ios::cur);  // threatWeights, raw i8, skipped
        skipLeb128Section(f);  // threatPsqtWeights, LEB128 i32, skipped

        ftWeights.resize(static_cast<size_t>(HALFKA_DIMS) * L1);
        readLeb128<i16>(f, ftWeights.data(), ftWeights.size());

        psqtWeights.resize(static_cast<size_t>(HALFKA_DIMS) * PSQT_BUCKETS);
        readLeb128<i32>(f, psqtWeights.data(), psqtWeights.size());

        for (int b = 0; b < LAYER_STACKS; b++) {
            u32 archHash; f.read(reinterpret_cast<char*>(&archHash), 4);
            LayerStack& s = stacks[b];

            readRaw<i32>(f, s.fc0_bias, L2 + 1);
            s.fc0_w.resize(static_cast<size_t>(L2 + 1) * L1);
            readRaw<i8>(f, s.fc0_w.data(), s.fc0_w.size());

            readRaw<i32>(f, s.fc1_bias, L3);
            s.fc1_w.resize(static_cast<size_t>(L3) * 64);  // PaddedInputDimensions(62)=64
            readRaw<i8>(f, s.fc1_w.data(), s.fc1_w.size());

            readRaw<i32>(f, s.fc2_bias, 1);
            s.fc2_w.resize(32);  // PaddedInputDimensions(32)=32
            readRaw<i8>(f, s.fc2_w.data(), s.fc2_w.size());
        }
        return !f.fail();
    }
};

// ───────────────────────── Evaluation ──────────────────────────────────────────
// Returns (psqt, positional) in SF's "internal units" (post /OutputScale, pre cp conversion).
static std::pair<int,int> sfEvaluate(const SfNetwork& net, const Board& board, double* usSecOut = nullptr) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int stm = board.sideToMove() == Color::WHITE ? 0 : 1;
    int perspectives[2] = {stm, 1 - stm};

    i32 psqtAcc[2][PSQT_BUCKETS];
    static thread_local i32 accumulation[2][L1];

    for (int p = 0; p < 2; p++) {
        int persp = p;  // compute raw accumulator for color `persp` directly (0=white,1=black)
        int ksq = board.kingSq(persp == 0 ? Color::WHITE : Color::BLACK).value();

        for (int j = 0; j < L1; j++) accumulation[persp][j] = net.ftBiases[j];
        for (int k = 0; k < PSQT_BUCKETS; k++) psqtAcc[persp][k] = 0;

        for (int sq = 0; sq < 64; sq++) {
            PieceType pt = board.pieceOn(sq);
            if (pt == PieceType::NONE) continue;
            int color = board.colorOn(sq) == Color::WHITE ? 0 : 1;
            int sfPt = static_cast<int>(pt) + 1;  // our PAWN=0..KING=5 -> SF PAWN=1..KING=6
            int pc = sfPiece(color, sfPt);
            int idx = halfKaMakeIndex(persp, sq, pc, ksq);
            const i16* w = &net.ftWeights[static_cast<size_t>(idx) * L1];
            for (int j = 0; j < L1; j++) accumulation[persp][j] += w[j];
            const i32* pw = &net.psqtWeights[static_cast<size_t>(idx) * PSQT_BUCKETS];
            for (int k = 0; k < PSQT_BUCKETS; k++) psqtAcc[persp][k] += pw[k];
        }
    }

    int pieceCount = (board.pieces(Color::WHITE, PieceType::PAWN) | board.pieces(Color::WHITE, PieceType::KNIGHT) |
                       board.pieces(Color::WHITE, PieceType::BISHOP) | board.pieces(Color::WHITE, PieceType::ROOK) |
                       board.pieces(Color::WHITE, PieceType::QUEEN) | board.pieces(Color::WHITE, PieceType::KING) |
                       board.pieces(Color::BLACK, PieceType::PAWN) | board.pieces(Color::BLACK, PieceType::KNIGHT) |
                       board.pieces(Color::BLACK, PieceType::BISHOP) | board.pieces(Color::BLACK, PieceType::ROOK) |
                       board.pieces(Color::BLACK, PieceType::QUEEN) | board.pieces(Color::BLACK, PieceType::KING)).popcount();
    int bucket = (pieceCount - 1) / 4;
    if (bucket < 0) bucket = 0; if (bucket > 7) bucket = 7;

    i32 psqt = (psqtAcc[perspectives[0]][bucket] - psqtAcc[perspectives[1]][bucket]) / 2;

    // transform(): pairwise clamp+multiply/512 -> 1024 u8 transformed features.
    // Builds the nonzero-index list (nnz) inline, same idea as SF's own NNZInfo --
    // exploits the fact that a clamped-to-0 accumulator value zeroes the whole pair.
    static thread_local u8  tf[L1];
    static thread_local int nnzIdx[L1];
    int nnzCount = 0;
    for (int p = 0; p < 2; p++) {
        int offset = (L1 / 2) * p;
        const i32* acc = accumulation[perspectives[p]];
        for (int j = 0; j < L1 / 2; j++) {
            i32 sum0 = std::clamp(acc[j], 0, FT_MAX_VAL);
            i32 sum1 = std::clamp(acc[j + L1 / 2], 0, FT_MAX_VAL);
            u8 v = static_cast<u8>(static_cast<unsigned>(sum0 * sum1) / 512);
            tf[offset + j] = v;
            if (v) nnzIdx[nnzCount++] = offset + j;
        }
    }

    const LayerStack& s = net.stacks[bucket];

    // fc_0: 1024 -> 32 (31 real + 1 skip). Sparse: only the nonzero tf[] entries
    // contribute, transposed accumulation order like SF's own scalar fallback.
    i32 fc0_out[L2 + 1];
    for (int o = 0; o < L2 + 1; o++) fc0_out[o] = s.fc0_bias[o];
    for (int n = 0; n < nnzCount; n++) {
        int i = nnzIdx[n];
        int val = tf[i];
        const i8* col = &s.fc0_w[i];  // strided by L1 per output row
        for (int o = 0; o < L2 + 1; o++) fc0_out[o] += static_cast<i32>(col[static_cast<size_t>(o) * L1]) * val;
    }

    // ac_sqr_0 (WeightScaleBits+1=7 shift) on indices [0,31); ac_0 (same shift) on [0,31)
    u8 concat[64] = {0};
    for (int i = 0; i < L2; i++) {
        i64 v = static_cast<i64>(fc0_out[i]) * fc0_out[i];
        i64 sq = v >> (2 * (WEIGHT_SCALE_BITS + 1) + 7);
        concat[i] = static_cast<u8>(std::min<i64>(127, sq));
    }
    for (int i = 0; i < L2; i++) {
        i32 v = fc0_out[i] >> (WEIGHT_SCALE_BITS + 1);
        concat[L2 + i] = static_cast<u8>(std::clamp(v, 0, 127));
    }

    // fc_1: 62(padded 64) -> 32
    i32 fc1_out[L3];
    for (int o = 0; o < L3; o++) {
        i64 sum = s.fc1_bias[o];
        const i8* row = &s.fc1_w[static_cast<size_t>(o) * 64];
        for (int i = 0; i < 64; i++) sum += static_cast<i32>(row[i]) * static_cast<i32>(concat[i]);
        fc1_out[o] = static_cast<i32>(sum);
    }

    // ac_1: clipped relu (shift=6) on 32
    u8 ac1[32];
    for (int i = 0; i < L3; i++) ac1[i] = static_cast<u8>(std::clamp(fc1_out[i] >> WEIGHT_SCALE_BITS, 0, 127));

    // fc_2: 32 -> 1
    i64 fc2_sum = s.fc2_bias[0];
    for (int i = 0; i < 32; i++) fc2_sum += static_cast<i32>(s.fc2_w[i]) * static_cast<i32>(ac1[i]);

    i64 fwdOut = fc2_sum + fc0_out[L2];  // + skip connection (raw fc0 output #31)
    i64 multiplier = 600LL * OUTPUT_SCALE;
    i64 denominator = static_cast<i64>(HIDDEN_ONE_VAL) * (1LL << WEIGHT_SCALE_BITS) * 2;
    i32 positionalRaw = static_cast<i32>((fwdOut * multiplier) / denominator);

    int psqtFinal = psqt / OUTPUT_SCALE;
    int positionalFinal = positionalRaw / OUTPUT_SCALE;

    if (usSecOut) {
        auto t1 = std::chrono::high_resolution_clock::now();
        *usSecOut = std::chrono::duration<double>(t1 - t0).count();
    }
    return {psqtFinal, positionalFinal};
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <net.nnue> \"<fen>\" [fen2] ...\n", argv[0]);
        return 1;
    }
    SfNetwork net;
    if (!net.load(argv[1])) return 1;
    std::fprintf(stderr, "net loaded ok\n");

    for (int i = 2; i < argc; i++) {
        Board b;
        if (!b.setFen(argv[i])) { std::fprintf(stderr, "bad fen: %s\n", argv[i]); continue; }
        double usSec = 0;
        auto [psqt, positional] = sfEvaluate(net, b, &usSec);
        std::printf("fen: %s\n  psqt=%d positional=%d total(internal,stm-pov)=%d  (%.2f us)\n",
                    argv[i], psqt, positional, psqt + positional, usSec * 1e6);
    }

    // crude throughput test on the last FEN, 200k evals
    if (argc > 2) {
        Board b;
        b.setFen(argv[argc - 1]);
        auto t0 = std::chrono::high_resolution_clock::now();
        const int N = 200000;
        volatile int sink = 0;
        for (int i = 0; i < N; i++) {
            auto [p, q] = sfEvaluate(net, b);
            sink += p + q;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("throughput: %d evals in %.3fs = %.0f evals/sec "
                    "(full from-scratch refresh every call, sparse fc_0, no SIMD)\n",
                    N, secs, N / secs);
    }
    return 0;
}
