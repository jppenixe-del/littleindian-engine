#include "uci.h"
#include "movegen.h"
#include "search.h"
#include "napoleon/nnue_net.h"
#include "napoleon/wdl_brain.h"
#include "tt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <chrono>

static constexpr char START_FEN[] =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ─── Perft ─────────────────────────────────────────────────────────────────
static uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;

    MoveList list;
    generateMoves(board, list);

    uint64_t nodes = 0;
    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        if (!board.isLegal(m)) continue;
        board.makeMove(m);
        nodes += perft(board, depth - 1);
        board.unmakeMove(m);
    }
    return nodes;
}

// Divide perft (shows per-move breakdown)
static void perftDivide(Board& board, int depth) {
    MoveList list;
    generateMoves(board, list);

    uint64_t total = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        if (!board.isLegal(m)) continue;

        board.makeMove(m);
        uint64_t nodes = perft(board, depth - 1);
        board.unmakeMove(m);
        total += nodes;

        // Print move in algebraic notation
        char buf[8];
        buf[0] = 'a' + (m.from() & 7);
        buf[1] = '1' + (m.from() >> 3);
        buf[2] = 'a' + (m.to() & 7);
        buf[3] = '1' + (m.to() >> 3);
        int len = 4;
        if (m.isPromo()) {
            const char promoChar[] = "nbrq";
            buf[len++] = promoChar[m.flags() & 3];
        }
        buf[len] = '\0';
        std::printf("%s: %llu\n", buf, (unsigned long long)nodes);
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\nNodes: %llu  Time: %.0f ms\n", (unsigned long long)total, ms);
}

// ─── Move parsing ──────────────────────────────────────────────────────────
static Move parseMove(const Board& board, const std::string& s) {
    if (s.size() < 4) return NULL_MOVE;
    int from = (s[0] - 'a') + (s[1] - '1') * 8;
    int to   = (s[2] - 'a') + (s[3] - '1') * 8;
    if (from < 0 || from >= 64 || to < 0 || to >= 64) return NULL_MOVE;

    // Detect promotion
    PieceType promo = PieceType::NONE;
    if (s.size() >= 5) {
        switch (s[4]) {
            case 'n': promo = PieceType::KNIGHT; break;
            case 'b': promo = PieceType::BISHOP; break;
            case 'r': promo = PieceType::ROOK;   break;
            case 'q': promo = PieceType::QUEEN;  break;
        }
    }

    // Match against legal moves
    MoveList list;
    generateMoves(const_cast<Board&>(board), list);
    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        if (m.from() != from || m.to() != to) continue;
        if (promo != PieceType::NONE && m.isPromo() && m.promoType() != promo) continue;
        return m;
    }
    return NULL_MOVE;
}

// ─── parsePosition ─────────────────────────────────────────────────────────
void uci::parsePosition(Board& board, const std::string& line) {
    std::istringstream ss(line);
    std::string token;
    ss >> token;  // "position"

    ss >> token;
    if (token == "startpos") {
        board.setFen(START_FEN);
        ss >> token;  // might be "moves"
    } else if (token == "fen") {
        std::string fen;
        while (ss >> token && token != "moves") fen += token + " ";
        if (!fen.empty()) fen.pop_back();
        board.setFen(fen);
        if (token != "moves") return;
    } else {
        board.setFen(START_FEN);
        return;
    }

    if (token == "moves") {
        while (ss >> token) {
            Move m = parseMove(board, token);
            if (!m.isNull()) board.makeMove(m);
        }
    }
}

// ─── UCI loop ──────────────────────────────────────────────────────────────
void uci::loop() {
    Board board;
    board.setFen(START_FEN);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            std::printf("id name littleindian\n");
            std::printf("id author littleindian\n");
            std::printf("option name Hash type spin default 16 min 1 max 16384\n");
            std::printf("option name Threads type spin default 1 min 1 max 1\n");
            std::printf("option name EvalFile type string default <embedded>\n");
            std::printf("option name WdlBrain type check default false\n");
            std::printf("option name WdlFearStrength type spin default 50 min 0 max 100\n");
            std::printf("option name NapkIncremental type check default false\n");
            std::printf("option name MultiPV type spin default 1 min 1 max 8\n");
            printTunableOptions();
            std::printf("uciok\n");
        } else if (cmd == "isready") {
            std::printf("readyok\n");
        } else if (cmd == "setoption") {
            std::string name_tok, name, value_tok, value;
            ss >> name_tok >> name >> value_tok >> value;
            if (name == "Hash") {
                int mb = std::stoi(value);
                gTT.resize(mb);
            } else if (name == "EvalFile") {
                if (value != "<embedded>" && !value.empty()) {
                    if (!napoleon::nnue::load(value))
                        std::printf("info string Failed to load net: %s\n", value.c_str());
                }
            } else if (name == "WdlBrain") {
                napoleon::wdlbrain::setEnabled(value == "true");
            } else if (name == "WdlFearStrength") {
                napoleon::wdlbrain::setFearStrength(std::stoi(value));
            } else if (name == "NapkIncremental") {
                napoleon::nnue::napkSetIncremental(value == "true");
            } else if (name == "MultiPV") {
                setMultiPV(std::atoi(value.c_str()));
            } else if (!setTunableParam(name, std::atoi(value.c_str()))) {
                std::printf("info string unknown option %s\n", name.c_str());
            }
        } else if (cmd == "ucinewgame") {
            board.setFen(START_FEN);
            gTT.clear();
        } else if (cmd == "position") {
            parsePosition(board, line);
        } else if (cmd == "perft") {
            int depth = 1;
            ss >> depth;
            perftDivide(board, depth);
        } else if (cmd == "go") {
            Limits limits;
            std::string tok;
            while (ss >> tok) {
                if (tok == "depth")     { ss >> limits.depth; }
                else if (tok == "movetime") { ss >> limits.movetime; }
                else if (tok == "wtime")    { ss >> limits.wtime; }
                else if (tok == "btime")    { ss >> limits.btime; }
                else if (tok == "winc")     { ss >> limits.winc; }
                else if (tok == "binc")     { ss >> limits.binc; }
                else if (tok == "movestogo"){ ss >> limits.movestogo; }
                else if (tok == "infinite") { limits.infinite = true; limits.depth = 64; }
                else if (tok == "nodes")    { ss >> limits.nodes; }
            }
            search(board, limits);
        } else if (cmd == "stop") {
            // handled via flag in a real engine; for now no-op
        } else if (cmd == "eval") {
            if (napoleon::nnue::isLoaded()) {
                int score = napoleon::nnue::evaluate(board);
                std::printf("eval: %d cp (nnue, stm pov)\n", score);
            } else {
                std::printf("eval: no net loaded\n");
            }
        } else if (cmd == "bench") {
            // Standard bench: a few positions at fixed depth
            static const char* BENCH_FENS[] = {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                nullptr
            };
            int benchDepth = 10;
            std::string tok;
            if (ss >> tok) benchDepth = std::stoi(tok);
            uint64_t totalNodes = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; BENCH_FENS[i]; ++i) {
                Board b;
                b.setFen(BENCH_FENS[i]);
                Limits lim;
                lim.depth = benchDepth;
                // infinite=true desliga o corte por tempo (soft/hard ficam a
                // 0) — bench fica só limitado por depth, sempre o mesmo nº de
                // nós na mesma posição independente da carga da máquina.
                // Sem isto, o soft limit de 5s podia cortar antes da depth
                // pedida sob CPU ocupada, dando uma "assinatura" instável.
                lim.infinite = true;
                uint64_t nodes = 0;
                search(b, lim, &nodes);
                totalNodes += nodes;
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            uint64_t benchNps = ms > 0 ? (uint64_t)(totalNodes * 1000.0 / ms) : totalNodes;
            std::printf("\n%llu nodes %llu nps\n", (unsigned long long)totalNodes, (unsigned long long)benchNps);
            std::printf("bench complete in %.0f ms\n", ms);
        } else if (cmd == "version") {
            napoleon::nnue::printNetInfo();
        } else if (cmd == "threattest") {
            int diff = napoleon::nnue::verifyFinny(board);
            std::printf("threattest: finny diff = %d\n", diff);
        } else if (cmd == "incrtest") {
            int depth = 4;
            std::string tok;
            if (ss >> tok) depth = std::stoi(tok);
            int mismatches = napkIncrementalSelfTest(board, depth);
            std::printf("incrtest: %d posicoes divergentes (depth=%d, 0=ok)\n", mismatches, depth);
        } else if (cmd == "seetest") {
            // Valores esperados calculados para a nossa tabela de peças
            // (P=100, N=325, B=325, R=500, Q=975) — algoritmo validado
            // contra o Coda (10/10) usando a tabela de peças dele antes
            // de se confirmarem estes números com os nossos valores.
            struct Case { const char* fen; const char* mv; int expect; };
            static const Case CASES[] = {
                {"3R2k1/4P3/3q4/3r4/8/8/8/6K1 b - - 0 1", "d6d8", -375},
                {"3R2k1/4P3/3r4/3q4/8/8/8/6K1 b - - 0 1", "d6d8", 100},
                {"4k3/8/8/4p3/2N5/8/8/4K3 w - - 0 1",     "c4e5", 100},
                {"4k3/3p4/4p3/3P4/8/8/8/4K3 w - - 0 1",   "d5e6", 0},
                {"4k3/8/2p1p3/3p4/8/8/8/3QK3 w - - 0 1",  "d1d5", -875},
                {"3r3k/8/8/3p4/8/8/3R4/3RK3 w - - 0 1",   "d2d5", 100},
                {"4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1",     "d5e6", 100},
                {"4k3/P7/8/8/8/8/8/4K3 w - - 0 1",        "a7a8q", 875},
                {"4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1",      "g7h8q", 1375},
                {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  "e1g1", 0},
            };
            const int numCases = sizeof(CASES) / sizeof(CASES[0]);
            int fails = 0;
            for (int ci = 0; ci < numCases; ++ci) {
                const Case& c = CASES[ci];
                Board b;
                b.setFen(c.fen);
                Move m = parseMove(b, c.mv);
                int got = m.isNull() ? -99999 : seeValue(b, m);
                bool ok = (got == c.expect);
                if (!ok) {
                    ++fails;
                    std::printf("seetest FAIL: %s %s esperado=%d obtido=%d\n", c.fen, c.mv, c.expect, got);
                }
            }
            std::printf("seetest: %d/%d casos OK\n", numCases - fails, numCases);
        } else if (cmd == "d") {
            // Display board
            for (int rank = 7; rank >= 0; --rank) {
                std::printf("%d  ", rank + 1);
                for (int file = 0; file < 8; ++file) {
                    int sq = rank * 8 + file;
                    PieceType pt = board.pieceOn(sq);
                    if (pt == PieceType::NONE) {
                        std::printf(". ");
                    } else {
                        const char* letters = "pnbrqk";
                        char c = letters[int(pt)];
                        if (board.colorOn(sq) == Color::WHITE) c = std::toupper(c);
                        std::printf("%c ", c);
                    }
                }
                std::printf("\n");
            }
            std::printf("   a b c d e f g h\n");
            std::printf("FEN: %s\n", board.toFen().c_str());
        } else if (cmd == "quit" || cmd == "exit") {
            break;
        }
        std::fflush(stdout);
    }
}
