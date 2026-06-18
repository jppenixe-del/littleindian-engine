#include "oracle.h"
#include "../board.h"
#include "../uci/move.h"
#include "../movegen.h"
#include <cstdio>

namespace napoleon::oracle
{

static std::unordered_map<uint64_t, std::string> g_map;
static bool g_built = false;
static bool g_enabled = false;  // 🦅 OFF par défaut : le mode combat (CombatMode) l'active

void setEnabled(bool on) { g_enabled = on; }
bool isEnabled() { return g_enabled; }

// uint16 → chaîne de 4-5 caractères (src/dst en notation algébrique)
static std::string decode(uint16_t v)
{
    int f = v & 0x3F;
    int t = (v >> 6) & 0x3F;
    int promo = (v >> 12) & 0x7;
    char buf[6];
    buf[0] = 'a' + (f & 7);
    buf[1] = '1' + (f >> 3);
    buf[2] = 'a' + (t & 7);
    buf[3] = '1' + (t >> 3);
    int n = 4;
    const char* pc = " nbrq";
    if (promo >= 1 && promo <= 4) { buf[4] = pc[promo]; n = 5; }
    buf[n] = '\0';
    return std::string(buf, n);
}

static void build()
{
    if (g_built) return;
    g_built = true;

    size_t count = 0;
    const uint16_t* D = rawData(count);

    size_t i = 0;
    while (i < count && D[i] != 0xFFFF)
    {
        int len = D[i++];
        if (len <= 0 || i + len > count) break;

        Board board;
        board.setToFen(Board::defaultFen, false);

        for (int k = 0; k < len; ++k)
        {
            uint16_t v = D[i + k];
            std::string uci = decode(v);

            uint64_t key = board.zkey().value;
            MoveList legal;
            genMoves<MoveGenType::LEGAL>(board, legal);
            uci::MoveStrFind find = uci::findMoveFromUCI(board, legal, uci.c_str());
            if (find.result != uci::MoveStrFind::Result::FOUND) break;

            if (g_map.find(key) == g_map.end())
                g_map[key] = uci;

            board.makeMove(find.move);
        }
        i += len;
    }
    std::fprintf(stderr, "🦅 [Oracle] %zu signatures indexees\n", g_map.size());
}

std::string query(const Board& board)
{
    if (!g_enabled) return "";
    if (!g_built) build();

    auto it = g_map.find(board.zkey().value);
    if (it == g_map.end()) return "";

    MoveList legal;
    genMoves<MoveGenType::LEGAL>(board, legal);
    uci::MoveStrFind find = uci::findMoveFromUCI(board, legal, it->second.c_str());
    if (find.result != uci::MoveStrFind::Result::FOUND) return "";

    return it->second;
}

} // namespace napoleon::oracle
