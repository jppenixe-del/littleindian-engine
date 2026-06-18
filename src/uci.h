#pragma once
#include "board.h"

namespace uci {
void loop();
void parsePosition(Board& board, const std::string& line);
} // namespace uci
