CXX    = g++
# Flags específicas de x86 só se aplicam em x86 — em ARM (ex.: Android/Termux)
# attacks.cpp já tem fallback de PEXT por software e nnue_net.cpp já deteta
# __ARM_NEON automaticamente, sem flags extra.
ARCH := $(shell uname -m)
ifeq ($(ARCH),$(filter $(ARCH),x86_64 amd64))
  ARCHFLAGS = -mbmi2 -mpopcnt -mavx2 -mfma
else
  ARCHFLAGS =
endif
CXXFLAGS = -std=c++20 -O3 $(ARCHFLAGS) -Wall -Wextra
LDFLAGS  =

# Embedded net (pass NET= to embed)
NET ?= nets/littleindian_1024.napk9

SRC_CORE = src/main.cpp src/attacks.cpp src/board.cpp src/movegen.cpp src/uci.cpp src/search.cpp

# NNUE sources (Phase 2+)
SRC_NNUE = src/napoleon/nnue_net.cpp src/napoleon/embedded_net.cpp src/napoleon/wdl_model.cpp src/napoleon/wdl_brain.cpp

# ── F1 target: core engine, no embedded net (perft only) ─────────────────
f1: $(SRC_CORE) $(SRC_NNUE)
	$(CXX) $(CXXFLAGS) -o littleindian $^

# ── native with embedded net ──────────────────────────────────────────────
native-embed: $(SRC_CORE) $(SRC_NNUE)
	$(CXX) $(CXXFLAGS) -march=native \
	  -DEMBEDDED_NET_PATH=\"$(abspath $(NET))\" \
	  -o littleindian $^

# ── AVX2 build ────────────────────────────────────────────────────────────
avx2-embed: $(SRC_CORE) $(SRC_NNUE)
	$(CXX) $(CXXFLAGS) -mavx2 \
	  -DEMBEDDED_NET_PATH=\"$(abspath $(NET))\" \
	  -o littleindian $^

clean:
	rm -f littleindian littleindian.o

.PHONY: f1 native-embed avx2-embed clean
