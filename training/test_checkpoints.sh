#!/bin/bash
# 🦅 test_checkpoints.sh — testa cada checkpoint (.li11) convertido em 1-2 posições fixas,
#   usando UM SÓ processo do motor (setoption EvalFile troca a rede em runtime, sem
#   recompilar). Produz uma tabela superbatch -> eval, para ver a evolução do treino.
#
# Uso: ./test_checkpoints.sh <pasta com sbNNN.li11> <binário do motor>

set -euo pipefail
DIR="${1:-nets/li11_checkpoints}"
BIN="${2:-./littleindian.checkpoint_tester}"

POS1="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"   # startpos
POS2="8/8/8/4k3/8/8/4K3/8 w - - 0 1"                              # rei vs rei (devia ser 0)

CMDS=""
SBS=$(ls "$DIR" | sed -n 's/^sb\([0-9]*\)\.li11$/\1/p' | sort -n)
for sb in $SBS; do
    CMDS+="setoption name EvalFile value $(pwd)/$DIR/sb${sb}.li11\n"
    CMDS+="position fen $POS1\n"
    CMDS+="eval\n"
    CMDS+="position fen $POS2\n"
    CMDS+="eval\n"
done
CMDS+="quit\n"

echo -e "$CMDS" | "$BIN" 2>&1 | awk -v sbs="$SBS" '
BEGIN { split(sbs, arr, "\n"); i = 0 }
/^eval:/ {
    n++
    val = $2
    if (n % 2 == 1) { i++; sb = arr[i]; startpos_val = val }
    else { printf "sb=%-4s startpos=%-7s kvk=%-7s\n", sb, startpos_val, val }
}
'
