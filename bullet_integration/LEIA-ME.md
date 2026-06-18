# 🦅 3 cabeças no bullet — SEM CHAOS (a solução que FUNCIONA)

## DESCOBERTA: o "Cycle found" NÃO eram as 3 cabeças. Era a CHAOS HEAD.
Mesmo o Caminho B (1 cabeça + chaos) deu "Cycle found". Porquê? Ter 2 saídas (eval + chaos)
a partilhar o acumulador faz o otimizador do bullet (FusePointwise/SwapOutputs) criar ciclo
com a geometria nova (big 128→16). A SOLUÇÃO: tirar a chaos do treino. A fear/chaos é
MARGINAL (<5 Elo, já tínhamos decidido) — não vale a pena ela partir o treino todo.
A chaos é injetada a ZEROS na .napk9 (o motor lê na mesma; fear neutra).

## Ficheiros a copiar para /mnt/c/data/env/bullet:
  napk9_nochaos.rs        → crates/bullet_lib/src/napk9.rs   (1 saída, SEM chaos)
  napk9_train_nochaos.rs  → examples/napk9_train.rs          (loss só do eval)
  auto_converter_3files.py → onde quiseres (injeta chaos a zeros)

## PASSO 1 — treinar as 3 cabeças (agora SEM ciclo, porque é 1 saída só)
```
cd /mnt/c/data/env/bullet
NAPK_HEAD=big    cargo run --release -p bullet_lib --example napk9_train --features cuda
NAPK_HEAD=small  cargo run --release -p bullet_lib --example napk9_train --features cuda
NAPK_HEAD=bullet cargo run --release -p bullet_lib --example napk9_train --features cuda
```
→ deve arrancar e treinar a 750k/s, SEM "Cycle found" (1 saída = sem ciclo).
   Checkpoints: NAPKa0s_V9_big-100, _small-100, _bullet-100.

## PASSO 2 — juntar os 3 (chaos a zeros automaticamente)
```
python3 auto_converter_3files.py --l1 128 \
    --out /mnt/d/Nap2Siriux/nets/v6/_train_128/npk9_master_128.napk9
```

## PASSO 3 — validar e jogar (eval + netstats + gauntlet vs v5)
```
echo -e "uci\nsetoption name EvalFile value .../npk9_master_128.napk9\nposition fen r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5\neval\nquit" | ./nap2siriux
python3 training/gauntlet.py --novo ./nap2siriux.novo --anterior ./nap2siriux.estavel --jogos 100 --tc 8+0.08
```

## VALIDADO: 3 checkpoints sem chaos → .napk9 (chaos a zeros) → motor carrega (L1=128,
   Chaos Head lida dos zeros, Threats 640×128). SCALE=410, big=L1→bigL2.

## ⚠️ ALTERNATIVA se quiseres tentar a chaos de volta um dia: o teu PRIMEIRO treino (a big
   512→32 + chaos) FUNCIONAVA. A diferença é a geometria. Se quiseres chaos, usa a big
   512→32 (big_size=512 separado do l1) — mas isso desalinha do motor. Por agora: SEM chaos
   (fear marginal, e o treino compila de certeza).

## Trade-off do B (já sabido): bullet/small com acc próprio mas usam o acc da big na .napk9
   → aproximadas; a big está perfeita. Aceitável (LazyNNUE usa a bullet só p/ filtro grosseiro).
