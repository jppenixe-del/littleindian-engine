# CLAUDE.md — littleindian: motor de xadrez NNUE‑nativo (do zero)

> **Documento de contexto para o Claude Code.**
> **Arranque de raiz.** `littleindian` é um motor UCI **escrito do zero**, **NNUE‑nativo** à volta do formato **NapK9** (propriedade do projeto). **Não tem base Sirius nem qualquer herança HCE** — abandonou‑se o enxerto NNUE‑sobre‑HCE precisamente pelos vícios do HCE (search e parâmetros mistuned para a escala handcrafted).
> Objetivo: motor forte para **teste de força** — motor‑contra‑motor (`cutechess-cli`), SPRT/gauntlet e listas tipo CCRL.

---

## 0. Princípios do arranque de raiz

1. **Código original.** Board, movegen, search, TT, time‑management e UCI são escritos de novo. Ideias podem vir de conhecimento público (Chess Programming Wiki) e de motores open‑source **a título de referência conceptual** — mas o código é nosso e, se alguma referência for usada, respeita‑se a sua licença e credita‑se. Nada de copiar código sem licença.
2. **NNUE desde o primeiro dia.** A avaliação é **NapK9 (rede 1024, full threats)** de raiz. **Não há eval HCE** — nem como fallback. Isto evita herdar margens/reduções calibradas para outra escala.
3. **Propriedade do projeto (carrega‑se tal e qual):** o **formato NapK9**, a **rede `nets/littleindian_1024.napk9`**, a definição de **full threats**, e o **trainer V10** em `bullet_integration/`. Tudo isto é trabalho original e mantém‑se.
4. **Tudo o que entra é testado** (SPRT). Como se começa do zero, os parâmetros de search **não são herdados de ninguém**: nascem com valores neutros e afinam‑se por SPSA à escala da NNUE.

### Fora do âmbito (não criar nem ligar)
`webpanel/`, `tampermonkey/`, `napoleon_bot.py`, configs de bot, qualquer overlay de assistência em jogo ao vivo / evasão de deteção. Este motor é um UCI normal para GUIs e `cutechess-cli`.

---

## 1. O que se reaproveita (vosso) vs. o que se escreve de novo

| Componente | Origem | Ação |
|---|---|---|
| Formato NapK9 (descodificador/forward) | **Vosso** | Reaproveitar `src/napoleon/nnue_net.{h,cpp}`, `embedded_net.cpp` como ponto de partida do módulo de eval |
| Rede 1024 + full threats | **Vosso** | `nets/littleindian_1024.napk9`, embebida no binário |
| Gerador de full threats (contrato treino↔motor) | **Vosso** | `bullet_integration/napk9_v10_features.rs` ↔ `gatherThreatsFull` (C++) |
| Trainer V10 + pipeline de dados | **Vosso** | `bullet_integration/`, `training/` |
| Board / bitboards / Zobrist | **Novo** | Escrever original |
| Move generation (magic bitboards) | **Novo** | Escrever original, validar por Perft |
| Search (PVS, ID, qsearch, poda, TT) | **Novo** | Escrever original, NNUE‑first |
| Time management | **Novo** | Escrever original |
| UCI | **Novo** | Escrever original (`id name littleindian`) |

> Nota: se algum ficheiro reaproveitado arrastar resquícios da base antiga, **limpar**. O objetivo é um motor sem dependência de código de terceiros sem licença.

---

## 2. Arquitetura alvo (módulos a criar)

```
littleindian/
├── src/
│   ├── core/        # bitboard.h, board.{h,cpp}, zobrist.h, attacks.{h,cpp}
│   ├── movegen/     # magic bitboards, gerador pseudo-legal + legalidade, Chess960
│   ├── search/      # search.cpp (PVS/ID/qsearch), pruning, move_ordering, history, tt
│   ├── eval/        # APENAS NapK9 — nnue_net.{h,cpp}, embedded_net.cpp, threats, accumulator
│   ├── time/        # time_man
│   ├── uci/         # uci.cpp, opções, comandos de debug
│   └── syzygy/      # (opcional, fase tardia)
├── nets/            # littleindian_1024.napk9
├── bullet_integration/   # trainer V10 (referência da rede)
├── training/             # validação, datagen, SPSA, gauntlet
└── Makefile
```

---

## 3. Núcleo de avaliação — NapK9 (vosso, NNUE‑nativo)

- Formato próprio, magic **`NAPK9LEB`**. Rede deste motor: **L1 = 1024, full threats ON**.
- Arquitetura: feature transformer + acumulador int16 (us/them), **8 material buckets** `(piece_count−1)/4`, **PSQT** aditivo, **OUTPUT_SCALE_CP = 408**.
- **Threats (full, V10):** `2(side) × 2(ataque/defesa) × 6(atacante) × 6(vítima) × 64(casa) = 9216`, **offset 22528** (22528 peças + 9216 threats = **31744**). Flag `fullThreats` ON.
- **Acumulador incremental** na pilha do estado de busca: push no makeMove (add/sub das features que mudam), pop no unmakeMove. O `evaluate` lê o acumulador pronto. Validar contra reconstrução completa (deve dar 0).
- **Contrato treino↔motor (sagrado):** o gerador de threads em Rust (`napk9_v10_features.rs`) e o `gatherThreatsFull` em C++ têm de produzir **índices idênticos**. Validar com `training/calibra_threats.py` → 0 divergências, mais o self‑test incremental (`threattest`).
- Forward pass em loops contíguos (auto‑vetorização `-O3`: AVX x86 / NEON ARM), sem intrinsics → portável.

---

## 4. Plano de fases (do zero)

> Cada fase tem um *gate*: não se avança sem o cumprir. Como é NNUE‑nativo, a rede entra cedo.

### F1 — Fundações (board + movegen)
- bitboards, Zobrist, attacks; magic bitboards para deslizantes.
- gerador de lances completo (roque/EP/promoções) + Chess960; legalidade.
- esqueleto UCI; comando `perft`.
- **Gate:** Perft correto até prof. 7 nas posições padrão.

### F2 — Eval NapK9 + busca mínima
- Integrar o módulo NapK9 (rede 1024 embebida), acumulador incremental, full threats ON.
- Negamax + alfa‑beta fail‑soft, PVS, iterative deepening, quiescência (capturas+xeques), TT lockless.
- time‑management básico; comandos `eval`, `bench`, `d`, `threattest`.
- **Gate:** joga partidas legais via GUI; `eval` coerente; `calibra_threats.py` = 0 divergências; `bench` reprodutível.

### F3 — Busca moderna (parâmetros nascem neutros)
- Ordenação: hash move, MVV‑LVA/SEE, killers, history + continuation, counter moves.
- Poda/reduções **uma a uma, com SPRT**: RFP → NMP → LMR → futility → LMP → SEE pruning → razoring → IIR → aspiration windows.
- **Gate:** cada técnica mantida só se ganhou Elo (STC, confirmado em LTC).

### F4 — Extensões e correção
- Singular extensions, ProbCut, check extensions, mate distance pruning.
- Correction history (peões/menores/não‑peão), escala halfmove.
- Lazy SMP (opção Threads).
- **Gate:** todas as adições passaram SPRT; multithread estável.

### F5 — Afinação à escala NNUE + release
- **SPSA** do motor inteiro (`training/spsa_tune.py`): margens, reduções, escalas de tempo nascem do zero e convergem à distribuição da NapK9. (É aqui que se paga a dívida de não herdar tuning de HCE.)
- Build **PGO + rede embebida**; alvos por CPU.
- Suíte de regressão (`bench` com nó‑conta fixa), gauntlet final.
- **Gate:** binário autónomo, Elo registado.

---

## 5. Build — rede 1024 embebida (obrigatório)

A rede é **compilada no binário** (via `.incbin` em `embedded_net.cpp`), não lida de disco em runtime.

```bash
# release:
make pgo-embed   NET=nets/littleindian_1024.napk9
# por CPU:
make avx2-embed  NET=nets/littleindian_1024.napk9
make avx512-embed NET=nets/littleindian_1024.napk9
make native-embed NET=nets/littleindian_1024.napk9
```
- Passar **sempre** `NET=nets/littleindian_1024.napk9` (o default do Makefile aponta para outra rede).
- Uma só rede → usar `*-embed`, **não** `*-dual`.
- Verificar autonomia: `hasEmbeddedNet()==true`; `eval`/`bench` correm sem nenhum `.napk9` no diretório.

---

## 6. Teste de força

```bash
# contrato de threats (antes de confiar na rede):
rustc -O bullet_integration/dump_threats.rs -o dump_threats
make native-embed NET=nets/littleindian_1024.napk9
python3 training/calibra_threats.py --engine ./littleindian --rust ./dump_threats   # 0 divergências

# validação da rede:
python3 training/test_napk9.py   --net nets/littleindian_1024.napk9
python3 training/valida_e_elo.py --bin ./littleindian --out relatorio.txt

# regressão antes de promover:
python3 training/gauntlet.py --novo ./littleindian.novo --anterior ./littleindian.estavel --jogos 200 --tc 8+0.08
```
- **SPRT** com `cutechess-cli`: STC (ex.: `8+0.08`) para triagem, LTC (ex.: `40+0.4`) para confirmar; limiares `[0.0, 2.0]`, `alpha=beta=0.05`; livro de aberturas equilibrado.
- A/B típicos: **threats ON vs OFF**, **littleindian vs versão anterior**, **PGO vs não‑PGO**.

---

## 7. Critérios de aceitação
- [ ] Sem código de terceiros sem licença; `CREDITS.md` reflete a realidade (motor original; referências conceptuais creditadas se usadas).
- [ ] Perft correto (CI verde) — invariante antes de medir força.
- [ ] Eval **só NapK9** (sem HCE); rede 1024 embebida, full threats ON.
- [ ] `calibra_threats.py` = 0 divergências; `threattest` bit‑a‑bit OK; acumulador incremental = reconstrução completa.
- [ ] Binário autónomo (`hasEmbeddedNet()==true`, corre sem `.napk9` no disco).
- [ ] `id name littleindian`.
- [ ] Cada técnica de search validada por SPRT; SPSA concluído à escala NNUE.
- [ ] `bench` reprodutível com nó‑conta documentada; gauntlet/SPRT aprovados.

## 8. Invariantes / armadilhas
- **Nunca** mexer só num lado do contrato de threats (Rust vs C++) sem re‑correr `calibra_threats.py`.
- **Não reintroduzir HCE** "para desenrascar" — o motor é NNUE‑nativo por desenho.
- Parâmetros de search **não se copiam** de motores HCE; nascem neutros e afinam‑se por SPSA.
- Perft tem de estar intacto antes de qualquer medição de força.
