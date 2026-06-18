# INTEGRAÇÃO POR BLOCOS — pôr o littleindian "na linha do que de melhor se faz"

> **Para o Claude Code.** Lista das técnicas que motores de topo como o **Reckless** usam, organizadas em **blocos** a integrar por ordem. Objetivo: **jogar bem (força) e rápido (nps + profundidade)**.
>
> **Regra de propriedade/licença:** o Reckless é **AGPL** e a sua rede tem licença própria. **Não se copia código nem rede.** Implementa‑se cada técnica em **código original** — são conceitos públicos (Chess Programming Wiki, comunidade Engine Programming). O nosso motor é NNUE‑nativo com NapK9; só importamos *ideias de busca/representação*.
>
> **Disciplina:** integrar **um bloco de cada vez**, e dentro do bloco **uma técnica de cada vez**, cada uma validada por **SPRT** (STC para triagem, LTC para confirmar). Manter só o que ganha Elo. Os parâmetros nascem neutros e afinam‑se por SPSA (não se herdam de motores HCE).
>
> As estimativas de Elo abaixo são aproximadas (valores públicos, ex.: testes do Sirius); servem só para **priorizar**, variam de motor para motor.

---

## Bloco 1 — Representação e geração (VELOCIDADE primeiro)
Base de tudo: quanto mais rápido o movegen, mais fundo a busca vai no mesmo tempo.

| Técnica | O que faz | Nota |
|---|---|---|
| Bitboards (Little‑Endian Rank‑File) | representação do tabuleiro em u64 | obrigatório |
| Fancy Magic Bitboards | ataques de peças deslizantes por lookup | obrigatório (grande ganho de nps) |
| Magics pré‑gerados | constantes mágicas fixas, não procuradas em runtime | acelera arranque |
| Geração de mapas de ataque em *compile time* (build script) | tabelas prontas no binário | menos trabalho em runtime |
| Movegen pseudo‑legal + filtro de legalidade | gera rápido, valida só o necessário | padrão |
| Copy‑Make **vs** Make/Unmake | gestão de estado por lance | **A/B**: testar os dois, ficar com o mais rápido |
| Zobrist hashing | chave da posição (+ chaves de peões/menores/não‑peão p/ correction history) | obrigatório |

**Gate:** Perft 100% correto até prof. 7. Medir nps base.

---

## Bloco 2 — Núcleo de busca
O esqueleto alfa‑beta que tudo o resto refina.

| Técnica | O que faz | Elo aprox. |
|---|---|---|
| Fail‑Soft Alpha‑Beta | poda básica com janelas | núcleo |
| Principal Variation Search (PVS) | re‑pesquisa só quando preciso | núcleo |
| Iterative Deepening | profundidade 1→N, reutiliza info | núcleo |
| Aspiration Windows | janela estreita à volta do score anterior | **~+100** |
| Quiescence Search | resolve capturas/xeques nas folhas | **~+180** |
| Mate Distance Pruning | encurta linhas de mate | pequeno |

**Gate:** joga partidas legais; `bench` reprodutível.

---

## Bloco 3 — Transposição e ordenação de lances
Ordenar bem é o que faz a poda funcionar. Alto retorno.

| Técnica | O que faz | Elo aprox. |
|---|---|---|
| Transposition Table lockless | cache de posições (SMP‑safe) | **~+100** (TT cutoffs) |
| Hash Move | tenta primeiro o lance da TT | grande |
| MovePicker em fases (staged) | gera/ordena por etapas, evita custo | nps |
| MVV‑LVA + SEE ordering | ordena capturas por ganho | grande |
| Killer Moves | 2 lances tranquilos que cortaram antes | médio |
| History Heuristic | bónus a lances que cortam | grande |
| Continuation History (1/2/4‑ply) | history condicionado aos lances anteriores | grande |
| Capture History | history para capturas | médio |

**Gate:** ordenação melhora nps e profundidade no `bench`; cada item por SPRT.

---

## Bloco 4 — Poda e reduções (o MAIOR salto de força)
Onde se ganha mais Elo. Introduzir **uma a uma**, sempre com SPRT. Parâmetros neutros → SPSA depois.

| Técnica | O que faz | Elo aprox. |
|---|---|---|
| Reverse Futility Pruning (RFP) | corta nós com eval muito acima de beta | grande |
| Null Move Pruning (NMP) | passa a vez; se ainda ganha, corta | grande |
| Razoring | cai em qsearch quando eval muito abaixo de alfa | médio |
| Futility Pruning | corta quietos sem esperança perto das folhas | médio |
| Late Move Pruning (LMP) | ignora lances tardios em depth baixo | grande |
| Delta Pruning (em qsearch) | corta capturas inúteis na quiescência | médio |
| SEE Pruning | corta lances com troca perdedora | **~+250** (SEE no geral) |
| Fractional Late Move Reductions (LMR) | reduz profundidade de lances tardios (tabela log) | **muito grande** |
| Internal Iterative Reductions (IIR) | reduz quando não há hash move | ~+8 |
| History Pruning | corta quietos com history muito negativo | médio |

**Gate:** cada técnica mantida só se passou SPRT (STC + LTC).

---

## Bloco 5 — Extensões e busca seletiva
Aprofundar onde interessa, sem rebentar o tempo.

| Técnica | O que faz | Elo aprox. |
|---|---|---|
| Check Extensions | +1 ply quando em xeque | médio |
| Singular Extensions | estende o único lance bom (com multicut, double/negative ext.) | grande |
| ProbCut | salta sub‑árvores com prova rápida acima de beta | médio |

**Gate:** SPRT por extensão; cuidado com explosão de árvore.

---

## Bloco 6 — Correção da avaliação (lado NNUE, já tens NapK9)
A rede já avalia; isto **corrige** o score com a história da partida. Alto retorno, barato.

| Técnica | O que faz | Elo aprox. |
|---|---|---|
| Correction History (peões / menores / não‑peão) | ajusta a eval por erros sistemáticos vistos antes | **~+100** |
| Material optimism | enviesa o score com base no material | médio |
| Escala pelo halfmove clock | aproxima da regra dos 50 lances | pequeno |
| TT eval adjustment | usa o score da TT para afinar a eval estática | ~+8 |

**Nota:** os *output buckets* e o *material scaling* já estão na NapK9 — confirmar que entram no caminho de eval.

---

## Bloco 7 — Velocidade da avaliação (SIMD + incremental)
Para "jogar rápido": a eval é o ponto quente. Isto multiplica nps.

| Técnica | O que faz | Nota |
|---|---|---|
| Acumulador incremental | atualiza só as features que o lance muda (push/pop) | já previsto; validar vs reconstrução completa (=0) |
| Finny / lazy updates | adia/agrupa atualizações do acumulador | nps |
| Forward pass vetorizável (AVX2/AVX‑512/NEON) | loops contíguos, auto‑vetorização `-O3` | grande em nps |
| Quantização correta (int16/int8) | escalas QA/QB certas | precisão + velocidade |

**Gate:** nps sobe sem mudar o resultado da eval (bit‑exato onde aplicável).

---

## Bloco 8 — Paralelismo e gestão de tempo
Escalar com cores e gastar o tempo onde conta. "Decidir bem o tempo" vale Elo real.

| Técnica | O que faz | Elo aprox. |
|---|---|---|
| Lazy SMP | múltiplas threads partilham a TT | escala com cores |
| Soft/Hard time limits | limite flexível + teto absoluto | **~+60** (TM no geral) |
| Node‑count time management | aloca por nós, não só por relógio | parte do TM |
| Best‑move stability | pára cedo se o melhor lance estabiliza | parte do TM |
| MultiPV | N variações (análise; off por defeito em jogo) | utilidade |

**Gate:** escalabilidade medida; sem perda de força em 1 thread.

---

## Bloco 9 — Build e performance final
O último 5–10% de velocidade vem daqui.

| Técnica | O que faz | Nota |
|---|---|---|
| PGO (Profile‑Guided Optimization) | recompila com perfil do `bench` | grande em nps |
| Dispatch por CPU (`target-cpu`, AVX2/512/NEON) | binário tira partido do hardware | velocidade |
| Rede 1024 embebida (`.incbin`) | binário autónomo | obrigatório (ver CLAUDE.md) |
| `bench` com nó‑conta fixa | assinatura de regressão por commit | qualidade |

**Gate:** `make pgo-embed NET=nets/littleindian_1024.napk9` estável; nps de release documentado.

---

## Ordem recomendada e ligação ao plano
- Blocos **1→2→3→4** dão a maior parte da força e devem vir cedo (mapeiam às fases F1–F3 do `CLAUDE.md`).
- Blocos **5→6** acrescentam o "joga bem" fino (F4).
- Blocos **7→8→9** são sobretudo "joga rápido" + afinação (F5), embora o incremental/SIMD (B7) deva entrar mal a eval esteja ligada.
- **Depois de tudo:** campanha **SPSA** a mover dezenas de parâmetros à escala da NapK9.

## Regras que não mudam
1. **Código original** — Reckless é AGPL; importa‑se a ideia, não o código nem a rede.
2. **Um bloco/uma técnica de cada vez, com SPRT.** Sem exceções "óbvias".
3. **Parâmetros neutros → SPSA.** Nada de copiar constantes de motores HCE.
4. **Perft intacto** antes de qualquer medição de força.
5. Camada de overlay/bot em jogo ao vivo continua **fora do âmbito**.
