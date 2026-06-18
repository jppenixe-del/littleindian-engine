# littleindian

**littleindian** é um motor de xadrez UCI escrito do zero, **NNUE-nativo desde o primeiro dia** à volta de um formato de rede neuronal original chamado **NapK9**. Não há herança de avaliação handcrafted (HCE) nem código herdado de outro motor — tabuleiro, geração de lances, busca, tabela de transposição, gestão de tempo e UCI são todos código original.

> **Estado: desenvolvimento inicial.** O motor está nas fases F1/F2 do roteiro (ver [Roteiro](#roteiro) abaixo): a geração de lances está validada por perft e existe uma busca mínima com NNUE, mas as técnicas de busca além de um PVS básico ainda estão a ser adicionadas e validadas uma a uma. Ainda não está pronto para torneio.

Outras línguas: [English](README.md) · [Français](README.fr.md)

---

## Porquê "NNUE-nativo"

A maioria dos motores NNUE enxertou uma rede neuronal numa busca afinada para avaliação handcrafted, mantendo os parâmetros de busca antigos. O littleindian evita isso de propósito: todos os parâmetros de busca (margens, reduções, gestão de tempo) nascem neutros e são afinados por SPSA/SPRT à escala de pontuação da própria rede NapK9, em vez de herdar valores calibrados para outra avaliação. Ver [`docs/CLAUDE.md`](docs/CLAUDE.md) para a justificação completa do desenho.

## Avaliação: NapK9

- Formato próprio, cabeçalho mágico `NAPK9LEB`.
- Feature transformer + acumulador int16 (us/them), 8 *material buckets*, PSQT aditivo, `OUTPUT_SCALE_CP = 408`.
- **Full threats**: `2(lado) × 2(ataque/defesa) × 6(atacante) × 6(vítima) × 64(casa) = 9216` features, além das 22528 features de peças (31744 no total).
- Acumulador atualizado incrementalmente (push no `makeMove`, pop no `unmakeMove`), validado contra uma reconstrução completa (tem de ser bit-a-bit igual — ver `threattest`).
- O gerador de features em Rust usado no treino (`bullet_integration/napk9_v10_features.rs`) e o gerador em C++ (`gatherThreatsFull`) têm sempre de produzir índices idênticos. Isto é verificado com `training/calibra_threats.py`.

A rede em `nets/littleindian_1024.napk9` é trabalho original, treinada com o trainer próprio do projeto em `bullet_integration/`.

## Compilar

A rede é compilada diretamente no binário via `.incbin` — não há leitura de nenhum ficheiro `.napk9` em runtime.

```bash
# só o motor base, sem rede embebida (perft, builds de desenvolvimento)
make f1

# build de release para o CPU local, com a rede embebida
make native-embed NET=nets/littleindian_1024.napk9

# build específico para AVX2
make avx2-embed NET=nets/littleindian_1024.napk9
```

Passar sempre `NET=nets/littleindian_1024.napk9` explicitamente — o valor por defeito do Makefile aponta para outra rede usada durante o desenvolvimento.

## Utilização (UCI)

Comandos UCI padrão (`position`, `go`, `isready`, `setoption`, `quit`) mais alguns específicos do motor:

| Comando | Função |
|---|---|
| `perft N` | Verifica correção/velocidade do gerador de lances até profundidade N |
| `eval` | Avaliação estática da posição atual (NapK9, do ponto de vista de quem joga) |
| `bench` | Busca fixa sobre um conjunto de posições embutido — usado como assinatura de regressão de nós/nps |
| `threattest` | Valida o acumulador incremental contra uma reconstrução completa (tem de reportar `diff = 0`) |
| `d` | Mostra o tabuleiro e o FEN |

## Testar alterações (SPRT)

Nenhuma técnica de busca é mantida sem ganhar Elo, medido com SPRT via [cutechess-cli](https://github.com/cutechess/cutechess) — nunca assumido. Ver [`training/sprt.py`](training/sprt.py):

```bash
# gerar um conjunto de aberturas a partir de um livro Polyglot (uma vez)
python3 training/gen_openings.py --book training/book.bin --out training/openings.epd --n 200 --plies 8

# triagem STC
python3 training/sprt.py --new ./littleindian.candidato --base ./littleindian.estavel --tc 8+0.08

# confirmação LTC do que passou em STC
python3 training/sprt.py --new ./littleindian.candidato --base ./littleindian.estavel --tc 40+0.4
```

Validação do contrato de threats antes de confiar numa rede:

```bash
rustc -O bullet_integration/dump_threats.rs -o dump_threats
python3 training/calibra_threats.py --engine ./littleindian --rust ./dump_threats   # tem de dar 0 divergências
```

## Estrutura do projeto

```
littleindian/
├── src/
│   ├── board.{h,cpp}, attacks.{h,cpp}, movegen.{h,cpp}   # tabuleiro, bitboards, magic attacks
│   ├── search.{h,cpp}, tt.h                              # PVS/ID/qsearch, tabela de transposição
│   ├── uci.{h,cpp}, main.cpp                             # interface UCI
│   └── napoleon/                                          # módulo de avaliação NapK9 (nnue_net, embedded_net)
├── nets/littleindian_1024.napk9   # rede embebida nos binários de release
├── bullet_integration/            # trainer NapK9 (Rust) e material de referência
├── training/                      # harness de SPRT, geração de livro de aberturas, validador de threats
└── docs/                          # justificação de desenho e roteiro de integração
```

## Roteiro

O motor é construído em fases com *gates* — cada fase tem uma condição concreta de aprovação/reprovação antes de avançar para a seguinte. As técnicas de busca propriamente ditas entram uma a uma a partir de [`docs/INTEGRACAO_BLOCOS.md`](docs/INTEGRACAO_BLOCOS.md), cada uma validada por SPRT.

- **F1 — Fundações**: bitboards, geração de lances por magic bitboards, Chess960, perft. ✅
- **F2 — Eval NapK9 + busca mínima**: rede embebida, acumulador incremental, PVS/ID/qsearch/TT. ✅ (em curso: ordenação de lances)
- **F3 — Busca moderna**: ordenação (hash move, MVV-LVA/SEE, killers, history), poda/reduções (RFP, NMP, LMR, futility, LMP, SEE pruning, IIR) — cada uma com gate de SPRT.
- **F4 — Extensões e correção**: singular extensions, ProbCut, correction history, Lazy SMP.
- **F5 — Afinação e release**: SPSA do motor completo, build PGO, suíte de regressão, gauntlet.

Detalhe completo em [`docs/CLAUDE.md`](docs/CLAUDE.md) e [`docs/INTEGRACAO_BLOCOS.md`](docs/INTEGRACAO_BLOCOS.md).

## Licença

O littleindian está licenciado sob a [GNU General Public License v3.0](LICENSE).

Ver [CREDITS.md](CREDITS.md) para ferramentas de terceiros usadas e referências conceptuais creditadas.
