# littleindian

**littleindian** est un moteur d'échecs UCI écrit à partir de zéro, **NNUE-natif dès le premier jour**, autour d'un format de réseau neuronal original appelé **NapK9**. Il n'y a aucun héritage d'évaluation manuelle (HCE) ni de code repris d'un autre moteur — plateau, génération de coups, recherche, table de transposition, gestion du temps et UCI sont tous du code original.

> **État : développement précoce.** Le moteur est aux phases F1/F2 de sa feuille de route (voir [Feuille de route](#feuille-de-route) ci-dessous) : la génération de coups est validée par perft et une recherche minimale appuyée sur le NNUE existe, mais les techniques de recherche au-delà d'un PVS basique sont encore ajoutées et validées une à la fois. Pas encore prêt pour la compétition.

Autres langues : [English](README.md) · [Português](README.pt.md)

---

## Pourquoi « NNUE-natif »

La plupart des moteurs NNUE ont greffé une évaluation par réseau de neurones sur une recherche réglée pour une évaluation manuelle, en conservant les anciens paramètres de recherche. littleindian évite cela délibérément : tous les paramètres de recherche (marges, réductions, gestion du temps) démarrent neutres et sont ajustés par SPSA/SPRT à l'échelle de score propre au réseau NapK9, plutôt que d'hériter de valeurs calibrées pour une autre évaluation. Voir [`docs/CLAUDE.md`](docs/CLAUDE.md) pour la justification complète de la conception.

## Évaluation : NapK9

- Format propriétaire, en-tête magique `NAPK9LEB`.
- Feature transformer + accumulateur int16 (us/them), 8 *material buckets*, PSQT additif, `OUTPUT_SCALE_CP = 408`.
- **Full threats** : `2(camp) × 2(attaque/défense) × 6(attaquant) × 6(victime) × 64(case) = 9216` caractéristiques, en plus des 22528 caractéristiques de pièces (31744 au total).
- Accumulateur mis à jour de façon incrémentale (push sur `makeMove`, pop sur `unmakeMove`), validé contre une reconstruction complète (doit être identique bit à bit — voir `threattest`).
- Le générateur de caractéristiques en Rust utilisé pour l'entraînement (`bullet_integration/napk9_v10_features.rs`) et celui en C++ (`gatherThreatsFull`) doivent toujours produire des index identiques. Cela est vérifié avec `training/calibra_threats.py`.

Le réseau fourni dans `nets/littleindian_1024.napk9` est un travail original, entraîné avec l'outil d'entraînement propre au projet dans `bullet_integration/`.

## Compilation

Le réseau est compilé directement dans le binaire via `.incbin` — aucun fichier `.napk9` n'est lu au moment de l'exécution.

```bash
# moteur de base uniquement, sans réseau embarqué (perft, builds de développement)
make f1

# build de release pour le CPU local, avec le réseau embarqué
make native-embed NET=nets/littleindian_1024.napk9

# build spécifique AVX2
make avx2-embed NET=nets/littleindian_1024.napk9
```

Toujours passer `NET=nets/littleindian_1024.napk9` explicitement — la valeur par défaut du Makefile pointe vers un autre réseau utilisé pendant le développement.

## Utilisation (UCI)

Commandes UCI standard (`position`, `go`, `isready`, `setoption`, `quit`) plus quelques commandes spécifiques au moteur :

| Commande | Fonction |
|---|---|
| `perft N` | Vérifie la correction/vitesse du générateur de coups jusqu'à la profondeur N |
| `eval` | Évaluation statique de la position actuelle (NapK9, du point de vue du joueur au trait) |
| `bench` | Recherche fixe sur un ensemble de positions intégré — utilisé comme signature de régression nœuds/nps |
| `threattest` | Valide l'accumulateur incrémental contre une reconstruction complète (doit indiquer `diff = 0`) |
| `d` | Affiche le plateau et le FEN |

## Tester les changements (SPRT)

Aucune technique de recherche n'est conservée sans gain d'Elo mesuré par SPRT via [cutechess-cli](https://github.com/cutechess/cutechess) — jamais supposé. Voir [`training/sprt.py`](training/sprt.py) :

```bash
# générer un ensemble d'ouvertures à partir d'un livre Polyglot (une fois)
python3 training/gen_openings.py --book training/book.bin --out training/openings.epd --n 200 --plies 8

# tri STC
python3 training/sprt.py --new ./littleindian.candidat --base ./littleindian.stable --tc 8+0.08

# confirmation LTC de ce qui a passé le STC
python3 training/sprt.py --new ./littleindian.candidat --base ./littleindian.stable --tc 40+0.4
```

Validation du contrat de menaces avant de faire confiance à un réseau :

```bash
rustc -O bullet_integration/dump_threats.rs -o dump_threats
python3 training/calibra_threats.py --engine ./littleindian --rust ./dump_threats   # doit indiquer 0 divergence
```

## Structure du projet

```
littleindian/
├── src/
│   ├── board.{h,cpp}, attacks.{h,cpp}, movegen.{h,cpp}   # plateau, bitboards, magic attacks
│   ├── search.{h,cpp}, tt.h                              # PVS/ID/qsearch, table de transposition
│   ├── uci.{h,cpp}, main.cpp                             # interface UCI
│   └── napoleon/                                          # module d'évaluation NapK9 (nnue_net, embedded_net)
├── nets/littleindian_1024.napk9   # réseau embarqué dans les binaires de release
├── bullet_integration/            # entraîneur NapK9 (Rust) et matériel de référence
├── training/                      # harnais SPRT, génération de livre d'ouvertures, validateur de menaces
└── docs/                          # justification de conception et feuille de route d'intégration
```

## Feuille de route

Le moteur est construit en phases verrouillées — chaque phase a une condition concrète de réussite/échec avant de passer à la suivante. Les techniques de recherche elles-mêmes sont introduites une à la fois à partir de [`docs/INTEGRACAO_BLOCOS.md`](docs/INTEGRACAO_BLOCOS.md), chacune validée par SPRT.

- **F1 — Fondations** : bitboards, génération de coups par magic bitboards, Chess960, perft. ✅
- **F2 — Évaluation NapK9 + recherche minimale** : réseau embarqué, accumulateur incrémental, PVS/ID/qsearch/TT. ✅ (en cours : ordre des coups)
- **F3 — Recherche moderne** : ordre des coups (coup de hachage, MVV-LVA/SEE, killers, history), élagage/réductions (RFP, NMP, LMR, futility, LMP, SEE pruning, IIR) — chacun verrouillé par SPRT.
- **F4 — Extensions et correction** : extensions singulières, ProbCut, correction history, Lazy SMP.
- **F5 — Réglage et sortie** : SPSA du moteur complet, build PGO, suite de régression, gauntlet.

Détails complets dans [`docs/CLAUDE.md`](docs/CLAUDE.md) et [`docs/INTEGRACAO_BLOCOS.md`](docs/INTEGRACAO_BLOCOS.md).

## Licence

littleindian est distribué sous licence [GNU General Public License v3.0](LICENSE).

Voir [CREDITS.md](CREDITS.md) pour les outils tiers utilisés et les références conceptuelles créditées.
