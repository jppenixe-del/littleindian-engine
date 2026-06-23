# littleindian

**littleindian** est un moteur d'échecs UCI écrit à partir de zéro. Il dispose de deux évaluations : un réseau de neurones original (NNUE, format **NapK9**, encore en refonte active) et une évaluation manuelle (HCE) complète, calibrée par Texel tuning, sélectionnable à l'exécution avec `setoption name EvalFile value none`. Plateau, génération de coups, recherche, table de transposition, gestion du temps et UCI sont tous du code original.

> **État : développement précoce.** Le moteur est aux phases F1/F2 de sa feuille de route (voir [Feuille de route](#feuille-de-route) ci-dessous) : la génération de coups est validée par perft et une recherche minimale existe, mais les techniques de recherche au-delà d'un PVS basique sont encore ajoutées et validées une à la fois. Pas encore prêt pour la compétition.

Autres langues : [English](README.md) · [Português](README.pt.md)

---

## Deux évaluations

- **NNUE (NapK9)** : format propriétaire, en refonte active — l'architecture concrète (tailles de couches, buckets, caractéristiques) change entre les sessions d'entraînement, donc elle n'est pas documentée ici en détail pour éviter une information obsolète. Active par défaut quand un réseau est intégré au binaire.
- **HCE** : PSQT + Matériel + Menaces + Mobilité + Sécurité du roi + Structure de pions, tous calibrés ensemble par Texel tuning sur des positions réelles (binpacks au format Stockfish). Active avec `setoption name EvalFile value none`. Utilisée comme référence de comparaison et dans des expériences de calibration menées en parallèle de celles du réseau.

Tous les paramètres de recherche (marges, réductions, gestion du temps) sont ajustés par SPSA/SPRT, et non hérités d'un autre moteur. Voir [`docs/CLAUDE.md`](docs/CLAUDE.md) pour la justification complète de la conception.

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
| `eval` | Évaluation statique de la position actuelle, du point de vue du joueur au trait (NNUE seulement) |
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
