#pragma once
// ═══════════════════════════════════════════════════════════════════════════
// 🦅 NAP2SIRIUX — PASSED PAWN (masques + bonus) ultra-optimisés bullet
// ═══════════════════════════════════════════════════════════════════════════
//   Idée Maréchal : contrer l'effet d'horizon sur les pions passés vs Stockfish.
//   - passedPawnMask(color, sq) : masque bitwise (colonne + adjacentes, devant)
//     pré-calculé en table statique → ZÉRO coût runtime (juste un lookup).
//   - passedPawnBonus(relRank)  : bonus exponentiel par rang (massif en 7).
//   - unblockedPenalty(...)     : pénalité si pion passé ennemi sans bloqueur.
//
//   Tout est constexpr / table statique : pas de calcul dans le hot path.
// ═══════════════════════════════════════════════════════════════════════════
#include "../defs.h"
#include "../bitboard.h"

namespace napoleon::passedpawn
{

// ── Table des masques de pion passé, indexée [color][square] ──
// Pour un pion en `sq` de couleur `c`, le masque couvre la colonne de `sq`
// + les colonnes adjacentes, sur TOUS les rangs DEVANT le pion (vers la
// promotion). Si aucun pion ennemi n'est dans ce masque → pion passé.
struct PassedMaskTable
{
    Bitboard mask[2][64];

    constexpr PassedMaskTable() : mask{}
    {
        for (i32 c = 0; c < 2; ++c)
        {
            for (i32 s = 0; s < 64; ++s)
            {
                const i32 file = s & 7;
                const i32 rank = s >> 3;
                u64 m = 0ull;

                // colonnes : celle du pion + adjacentes (si existent)
                for (i32 df = -1; df <= 1; ++df)
                {
                    const i32 f = file + df;
                    if (f < 0 || f > 7) continue;
                    // rangs DEVANT le pion (sens dépend de la couleur)
                    if (c == 0) // WHITE avance vers rang 7
                    {
                        for (i32 r = rank + 1; r <= 7; ++r)
                            m |= (1ull << (r * 8 + f));
                    }
                    else        // BLACK avance vers rang 0
                    {
                        for (i32 r = rank - 1; r >= 0; --r)
                            m |= (1ull << (r * 8 + f));
                    }
                }
                mask[c][s] = Bitboard(m);
            }
        }
    }
};

inline constexpr PassedMaskTable PASSED_TABLE{};

// Masque de pion passé (lookup pur, zéro calcul) — pour search ET éval.
inline Bitboard passedPawnMask(Color c, Square sq)
{
    return PASSED_TABLE.mask[static_cast<i32>(c)][sq.value()];
}

// ── Bonus EXPONENTIEL par rang relatif (idée Maréchal) ──
// Rang relatif : 0..7 du point de vue du camp du pion. Bonus en centipawns,
// modéré en rang 4-5, fort en 6, massif en 7 (1 case de la promotion).
// Indexé par relativeRank() ; valeurs calibrées échelle Sirius (~100cp/pion).
inline constexpr i32 PASSED_BONUS[8] = {
    0,    // rang 1
    8,    // rang 2
    20,   // rang 3
    45,   // rang 4
    90,   // rang 5  (modéré, renforcé)
    175,  // rang 6  (fort, renforcé)
    340,  // rang 7  (massif : à 1 case de Dame, renforcé)
    0     // rang 8 = promotion
};

inline constexpr i32 passedPawnBonus(i32 relRank)
{
    return PASSED_BONUS[relRank & 7];
}

// Pénalité supplémentaire pour un pion passé ENNEMI sans bloqueur devant
// (aucune de nos pièces sur la case immédiatement devant le pion).
// On l'applique en plus du bonus inversé → double peine pour le danger réel.
inline constexpr i32 UNBLOCKED_EXTRA[8] = {
    0, 0, 0, 15, 40, 85, 180, 0   // malus renforcé si rang 6-7 non bloqué
};

inline constexpr i32 unblockedExtra(i32 relRank)
{
    return UNBLOCKED_EXTRA[relRank & 7];
}

// ── Évaluation des pions passés (POV side-to-move), ultra-rapide ──
// Parcourt les pions des deux camps, détecte les passés via masque bitwise,
// applique le bonus exponentiel. Pénalise en plus les passés ENNEMIS non
// bloqués (aucune pièce à nous devant). Retourne un delta cp côté STM.
// 🦅 RENFORCÉ : règle du carré (pion inarrêtable = bonus/malus massif),
// prise en compte du trait (tempo décisif dans une course de pions).
// Conçu pour le hot path : lookups de table + popcount, pas d'allocation.

// Règle du carré : un pion passé est "inarrêtable" par le roi ennemi seul si
// le roi ne peut pas entrer dans le carré de promotion. Calcul O(1) en cases.
// pawnSq : case du pion ; promoRank : rang de promotion (7 blanc, 0 noir) ;
// kingSq : roi ennemi ; sideToMoveIsPusher : true si c'est au camp du pion de jouer.
inline bool pawnIsUnstoppable(i32 pawnSq, Color pawnColor, i32 enemyKingSq,
                              bool pusherToMove)
{
    const i32 pf = pawnSq & 7, pr = pawnSq >> 3;
    const i32 kf = enemyKingSq & 7, kr = enemyKingSq >> 3;
    // distance du pion à la promotion (en cases)
    i32 promoRank = (pawnColor == Color::WHITE) ? 7 : 0;
    i32 pawnDist = (pawnColor == Color::WHITE) ? (7 - pr) : pr;
    // un pion en 2e rang relatif peut avancer de 2 → réduit la distance
    i32 startRank = (pawnColor == Color::WHITE) ? 1 : 6;
    if (pr == startRank) pawnDist -= 1;
    // si ce n'est pas au pousseur de jouer, le roi gagne un tempo
    i32 kingDist = std::max(std::abs(kf - pf), std::abs(kr - promoRank));
    i32 effKingDist = kingDist - (pusherToMove ? 0 : 1);
    // pion gagne la course si le roi ne peut pas atteindre le carré à temps
    return pawnDist < effKingDist;
}

template <typename BoardT>
inline i32 passedPawnScore(const BoardT& board, Color stm)
{
    const Color them = ~stm;
    const Bitboard ourPawns   = board.pieces(stm,  PieceType::PAWN);
    const Bitboard theirPawns = board.pieces(them, PieceType::PAWN);
    const Bitboard allOurs     = board.pieces(stm);
    const Bitboard allTheirs   = board.pieces(them);
    const i32 ourKing   = board.kingSq(stm).value();
    const i32 theirKing = board.kingSq(them).value();

    // Compte des pièces lourdes/légères de chaque camp (0 = finale de pions pure).
    // La règle du carré "stricte" ne vaut que sans pièces, MAIS un pion très
    // avancé (6e/7e) reste dangereux AVEC pièces : on garde un malus progressif.
    const i32 enemyPieces =
        (board.pieces(them, PieceType::QUEEN) | board.pieces(them, PieceType::ROOK)
         | board.pieces(them, PieceType::BISHOP) | board.pieces(them, PieceType::KNIGHT)).popcount();
    const i32 ourPieces =
        (board.pieces(stm, PieceType::QUEEN) | board.pieces(stm, PieceType::ROOK)
         | board.pieces(stm, PieceType::BISHOP) | board.pieces(stm, PieceType::KNIGHT)).popcount();

    // Masque des colonnes contenant un pion passé (pour détecter les connectés).
    auto fileMaskOf = [](i32 sq) -> u64 {
        const u64 FA = 0x0101010101010101ull;
        return FA << (sq & 7);
    };

    i32 score = 0;
    u64 ourPassedFiles = 0, theirPassedFiles = 0;
    i32 ourMostAdvanced = -1, theirMostAdvanced = -1;   // rang relatif max

    // ── Nos pions passés → bonus ──
    Bitboard bb = ourPawns;
    while (bb.any())
    {
        const Square sq = bb.poplsb();
        if ((theirPawns & passedPawnMask(stm, sq)).empty())
        {
            const i32 rr = sq.relativeRank(stm);
            score += passedPawnBonus(rr);
            ourPassedFiles |= fileMaskOf(sq.value());
            if (rr > ourMostAdvanced) ourMostAdvanced = rr;

            const i32 fwd = (stm == Color::WHITE) ? sq.value() + 8 : sq.value() - 8;
            if (fwd >= 0 && fwd < 64)
            {
                const Bitboard front = Bitboard::fromSquare(Square(fwd));
                if ((allOurs & front).empty() && (allTheirs & front).empty())
                    score += passedPawnBonus(rr) / 4;
            }
            // 🦅 RÈGLE DU CARRÉ stricte (finale sans pièces) → bonus massif
            if (enemyPieces == 0 &&
                pawnIsUnstoppable(sq.value(), stm, theirKing, true))
                score += 600;
            // 🦅 NOUVEAU : pion très avancé (6e/7e) dangereux MÊME avec pièces.
            // Bonus dégressif selon le nb de pièces ennemies (moins de pièces =
            // plus dur à arrêter). Résout le "rouleau" en milieu/finale.
            else if (rr >= 5)
                score += (rr == 6 ? 120 : 60) / (1 + enemyPieces);
        }
    }

    // ── Pions passés ENNEMIS → malus (symétrique, c'est LE danger pour toi) ──
    bb = theirPawns;
    while (bb.any())
    {
        const Square sq = bb.poplsb();
        if ((ourPawns & passedPawnMask(them, sq)).empty())
        {
            const i32 rr = sq.relativeRank(them);
            score -= passedPawnBonus(rr);
            theirPassedFiles |= fileMaskOf(sq.value());
            if (rr > theirMostAdvanced) theirMostAdvanced = rr;

            const i32 fwd = (them == Color::WHITE) ? sq.value() + 8 : sq.value() - 8;
            if (fwd >= 0 && fwd < 64)
            {
                const Bitboard front = Bitboard::fromSquare(Square(fwd));
                if ((allOurs & front).empty())
                    score -= unblockedExtra(rr);
            }
            if (ourPieces == 0 &&
                pawnIsUnstoppable(sq.value(), them, ourKing, false))
                score -= 600;
            else if (rr >= 5)
                score -= (rr == 6 ? 120 : 60) / (1 + ourPieces);
        }
    }

    // 🦅 Early-out perf : si aucun pion passé des deux côtés, rien à calculer
    // pour les rouleaux/courses → on évite les popcount et lambdas (hot path).
    if (!ourPassedFiles && !theirPassedFiles)
        return score;

    // 🦅 PIONS PASSÉS CONNECTÉS + ROULEAU (la signature d'attaque adverse).
    // Détection bitwise des colonnes passées ayant un voisin passé (connectés).
    auto connectedCount = [](u64 files) -> i32 {
        u64 adj = ((files & ~0x0101010101010101ull) >> 1)
                | ((files & ~0x8080808080808080ull) << 1);
        return __builtin_popcountll(files & adj);
    };
    const i32 ourConn   = connectedCount(ourPassedFiles);
    const i32 theirConn = connectedCount(theirPassedFiles);

    // Connectés : bonus MODÉRÉ (pions connectés se soutiennent, mais ce n'est
    // pas décisif en soi). Plafonné pour ne JAMAIS dominer l'éval matérielle.
    score += (15 + 5 * std::max(0, ourMostAdvanced - 3)) * ourConn;
    score -= (15 + 5 * std::max(0, theirMostAdvanced - 3)) * theirConn;

    // 🦅 COURSE DE PIONS (qui promeut en premier) — modéré, seulement si les
    // deux camps ont des passés avancés et que la position se simplifie.
    if (ourMostAdvanced >= 5 && theirMostAdvanced >= 5
        && enemyPieces <= 1 && ourPieces <= 1)
    {
        i32 raceDelta = (ourMostAdvanced - theirMostAdvanced);
        score += raceDelta * 40;   // tempo de promotion, borné
    }

    return score;
}

} // namespace napoleon::passedpawn
