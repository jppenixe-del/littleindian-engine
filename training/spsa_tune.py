#!/usr/bin/env python3
# 🦅 littleindian — SPSA TUNER (recalibrar o motor à escala da NNUE NapK9).
#
#   Adaptado do spsa_tune.py do Nap2Siriux (mesmo motor de fundo, parâmetros
#   diferentes — ver bullet_integration/napk9_train_v10_coda.rs e o commit
#   "Expose search parameters as UCI spin options for SPSA tuning" para a
#   lista real de opções tunable do littleindian).
#
#   PORQUÊ: o littleindian é NNUE-nativo desde o dia 1 (sem herança HCE), mas
#   os parâmetros de busca nascem NEUTROS (docs/CLAUDE.md) — SPSA reafina-os
#   à distribuição real da NapK9, em vez de adivinhar à mão. Mover DEZENAS de
#   parâmetros ao mesmo tempo, em passos pequenos, com muitos jogos, até a
#   busca se alinhar com o que a rede produz — como o Coda/SF/Reckless fazem.
#
#   COMO O SPSA FUNCIONA (Simultaneous Perturbation Stochastic Approximation):
#     A cada iteração:
#       1. gera um vetor aleatório de ±1 (delta) p/ TODOS os params de uma vez
#       2. cria θ+ = θ + c·delta  e  θ- = θ − c·delta  (perturba tudo simultaneamente)
#       3. joga um mini-match θ+ vs θ-  (poucos jogos, ex: 8-16) → resultado y±
#       4. estima o gradiente: g ≈ (y+ − y−) / (2·c·delta)   (uma medição p/ TODOS os params!)
#       5. atualiza: θ = θ + a·g   (na direção que ganha jogos)
#       6. a e c decaem com as iterações (a_k = a/(k+A)^α, c_k = c/k^γ)
#     Ao fim de milhares de iterações, θ converge p/ o ótimo conjunto — SEM testar 1 param de cada vez.
#
#   USO (no server, corre dias — usa nohup):
#     nohup python3 training/spsa_tune.py --bin ./littleindian.stable \
#         --iters 20000 --jogos-iter 12 --tc 8+0.08 --paralelo 4 \
#         --grupo eval_scale > training/spsa.log 2>&1 &
#     tail -f training/spsa.log   |   cat training/spsa_state.json
#
#   GRUPOS (--grupo): subconjuntos de params p/ focar o tuning (mais rápido a
#   convergir que afinar tudo de uma vez):
#     eval_scale  → margens que escalam DIRETAMENTE com a eval (RFP/NMP/razor/FP/SEE/probcut/SE)
#     lmr         → forma da redução (LmrCx100, LmrMinDepth, LmrMinMoves)
#     depths      → thresholds de profundidade de cada poda
#     pruning     → eval_scale + lmr + depths
#     all         → todos os 30
#
#   Requer: pip install chess

import argparse, subprocess, sys, random, os, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import chess, chess.engine
except ImportError:
    print("❌ precisa de python-chess: pip install chess"); sys.exit(1)

# ═══ LIVRO DE ABERTURAS (variedade p/ não overfit — 40 linhas equilibradas) ═══════════════════
OPENINGS = [
    "e2e4 e7e5", "e2e4 c7c5", "d2d4 d7d5", "d2d4 g8f6", "e2e4 e7e6",
    "c2c4 e7e5", "g1f3 d7d5", "e2e4 c7c6", "d2d4 e7e6", "e2e4 d7d5",
    "c2c4 g8f6", "d2d4 f7f5", "e2e4 g7g6", "g1f3 g8f6", "c2c4 c7c5",
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "d2d4 g8f6 c2c4 e7e6",
    "d2d4 d7d5 c2c4 c7c6", "e2e4 e7e6 d2d4 d7d5", "g1f3 g8f6 c2c4 g7g6",
    "e2e4 c7c5 b1c3 b8c6", "d2d4 g8f6 g1f3 g7g6", "c2c4 e7e5 b1c3 g8f6",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6", "e2e4 c7c6 d2d4 d7d5",
    "e2e4 d7d6 d2d4 g8f6", "d2d4 e7e6 c2c4 f8b4", "g1f3 d7d5 g2g3 g8f6",
    "e2e4 g7g6 d2d4 f8g7", "c2c4 c7c5 g1f3 g8f6", "d2d4 f7f5 g2g3 g8f6",
    "e2e4 e7e5 f1c4 g8f6", "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5", "e2e4 c7c5 g1f3 e7e6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4", "d2d4 d7d5 c2c4 e7e6", "g1f3 c7c5 c2c4 b8c6",
    "e2e4 d7d5 e4d5 d8d5",
]

# ═══ GRUPOS DE PARÂMETROS (nomes = opções UCI reais do littleindian) ═════════════════════════
GRUPO_EVAL_SCALE = [
    "RfpMargin", "RazorBase", "RazorMult", "FutilityBase", "FutilityMargin",
    "SeePruneMargin", "AspirationDelta", "ProbcutMargin", "SeMargin",
    "HistPruneMargin", "DeltaMargin", "LmpBase", "LmpMult",
]
GRUPO_LMR = ["LmrCx100", "LmrMinDepth", "LmrMinMoves"]
GRUPO_DEPTHS = [
    "RfpMaxDepth", "NmpMinDepth", "IirMinDepth", "ProbcutMinDepth", "SeMinDepth",
    "LmpMaxDepth", "FutilityMaxDepth", "SeePruneMaxDepth", "HistPruneMaxDepth",
]

def grupo_params(nome, defaults):
    if nome == "eval_scale":
        return [p for p in GRUPO_EVAL_SCALE if p in defaults]
    if nome == "lmr":
        return [p for p in GRUPO_LMR if p in defaults]
    if nome == "depths":
        return [p for p in GRUPO_DEPTHS if p in defaults]
    if nome == "pruning":
        return sorted(set(GRUPO_EVAL_SCALE + GRUPO_LMR + GRUPO_DEPTHS) & set(defaults))
    if nome == "all":
        return sorted(defaults.keys())
    pedidos = [x.strip() for x in nome.split(",") if x.strip()]
    return [p for p in pedidos if p in defaults]

# ═══ leitura dos params do binário ════════════════════════════════════════════════════════════
def ler_defaults(bin_path):
    out = subprocess.run([bin_path], input="uci\nquit\n", capture_output=True, text=True, timeout=30).stdout
    opts = {}
    for line in out.splitlines():
        if line.startswith("option name") and "type spin" in line:
            try:
                parts = line.split()
                name = parts[2]
                d = int(parts[parts.index("default")+1])
                mn = int(parts[parts.index("min")+1])
                mx = int(parts[parts.index("max")+1])
                opts[name] = (d, mn, mx)
            except (ValueError, IndexError):
                pass
    return opts

# ═══ um jogo θ+ vs θ- ══════════════════════════════════════════════════════════════════════════
def jogo(bin_path, optsP, optsM, base_opts, opening, tc_base, tc_inc, p_de_brancas):
    """1 jogo θ+ vs θ-. Devolve 1.0 se θ+ ganha, 0.5 empate, 0.0 se θ- ganha."""
    engP = engM = None
    try:
        engP = chess.engine.SimpleEngine.popen_uci(bin_path, stderr=subprocess.DEVNULL)
        engM = chess.engine.SimpleEngine.popen_uci(bin_path, stderr=subprocess.DEVNULL)
        for eng, ov in ((engP, optsP), (engM, optsM)):
            cfg = dict(base_opts); cfg.update(ov)
            for k, v in cfg.items():
                try: eng.configure({k: v})
                except Exception: pass
        board = chess.Board()
        for mv in opening.split():
            try: board.push_uci(mv)
            except Exception: break
        white, black = (engP, engM) if p_de_brancas else (engM, engP)
        wt = bt = tc_base; winc = binc = tc_inc
        while not board.is_game_over(claim_draw=True):
            eng = white if board.turn == chess.WHITE else black
            t0 = time.time()
            res = eng.play(board, chess.engine.Limit(
                white_clock=wt, black_clock=bt, white_inc=winc, black_inc=binc))
            dt = time.time() - t0
            if board.turn == chess.WHITE: wt = max(0.05, wt - dt + winc)
            else: bt = max(0.05, bt - dt + binc)
            if res.move is None: break
            board.push(res.move)
            if board.fullmove_number > 200: break
        r = board.result(claim_draw=True)
        if r == "1-0": s = 1.0 if p_de_brancas else 0.0
        elif r == "0-1": s = 0.0 if p_de_brancas else 1.0
        else: s = 0.5
        return s
    except Exception:
        return 0.5
    finally:
        for e in (engP, engM):
            try:
                if e: e.quit()
            except Exception: pass

def mini_match(bin_path, optsP, optsM, base_opts, n, tc_base, tc_inc, paralelo, op_offset):
    """Mini-match θ+ vs θ-, cores alternadas. Devolve score médio de θ+ (0..1)."""
    tarefas = [(OPENINGS[(op_offset + i) % len(OPENINGS)], i % 2 == 0) for i in range(n)]
    total = 0.0
    with ThreadPoolExecutor(max_workers=paralelo) as ex:
        futs = [ex.submit(jogo, bin_path, optsP, optsM, base_opts, op, tc_base, tc_inc, pb)
                for op, pb in tarefas]
        for f in as_completed(futs):
            total += f.result()
    return total / n

# ═══ SPSA ═════════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="🦅 SPSA — recalibrar o littleindian à escala da NapK9")
    ap.add_argument("--bin", default="./littleindian.stable", help="binário (já expõe os spin tunables)")
    ap.add_argument("--iters", type=int, default=20000, help="nº de iterações SPSA (corre dias)")
    ap.add_argument("--jogos-iter", type=int, default=12, dest="jogos_iter",
                    help="jogos por iteração (θ+ vs θ-); 8-16 típico")
    ap.add_argument("--tc", default="8+0.08", help="time control base+inc (segundos)")
    ap.add_argument("--paralelo", type=int, default=4, help="jogos em paralelo (cuidado com CPU partilhado)")
    ap.add_argument("--options", default="Hash=16", help="opções base fixas (k=v,k=v)")
    ap.add_argument("--grupo", default="eval_scale",
                    help="eval_scale|lmr|depths|pruning|all|<lista k,k,k> — que params tunar")
    ap.add_argument("--estado", default="training/spsa_state.json", help="ficheiro de estado (retoma)")
    # constantes SPSA (standard: Spall)
    ap.add_argument("--alpha", type=float, default=0.602)
    ap.add_argument("--gamma", type=float, default=0.101)
    ap.add_argument("--a-ratio", type=float, default=0.1, dest="a_ratio",
                    help="A = a_ratio * min(iters,5000) (estabiliza o início)")
    ap.add_argument("--a-gain", type=float, default=2.0, dest="a_gain",
                    help="ganho base do passo SPSA")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"❌ binário não encontrado: {args.bin}"); sys.exit(1)

    tc_base = float(args.tc.split("+")[0]); tc_inc = float(args.tc.split("+")[1])
    base_opts = {}
    for kv in args.options.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1); k = k.strip(); v = v.strip()
            base_opts[k] = (v.lower() == "true") if v.lower() in ("true", "false") else _num(v)

    defaults = ler_defaults(args.bin)
    if not defaults:
        print("❌ o binário não exposes opções type spin — confirma que é um build recente "
              "(commit 'Expose search parameters as UCI spin options')."); sys.exit(1)
    params = grupo_params(args.grupo, defaults)
    if not params:
        print(f"❌ grupo '{args.grupo}' não bate com nenhum param. Disponíveis (amostra): "
              f"{sorted(defaults)[:10]}..."); sys.exit(1)

    P = len(params)
    print(f"🦅 SPSA a recalibrar {P} parâmetros à NapK9  (grupo: {args.grupo})")
    print(f"   binário: {args.bin}  | TC {args.tc} | {args.jogos_iter} jogos/iter | {args.paralelo} paralelo")
    print(f"   base: {base_opts}")
    print(f"   params: {', '.join(params)}\n")

    theta, k0 = {}, 0
    if os.path.exists(args.estado):
        try:
            st = json.load(open(args.estado))
            theta = {p: float(st["theta"][p]) for p in params if p in st.get("theta", {})}
            k0 = int(st.get("iter", 0))
            print(f"↻ retomado do estado: iteração {k0}")
        except Exception as e:
            print(f"⚠️ estado ilegível ({e}), começo do zero")
    for p in params:
        if p not in theta: theta[p] = float(defaults[p][0])

    A = max(10.0, args.a_ratio * min(args.iters, 5000))
    ranges = {p: max(1, defaults[p][2] - defaults[p][1]) for p in params}
    a = float(args.a_gain)

    def clamp(p, v): return max(defaults[p][1], min(defaults[p][2], v))
    def opts_from(theta_vec): return {p: int(round(theta_vec[p])) for p in params}

    t_start = time.time()
    for k in range(k0, args.iters):
        ck = max(1e-6, 1.0 / ((k + 1) ** args.gamma))
        ak = a / ((k + 1 + A) ** args.alpha)
        delta = {p: (1 if random.random() < 0.5 else -1) for p in params}
        cstep = {p: ck * 0.06 * ranges[p] for p in params}
        thetaP = {p: clamp(p, theta[p] + cstep[p] * delta[p]) for p in params}
        thetaM = {p: clamp(p, theta[p] - cstep[p] * delta[p]) for p in params}

        y = mini_match(args.bin, opts_from(thetaP), opts_from(thetaM), base_opts,
                       args.jogos_iter, tc_base, tc_inc, args.paralelo, op_offset=k * args.jogos_iter)
        diff = (y - 0.5) * 2.0
        for p in params:
            g = diff / delta[p]
            theta[p] = clamp(p, theta[p] + ak * ranges[p] * g)

        st = {"iter": k + 1, "theta": theta, "grupo": args.grupo,
              "params": params, "ultimo_y": y, "ts": time.time()}
        json.dump(st, open(args.estado, "w"), indent=1)

        if (k + 1) % 5 == 0 or k == k0:
            dt = time.time() - t_start
            ips = (k + 1 - k0) / max(1e-6, dt)
            eta_h = (args.iters - (k + 1)) / max(1e-6, ips) / 3600
            top = sorted(params, key=lambda p: abs(theta[p] - defaults[p][0]) / ranges[p], reverse=True)[:5]
            mudancas = ", ".join(f"{p}={int(round(theta[p]))}(def {defaults[p][0]})" for p in top)
            print(f"[iter {k+1}/{args.iters}] y={y:.3f}  {ips*3600:.0f} it/h  ETA {eta_h:.1f}h  "
                  f"| maiores Δ: {mudancas}", flush=True)

    print("\n🦅 SPSA terminado. Valores finais:")
    for p in params:
        v = int(round(theta[p]))
        if v != defaults[p][0]:
            print(f"   setoption name {p} value {v}   # era {defaults[p][0]}")
    json.dump({"iter": args.iters, "theta": theta, "params": params, "final": True},
              open(args.estado.replace(".json", "_final.json"), "w"), indent=1)

def _num(s):
    try: return int(s)
    except ValueError:
        try: return float(s)
        except ValueError: return s

if __name__ == "__main__":
    main()
