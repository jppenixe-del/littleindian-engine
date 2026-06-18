#!/usr/bin/env python3
# 🦅 auto_converter_3files V9.7-B — junta os 3 checkpoints do CAMINHO B (cabeças treinadas
#   separadamente) numa só .napk9 de 3 cabeças.
#
#   Cada checkpoint (NAPK_HEAD=big/small/bullet) tem o layout do napk9_train_single.rs:
#     [accw, accb, psqtw, psqtb, {p}_l1w, {p}_l1b, {p}_l2w, {p}_l2b, {p}_l3w, {p}_l3b,
#      chaos_l1w..chaos_l3b]   (p = bullet|small|g)
#
#   ESTRATÉGIA: o .napk9 final usa acc/psqt/chaos da BIG (o motor partilha 1 acc/psqt entre
#   as 3 cabeças; a big é a que decide). As cabeças bullet/small contribuem só com os seus
#   batalhões densos (l1/l2/l3). ⚠️ As cabeças peq. foram treinadas com acc PRÓPRIO ≠ acc da
#   big → ficam aproximadas (ok p/ o explorer do LazyNNUE; a big está perfeita). Para as ter
#   perfeitas, treinar bullet/small com o acc da big CONGELADO (ver LEIA-ME, melhoria futura).
#
#   Uso:
#     python3 auto_converter_3files.py \
#         --big    checkpoints_napk/NAPKa0s_V9_big-100/<ficheiro> \
#         --small  checkpoints_napk/NAPKa0s_V9_small-100/<ficheiro> \
#         --bullet checkpoints_napk/NAPKa0s_V9_bullet-100/<ficheiro> \
#         --out /mnt/d/Nap2Siriux/nets/v6/_train_128/npk9_master_128.napk9 --l1 128
#   (se não passares os caminhos, procura os checkpoints -100 mais recentes de cada cabeça.)

import os, sys, glob, re, struct, argparse
import numpy as np

MAGIC = b"NAPK9LEB"
VERSION = 1
LEB128_CHUNK_MAGIC = b"COMPRESSED_LEB128"
TOTAL_FEATURES = 22528
MATERIAL_BUCKETS = 8
QA = 255
QB = 64

# 🦅 SFNNv13: dimensões fixas da big (afunil, sem camada L1→L1 gorda). = BIG_DL1/BIG_DL2 do motor.
BIG_DL1 = 32
BIG_DL2 = 32

def detect_big_l2(l1):
    # 🦅 SFNNv13: a big afunila L1×2 → 32 → 32 → 1×8. L2 fixo = 32. (mantido p/ refs.)
    return BIG_DL2

def encode_chunk(values):
    val = np.asarray(values).astype(np.int64).flatten()
    n = len(val)
    if n == 0:
        return LEB128_CHUNK_MAGIC + struct.pack('<I', 0)
    zz = np.where(val >= 0, val << 1, ((-val - 1) << 1) | 1).astype(np.uint64)
    bits = np.floor(np.log2(np.maximum(zz, 1))).astype(np.int64) + 1
    nb = np.maximum((bits + 6) // 7, 1).astype(np.int64)
    out = np.zeros(int(nb.sum()), dtype=np.uint8)
    off = np.zeros(n, dtype=np.int64)
    if n > 1:
        off[1:] = np.cumsum(nb[:-1])
    for L in range(1, int(nb.max()) + 1):
        m = (nb == L)
        if not m.any(): continue
        idx = np.where(m)[0]; pos = off[idx]; zsub = zz[idx]
        for b in range(L):
            bv = ((zsub >> np.uint64(7 * b)) & np.uint64(0x7F)).astype(np.uint8)
            if b < L - 1: bv = bv | np.uint8(0x80)
            out[pos + b] = bv
    payload = out.tobytes()
    return LEB128_CHUNK_MAGIC + struct.pack('<I', len(payload)) + payload

# 🦅 nomes do ficheiro de PESOS dentro da pasta do checkpoint do bullet, por PRIORIDADE.
#    O bullet guarda os pesos quantizados em "quantised.bin". A pasta tem também log.txt e
#    o estado do optimiser — que NÃO queremos. (Bug antigo: apanhava o log.txt.)
WEIGHT_NAMES = ["quantised.bin", "quantized.bin", "params.bin", "weights.bin", "raw.bin"]

def _pick_weight_file(folder):
    """Dado um diretório de checkpoint, devolve o ficheiro de PESOS (não log/optimiser/json)."""
    # 1) nomes conhecidos por prioridade
    for name in WEIGHT_NAMES:
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
    # 2) fallback: o MAIOR .bin que não seja do optimiser (os pesos são grandes)
    bins = [f for f in glob.glob(os.path.join(folder, "*.bin"))
            if "optimi" not in os.path.basename(f).lower()]
    if bins:
        return max(bins, key=lambda f: os.path.getsize(f))
    # 3) último recurso: o maior ficheiro que não seja log/json/optimiser
    others = [f for f in glob.glob(os.path.join(folder, "*"))
              if os.path.isfile(f)
              and not f.endswith(".json")
              and "log" not in os.path.basename(f).lower()
              and "optimi" not in os.path.basename(f).lower()]
    if others:
        return max(others, key=lambda f: os.path.getsize(f))
    return None

def find_latest(folder, head):
    """Encontra o ficheiro de PESOS do checkpoint -N mais alto para uma cabeça."""
    pats = glob.glob(os.path.join(folder, f"NAPKa0s_V9_{head}-*"))
    best, bn = None, -1
    for e in pats:
        m = re.search(r'-(\d+)$', os.path.basename(e).split('.')[0])
        if not m: continue
        num = int(m.group(1))
        if num <= bn: continue
        if os.path.isdir(e):
            w = _pick_weight_file(e)
            if w: bn, best = num, w
        elif os.path.isfile(e):
            bn, best = num, e
    return best

# 🦅 O bullet alinha CADA tensor quantizado a 32 bytes (= 16 int16) no quantised.bin.
#    (Confirmado: tamanho do ficheiro só bate com este alinhamento.) Logo, depois de ler
#    cada tensor, é preciso SALTAR o padding até ao próximo múltiplo de 16 int16.
BULLET_ALIGN_I16 = 16
def battalion(l1w, l1b, l2w, l2b, l3w_b, l3b):
    """8 buckets: l1w/l2w partilhados, l3 por bucket."""
    s = bytearray()
    for bk in range(8):
        s.extend(encode_chunk(l1w)); s.extend(encode_chunk(l1b))
        s.extend(encode_chunk(l2w)); s.extend(encode_chunk(l2b))
        s.extend(encode_chunk(l3w_b[bk])); s.extend(encode_chunk(np.array([l3b[bk]], dtype=np.int16)))
    return s


# ════════════════════════════════════════════════════════════════════════════
# 🦅 read_juntas — lê o raw.bin ÚNICO do napk9_train_juntas.rs (as 3 cabeças num só checkpoint).
#   Ordem no raw.bin = ordem dos new_affine: acc, psqt, bullet_l1/l2/l3, small_l1/l2/l3, g_l1/l2/l3.
#   (cada camada: weight depois bias). Devolve dict com acc/psqt + as 3 cabeças + chaos a zeros.
# ════════════════════════════════════════════════════════════════════════════
def _biggest_bin(folder):
    """fallback: o maior .bin da pasta que não seja optimiser/state (preferir raw.bin)."""
    raw = os.path.join(folder, "raw.bin")
    if os.path.isfile(raw): return raw
    bins = [f for f in glob.glob(os.path.join(folder, "*.bin"))
            if "optimiser" not in f.lower() and "state" not in f.lower()]
    return max(bins, key=os.path.getsize) if bins else None

def read_juntas(path, l1):
    folder = os.path.dirname(path)
    raw_path = os.path.join(folder, "raw.bin")
    if not os.path.isfile(raw_path):
        raw_path = path
    raw = np.fromfile(raw_path, dtype=np.float32)
    # 🦅 PROTEÇÃO: confirmar que o raw.bin bate com o L1 pedido (senão reshape rebenta = lixo).
    bl2 = detect_big_l2(l1)
    # 🦅 V10 FIX (s29): o acc pode ter 23169 inputs (clássico, threats 640) OU 31745 (V10 full
    #   threats 9216). Detetar pelo tamanho total em vez de assumir 23169 — era isto que
    #   rejeitava o raw.bin V10 (32929240 floats = geometria V10 PERFEITA p/ L1=1024).
    def total_exp(n_in, L):
        e = n_in*L + L + n_in*8 + 8
        for _h,(a,b) in [("bullet",(16,32)),("small",(32,32)),("big",(BIG_DL1,BIG_DL2))]:
            e += a*L*2 + a + b*a + b + 8*b + 8
        return e
    n_in = None
    for cand in (31745, 23169, 22529):   # full V10, 640 clássico, none (s29: NAPK_THREATS)
        if len(raw) == total_exp(cand, l1): n_in = cand; break
    if n_in is None:
        guess = None
        for L in [128,256,384,512,768,1024]:
            for cand in (31745, 23169, 22529):
                if total_exp(cand, L) == len(raw): guess = (L, cand); break
            if guess: break
        print(f"🔴 raw.bin tem {len(raw)} floats; p/ L1={l1} esperava {total_exp(23169,l1)} (clássico) ou {total_exp(31745,l1)} (V10).")
        if guess: print(f"   → este checkpoint é L1={guess[0]} ({'V10' if guess[1]==31745 else 'clássico'}). Usa --l1 {guess[0]}.")
        else:     print(f"   → L1 não reconhecido (geometria invulgar).")
        sys.exit(1)
    _modo = {31745: 'FULL V10 (9216)', 23169: 'threats 640', 22529: 'SEM threats'}[n_in]
    print(f"🦅 geometria detetada: acc {n_in} inputs ({_modo}) × L1={l1}")
    ptr = [0]
    def q(scale, shape):
        size = int(np.prod(shape))
        seg = raw[ptr[0]:ptr[0]+size]; ptr[0] += size
        qv = np.sign(seg) * np.floor(np.abs(seg) * scale + 0.5)
        return np.clip(qv, -32768, 32767).astype(np.int16).reshape(shape)

    big_l2 = detect_big_l2(l1)
    S_ACC = 255; S_PSQT = 255; S_L12 = 64; S_L3 = 255 * 64
    d = {}
    # ── partilhados (acc, psqt) — primeiro no raw, como no .rs ──
    d["accw"]  = q(S_ACC, (n_in, l1)); d["accb"]  = q(S_ACC, (l1,))
    d["psqtw"] = q(S_PSQT, (n_in, 8)); d["psqtb"] = q(S_PSQT, (8,))
    # ── as 3 cabeças, na ORDEM dos new_affine: bullet, small, g(big) ──
    for head, (h1, h2) in [("bullet", (16, 32)), ("small", (32, 32)), ("big", (BIG_DL1, BIG_DL2))]:
        d[f"{head}_l1w"] = q(S_L12, (h1, l1*2)); d[f"{head}_l1b"] = q(S_L12, (h1,))
        d[f"{head}_l2w"] = q(S_L12, (h2, h1));   d[f"{head}_l2b"] = q(S_L12, (h2,))
        d[f"{head}_l3w"] = q(S_L3,  (8, h2));    d[f"{head}_l3b"] = q(S_L3,  (8,))
    # ── chaos a ZEROS (o juntas não treina chaos; o motor lê na mesma) ──
    d["chaos_l1w"] = np.zeros((32, l1*2), dtype=np.int16); d["chaos_l1b"] = np.zeros((32,), dtype=np.int16)
    d["chaos_l2w"] = np.zeros((16, 32), dtype=np.int16);   d["chaos_l2b"] = np.zeros((16,), dtype=np.int16)
    d["chaos_l3w"] = np.zeros((1, 16), dtype=np.int16);    d["chaos_l3b"] = np.zeros((1,), dtype=np.int16)
    leftover = len(raw) - ptr[0]
    if leftover != 0:
        print(f"  ⚠️ sobram {leftover} float32 no raw.bin (L1={l1} errado? geometria?)")
        # geometria esperada p/ ajudar a diagnosticar
        exp = 23169*l1 + l1 + 23169*8 + 8
        for h,(a,b) in [("bullet",(16,32)),("small",(32,32)),("big",(BIG_DL1,BIG_DL2))]:
            exp += a*l1*2 + a + b*a + b + 8*b + 8
        print(f"     (raw tem {len(raw)} floats; esperado p/ L1={l1} = {exp})")
    else:
        print(f"  ✅ raw.bin lido inteiro (L1={l1}): acc+psqt+bullet+small+big, geometria certa")
    return d

def main():
    ap = argparse.ArgumentParser(description="Converte o checkpoint ÚNICO do napk9_train_juntas → .napk9")
    ap.add_argument("--checkpoint", help="pasta do checkpoint juntas (com raw.bin). Ex: checkpoints_napk/NAPKa0s_juntas_128-100")
    ap.add_argument("--checkpoints", default="/mnt/c/data/env/bullet/checkpoints_napk",
                    help="pasta base p/ procurar o NAPKa0s_juntas_*-N mais recente")
    ap.add_argument("--out", default="/mnt/d/Nap2Siriux/nets/bullet/npk9_master_juntas.napk9")
    ap.add_argument("--l1", type=int, default=128)
    ap.add_argument("--tag", default="", help="tag do dataset (ex: 20M, 160M, 40M) p/ achar NAPKa0s_juntas_<L1>_<tag>-N")
    args = ap.parse_args()

    # 🦅 prefixo do checkpoint: com --tag distingue datasets (NAPKa0s_juntas_128_20M vs _160M).
    ckpt_prefix = f"NAPKa0s_juntas_{args.l1}_{args.tag}" if args.tag else f"NAPKa0s_juntas_{args.l1}"

    # localizar o checkpoint (passado, ou o juntas -N mais recente)
    ckpt = args.checkpoint
    if ckpt:
        # 🦅 --checkpoint EXPLÍCITO: se não existe, ERRO (não cair em auto-find silencioso → bug do
        #   1024 que apanhava o 128). Aceita pasta -16/-N (o treino salva -<end_sb>, não sempre -100).
        if os.path.isdir(ckpt):
            raw = os.path.join(ckpt, "raw.bin")
            path = raw if os.path.isfile(raw) else _biggest_bin(ckpt)
        elif os.path.isfile(ckpt):
            path = ckpt
        else:
            print(f"🔴 --checkpoint '{ckpt}' não existe (nem pasta nem ficheiro).")
            # ajudar: listar os checkpoints que existem
            base = os.path.dirname(ckpt) or args.checkpoints
            existing = sorted(glob.glob(os.path.join(base, "NAPKa0s_juntas_*")))
            if existing:
                print("   checkpoints disponíveis:")
                for e in existing: print(f"     {e}")
            print(f"   (nota: o treino salva -<superbatches>, ex: NAPKa0s_juntas_1024-16, não -100)")
            sys.exit(1)
    else:
        # auto-find: SÓ o L1 (+tag se dada) pedido.
        cands = glob.glob(os.path.join(args.checkpoints, f"{ckpt_prefix}-*"))
        if not cands:
            print(f"❌ nenhum checkpoint {ckpt_prefix}-* em {args.checkpoints}")
            outros = glob.glob(os.path.join(args.checkpoints, "NAPKa0s_juntas_*"))
            if outros:
                print("   (existem outros:)"); [print(f"     {o}") for o in sorted(outros)]
            sys.exit(1)
        def num(p):
            m = re.search(r"-(\d+)$", p); return int(m.group(1)) if m else -1
        best = max(cands, key=num)
        raw = os.path.join(best, "raw.bin")
        path = raw if os.path.isfile(raw) else _biggest_bin(best)
        print(f"🦅 checkpoint auto ({ckpt_prefix}): {best}")

    if not path or not os.path.exists(path):
        print(f"❌ raw.bin não encontrado no checkpoint. Tens o raw.bin (não só o quantised.bin)?")
        sys.exit(1)
    print(f"🦅 converter juntas (L1={args.l1}): {path}")

    h = read_juntas(path, args.l1)

    accw, psqtw = h["accw"], h["psqtw"]
    # 🦅 V10 auto-detetado pelo nº de linhas do acc: 31745 → full threats (9216); 23169 → 640.
    n_in = accw.shape[0]
    n_in = accw.shape[0]   # (reatribuição inofensiva — mesmo valor)
    is_v10 = n_in >= 31744
    has_640 = (n_in == 23169)
    has_none = (n_in == 22529)
    thr_end = 22528 + (9216 if is_v10 else 640)   # p/ none: ver abaixo (chunk de zeros)
    if is_v10: print("  🔥 V10 FULL THREATS detetado (acc 31745): split 22528 + 9216, flags |= 8")
    ft_weight     = accw[:22528, :].flatten()
    if has_none:
        # 🦅 modo NONE: chunk de threats = ZEROS 640×L1 → o motor deteta nonZero=False e
        #   desliga (hasThreats=false) — compatível com TODOS os binários existentes.
        threat_weight = np.zeros(640 * l1, dtype=np.float32)
        print("  ⚪ SEM threats: chunk de zeros 640×L1 (motor desliga threats sozinho)")
    else:
        threat_weight = accw[22528:thr_end, :].flatten()
        if has_640: print("  ⚔️ threats 640 clássico: split 22528 + 640 (flags SEM bit 8)")
    ft_psqt       = psqtw[:22528, :].flatten()

    out = bytearray()
    flags = 1 | 2 | 4
    if is_v10: flags |= 8   # 🦅 V10: o motor lê o bit 8 → tensor threats 9216
    out += struct.pack('<8sIIIIIII', MAGIC, VERSION, args.l1, MATERIAL_BUCKETS,
                       TOTAL_FEATURES, QA, QB, flags)
    out += encode_chunk(h["accb"])
    out += encode_chunk(ft_weight)
    out += encode_chunk(ft_psqt)
    # 3 batalhões — TODOS do MESMO checkpoint coerente
    out += battalion(h["bullet_l1w"], h["bullet_l1b"], h["bullet_l2w"], h["bullet_l2b"], h["bullet_l3w"], h["bullet_l3b"])
    out += battalion(h["small_l1w"],  h["small_l1b"],  h["small_l2w"],  h["small_l2b"],  h["small_l3w"],  h["small_l3b"])
    out += battalion(h["big_l1w"],    h["big_l1b"],    h["big_l2w"],    h["big_l2b"],    h["big_l3w"],    h["big_l3b"])
    out += encode_chunk(h["chaos_l1w"]) + encode_chunk(h["chaos_l1b"])
    out += encode_chunk(h["chaos_l2w"]) + encode_chunk(h["chaos_l2b"])
    out += encode_chunk(h["chaos_l3w"]) + encode_chunk(h["chaos_l3b"])
    out += encode_chunk(threat_weight)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(out)
    print(f"[✓] .napk9 COERENTE (3 cabeças do MESMO checkpoint): {args.out}  ({len(out)} bytes)")
    print(f"    acc/psqt partilhados; bullet+small+big todos do treino juntas (coerentes). chaos=0.")

if __name__ == "__main__":
    main()
