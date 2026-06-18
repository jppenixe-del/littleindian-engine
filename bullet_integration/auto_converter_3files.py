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

def detect_big_l2(l1):
    if l1 <= 128: return 16
    if l1 <= 256: return 32
    if l1 <= 512: return 32
    if l1 <= 768: return 48
    if l1 <= 1024: return 64
    return 96

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

def read_single(path, l1, head):
    """Lê os pesos de 1 cabeça SEM chaos → dict acc/psqt/cabeça, quantizados em i16.

    🦅 LIÇÃO (sessão 17): o quantised.bin do bullet tem PADDING de alinhamento irregular entre
    tensores (difícil de prever). O raw.bin (float32) é LIMPO e contíguo — bate exatamente com
    a geometria pura nas 3 cabeças. Por isso lemos do raw.bin e quantizamos NÓS, com as MESMAS
    escalas do save_ids_for (accw/accb/psqt ×255, l1/l2 ×64, l3 ×255*64) e o MESMO arredondamento
    do Rust (.round() = round-half-away = floor(x+0.5) para +, simétrico). A chaos é injetada a
    ZEROS no .napk9 (o treino no-chaos não a tem; fear marginal; o motor lê na mesma).

    `path` é o ficheiro escolhido pelo find_latest. Se for o quantised.bin, trocamos para o
    raw.bin que está ao lado (mesma pasta). Se já for raw.bin, usa-se direto.
    """
    # garantir que lemos o raw.bin (float32), não o quantised.bin (com padding)
    folder = os.path.dirname(path)
    raw_path = os.path.join(folder, "raw.bin")
    if not os.path.isfile(raw_path):
        # fallback: se não houver raw.bin, tenta o próprio path (pode já ser raw)
        raw_path = path
    raw = np.fromfile(raw_path, dtype=np.float32)
    ptr = [0]

    def q(scale, shape):
        """lê um tensor float32, quantiza ×scale com round-half-away, devolve i16 (clip)."""
        size = int(np.prod(shape))
        seg = raw[ptr[0]:ptr[0]+size]; ptr[0] += size
        qv = np.sign(seg) * np.floor(np.abs(seg) * scale + 0.5)
        qv = np.clip(qv, -32768, 32767).astype(np.int16)
        return qv.reshape(shape)

    big_l2 = detect_big_l2(l1)
    h1, h2 = {"bullet": (16, 32), "small": (32, 32), "big": (l1, big_l2)}[head]
    # escalas IGUAIS ao save_ids_for (napk9_train_nochaos.rs)
    S_ACC = 255; S_PSQT = 255; S_L12 = 64; S_L3 = 255 * 64
    d = {}
    d["accw"]  = q(S_ACC, (23169, l1)); d["accb"]  = q(S_ACC, (l1,))
    d["psqtw"] = q(S_PSQT, (23169, 8)); d["psqtb"] = q(S_PSQT, (8,))
    d["l1w"] = q(S_L12, (h1, l1*2)); d["l1b"] = q(S_L12, (h1,))
    d["l2w"] = q(S_L12, (h2, h1));   d["l2b"] = q(S_L12, (h2,))
    d["l3w"] = q(S_L3,  (8, h2));    d["l3b"] = q(S_L3,  (8,))
    # 🦅 chaos a ZEROS (treino no-chaos). Geometria do motor: chaos_l1 (32,L1*2), l2 (16,32), l3 (1,16).
    d["chaos_l1w"] = np.zeros((32, l1*2), dtype=np.int16); d["chaos_l1b"] = np.zeros((32,), dtype=np.int16)
    d["chaos_l2w"] = np.zeros((16, 32), dtype=np.int16);   d["chaos_l2b"] = np.zeros((16,), dtype=np.int16)
    d["chaos_l3w"] = np.zeros((1, 16), dtype=np.int16);    d["chaos_l3b"] = np.zeros((1,), dtype=np.int16)
    leftover = len(raw) - ptr[0]
    if leftover != 0:
        print(f"  ⚠️ {head}: sobram {leftover} float32 no raw.bin (L1 errado? geometria?)")
    else:
        print(f"  ✅ {head}: lido do raw.bin (h1={h1}, h2={h2}), quantizado i16, geometria certa")
    return d

def battalion(l1w, l1b, l2w, l2b, l3w_b, l3b):
    """8 buckets: l1w/l2w partilhados, l3 por bucket."""
    s = bytearray()
    for bk in range(8):
        s.extend(encode_chunk(l1w)); s.extend(encode_chunk(l1b))
        s.extend(encode_chunk(l2w)); s.extend(encode_chunk(l2b))
        s.extend(encode_chunk(l3w_b[bk])); s.extend(encode_chunk(np.array([l3b[bk]], dtype=np.int16)))
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big"); ap.add_argument("--small"); ap.add_argument("--bullet")
    ap.add_argument("--checkpoints", default="/mnt/c/data/env/bullet/checkpoints_napk")
    ap.add_argument("--out", default="/mnt/d/Nap2Siriux/nets/v6/_train_128/npk9_master_128.napk9")
    ap.add_argument("--l1", type=int, default=128)
    args = ap.parse_args()

    big_p    = args.big    or find_latest(args.checkpoints, "big")
    small_p  = args.small  or find_latest(args.checkpoints, "small")
    bullet_p = args.bullet or find_latest(args.checkpoints, "bullet")
    for name, p in [("big", big_p), ("small", small_p), ("bullet", bullet_p)]:
        if not p or not os.path.exists(p):
            print(f"❌ checkpoint da cabeça '{name}' não encontrado. Treinaste NAPK_HEAD={name}?")
            sys.exit(1)
    print(f"🦅 a juntar 3 cabeças (L1={args.l1}):")
    print(f"   big    = {big_p}")
    print(f"   small  = {small_p}")
    print(f"   bullet = {bullet_p}")

    # 🦅 PROTEÇÃO (lição da sessão 17): verificar que os checkpoints são MESMO do L1 pedido.
    #    Checkpoints de 128 NÃO fazem uma .napk9 256 — a geometria não bate e a rede sai
    #    CORROMPIDA (satura a 3000cp em tudo). Recusar em vez de gerar lixo.
    def detect_l2(l1):
        return 16 if l1<=128 else (32 if l1<=512 else (48 if l1<=768 else (64 if l1<=1024 else 96)))
    def ckpt_l1(path, head):
        raw_path = os.path.join(os.path.dirname(path), "raw.bin")
        if not os.path.isfile(raw_path): raw_path = path
        n = os.path.getsize(raw_path) // 4
        for L in [128, 256, 512, 768, 1024, 2048]:
            h1, h2 = (16,32) if head=="bullet" else ((32,32) if head=="small" else (L, detect_l2(L)))
            geom = 23169*L + L + 23169*8 + 8 + h1*L*2 + h1 + h2*h1 + h2 + 8*h2 + 8
            if geom == n: return L
        return None
    for p, name in [(big_p,"big"), (small_p,"small"), (bullet_p,"bullet")]:
        real_l1 = ckpt_l1(p, name)
        if real_l1 is not None and real_l1 != args.l1:
            print(f"\n🔴 ERRO FATAL: o checkpoint '{name}' é de L1={real_l1}, mas pediste --l1 {args.l1}.")
            print(f"   Geometria L1={real_l1} ≠ L1={args.l1} → a rede sairia CORROMPIDA. PARO aqui.")
            print(f"   SOLUÇÃO: usa --l1 {real_l1} (estes checkpoints), ou treina de novo com L1={args.l1}.")
            sys.exit(1)
        elif real_l1 is None:
            print(f"   ⚠️ {name}: não consegui confirmar o L1 do checkpoint (geometria invulgar). Continuo,")
            print(f"      mas se o eval sair saturado (3000cp), o L1 pode não bater com --l1 {args.l1}.")

    big    = read_single(big_p,    args.l1, "big")
    small  = read_single(small_p,  args.l1, "small")
    bullet = read_single(bullet_p, args.l1, "bullet")

    # acc/psqt/chaos vêm da BIG (a que decide; o motor partilha 1 acc/psqt)
    accw, psqtw = big["accw"], big["psqtw"]
    ft_weight    = accw[:22528, :].flatten()
    threat_weight = accw[22528:23168, :].flatten()
    ft_psqt      = psqtw[:22528, :].flatten()

    out = bytearray()
    flags = 1 | 2 | 4
    out += struct.pack('<8sIIIIIII', MAGIC, VERSION, args.l1, MATERIAL_BUCKETS,
                       TOTAL_FEATURES, QA, QB, flags)
    out += encode_chunk(big["accb"])      # ft bias
    out += encode_chunk(ft_weight)        # ft weight
    out += encode_chunk(ft_psqt)          # ft psqt
    # 3 batalhões: bullet (do checkpoint bullet), small (do small), big (do big)
    out += battalion(bullet["l1w"], bullet["l1b"], bullet["l2w"], bullet["l2b"], bullet["l3w"], bullet["l3b"])
    out += battalion(small["l1w"],  small["l1b"],  small["l2w"],  small["l2b"],  small["l3w"],  small["l3b"])
    out += battalion(big["l1w"],    big["l1b"],    big["l2w"],    big["l2b"],    big["l3w"],    big["l3b"])
    # chaos (da big)
    out += encode_chunk(big["chaos_l1w"]) + encode_chunk(big["chaos_l1b"])
    out += encode_chunk(big["chaos_l2w"]) + encode_chunk(big["chaos_l2b"])
    out += encode_chunk(big["chaos_l3w"]) + encode_chunk(big["chaos_l3b"])
    out += encode_chunk(threat_weight)    # threats

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(out)
    print(f"[✓] .napk9 de 3 cabeças (caminho B): {args.out}  ({len(out)} bytes)")
    print(f"    acc/psqt/chaos da BIG; batalhões bullet+small+big dos respetivos checkpoints.")

if __name__ == "__main__":
    main()
