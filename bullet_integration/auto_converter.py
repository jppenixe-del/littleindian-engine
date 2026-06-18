#!/usr/bin/env python3
# 🦅 auto_converter V9.7 — fita Rust int16 (3 CABEÇAS) → NAPK9LEB.
#   Agora exporta os PESOS REAIS das 3 cabeças (bullet/small/big), não zeros.
#   Alinhado à geometria do motor (L1=128): bullet 16→32, small 32→32, big 128→16.
import os, sys, time, glob, re, struct
import numpy as np

MAGIC = b"NAPK9LEB"
VERSION = 1
LEB128_CHUNK_MAGIC = b"COMPRESSED_LEB128"
TOTAL_FEATURES = 22528
MATERIAL_BUCKETS = 8
QA = 255
QB = 64

# 🦅 Geometria (tem de bater com napk9.rs e com o motor). L1 lido do save (default 128).
def detect_big_l2(l1):
    if l1 <= 128: return 16
    if l1 <= 256: return 32
    if l1 <= 512: return 32
    if l1 <= 768: return 48
    if l1 <= 1024: return 64
    return 96

def encode_array_leb128_chunk(values: np.ndarray) -> bytes:
    val = values.astype(np.int64).flatten()
    n = len(val)
    if n == 0:
        return LEB128_CHUNK_MAGIC + struct.pack('<I', 0)
    zz = np.where(val >= 0, val << 1, ((-val - 1) << 1) | 1).astype(np.uint64)
    safe_zz = np.maximum(zz, 1)
    bits_used = np.floor(np.log2(safe_zz)).astype(np.int64) + 1
    n_bytes = np.maximum((bits_used + 6) // 7, 1).astype(np.int64)
    total_bytes = int(n_bytes.sum())
    out = np.zeros(total_bytes, dtype=np.uint8)
    offsets = np.zeros(n, dtype=np.int64)
    if n > 1:
        offsets[1:] = np.cumsum(n_bytes[:-1])
    max_L = int(n_bytes.max())
    for L in range(1, max_L + 1):
        mask = (n_bytes == L)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        positions = offsets[idx]
        zz_subset = zz[idx]
        for b in range(L):
            byte_vals = ((zz_subset >> np.uint64(7 * b)) & np.uint64(0x7F)).astype(np.uint8)
            if b < L - 1:
                byte_vals = byte_vals | np.uint8(0x80)
            out[positions + b] = byte_vals
    payload = out.tobytes()
    return LEB128_CHUNK_MAGIC + struct.pack('<I', len(payload)) + payload

def encontrar_ultimo_checkpoint(folder):
    padrao = os.path.join(folder, "NAPKa0s_V9_3heads-*")
    entradas = glob.glob(padrao)
    if not entradas:
        # fallback ao nome antigo
        entradas = glob.glob(os.path.join(folder, "NAPKa0s_V9_*-*"))
    if not entradas:
        return None, -1
    maior_num = -1; melhor = None
    for entrada in entradas:
        m = re.search(r'-(\d+)', os.path.basename(entrada))
        if not m: continue
        num = int(m.group(1))
        if os.path.isdir(entrada):
            for sub in glob.glob(os.path.join(entrada, "*")):
                if os.path.isfile(sub) and not sub.endswith(".json"):
                    if num > maior_num: maior_num, melhor = num, sub
        elif os.path.isfile(entrada):
            if num > maior_num: maior_num, melhor = num, entrada
    return melhor, maior_num

def battalion_chunks(l1w, l1b, l2w, l2b, l3w_buckets, l3b, l1_out, l2_out):
    """Empacota um batalhão (1 cabeça) × 8 buckets. l1w/l2w são partilhados entre buckets
       (a g_l3 é que é por bucket). l3w_buckets shape [8, l2_out], l3b shape [8]."""
    stream = bytearray()
    for bucket in range(8):
        bb = bytearray()
        bb.extend(encode_array_leb128_chunk(l1w))                       # [l1_out, L1*2]
        bb.extend(encode_array_leb128_chunk(l1b))                       # [l1_out]
        bb.extend(encode_array_leb128_chunk(l2w))                       # [l2_out, l1_out]
        bb.extend(encode_array_leb128_chunk(l2b))                       # [l2_out]
        bb.extend(encode_array_leb128_chunk(l3w_buckets[bucket]))       # [l2_out]
        bb.extend(encode_array_leb128_chunk(np.array([l3b[bucket]], dtype=np.int16)))
        stream.extend(bb)
    return stream

def executar_conversao(input_bin, output_target, l1=128):
    print(f"\n[⚡] auto_converter V9.7 (3 CABEÇAS) — {input_bin} (L1={l1})")
    big_l2 = detect_big_l2(l1)
    raw = np.fromfile(input_bin, dtype=np.int16)
    ptr = 0
    def take(shape):
        nonlocal ptr
        size = int(np.prod(shape))
        arr = raw[ptr:ptr+size].reshape(shape)
        ptr += size
        return arr

    # ── ordem EXATA do save_format do napk9_train.rs ──
    accw  = take((23169, l1))
    accb  = take((l1,))
    psqtw = take((23169, 8))
    psqtb = take((8,))

    # BULLET (16 → 32 → 1×8)
    bullet_l1w = take((16, l1*2)); bullet_l1b = take((16,))
    bullet_l2w = take((32, 16));   bullet_l2b = take((32,))
    bullet_l3w = take((8, 32));    bullet_l3b = take((8,))   # g_l3 = [1*8, l2_out] → [8,32]

    # SMALL (32 → 32 → 1×8)
    small_l1w = take((32, l1*2)); small_l1b = take((32,))
    small_l2w = take((32, 32));   small_l2b = take((32,))
    small_l3w = take((8, 32));    small_l3b = take((8,))

    # BIG (L1 → bigL2 → 1×8)
    g_l1w = take((l1, l1*2));     g_l1b = take((l1,))
    g_l2w = take((big_l2, l1));   g_l2b = take((big_l2,))
    g_l3w = take((8, big_l2));    g_l3b = take((8,))

    # CHAOS (32 → 16 → 1)
    chaos_l1w = take((32, l1*2)); chaos_l1b = take((32,))
    chaos_l2w = take((16, 32));   chaos_l2b = take((16,))
    chaos_l3w = take((1, 16));    chaos_l3b = take((1,))

    print(f"    lidos {ptr} int16 (ficheiro tem {len(raw)})")
    if ptr != len(raw):
        print(f"    ⚠️ sobram {len(raw)-ptr} int16 — a geometria pode não bater! (L1 errado?)")

    # ── inputs do feature transformer (22528 peças + 640 threats) ──
    ft_weight_real    = accw[:22528, :].flatten()
    threat_weight_real = accw[22528:23168, :].flatten()
    ft_psqt_real      = psqtw[:22528, :].flatten()

    chunk_ft_bias   = encode_array_leb128_chunk(accb)
    chunk_ft_weight = encode_array_leb128_chunk(ft_weight_real)
    chunk_ft_psqt   = encode_array_leb128_chunk(ft_psqt_real)
    chunk_threats   = encode_array_leb128_chunk(threat_weight_real)

    # ── os 3 batalhões com PESOS REAIS (motor lê bullet, small, big nesta ordem) ──
    battalions = bytearray()
    battalions.extend(battalion_chunks(bullet_l1w, bullet_l1b, bullet_l2w, bullet_l2b,
                                       bullet_l3w, bullet_l3b, 16, 32))
    battalions.extend(battalion_chunks(small_l1w, small_l1b, small_l2w, small_l2b,
                                       small_l3w, small_l3b, 32, 32))
    battalions.extend(battalion_chunks(g_l1w, g_l1b, g_l2w, g_l2b,
                                       g_l3w, g_l3b, l1, big_l2))

    # ── chaos head ──
    chaos = bytearray()
    chaos.extend(encode_array_leb128_chunk(chaos_l1w) + encode_array_leb128_chunk(chaos_l1b))
    chaos.extend(encode_array_leb128_chunk(chaos_l2w) + encode_array_leb128_chunk(chaos_l2b))
    chaos.extend(encode_array_leb128_chunk(chaos_l3w) + encode_array_leb128_chunk(chaos_l3b))

    flags = 1 | 2 | 4
    header = struct.pack('<8sIIIIIII', MAGIC, VERSION, l1, MATERIAL_BUCKETS,
                         TOTAL_FEATURES, QA, QB, flags)

    os.makedirs(os.path.dirname(output_target), exist_ok=True)
    with open(output_target, 'wb') as out:
        out.write(header)
        out.write(chunk_ft_bias)
        out.write(chunk_ft_weight)
        out.write(chunk_ft_psqt)
        out.write(battalions)     # 3 cabeças com pesos REAIS
        out.write(chaos)
        out.write(chunk_threats)
    print(f"[✓] CONCLUÍDO (3 cabeças reais): {output_target}")

def main():
    checkpoints = "/mnt/c/data/env/bullet/checkpoints_napk"
    output = "/mnt/d/Nap2Siriux/nets/v6/_train_128/npk9_master_128.napk9"
    l1 = 128
    for a in sys.argv[1:]:
        if a.startswith("--l1="): l1 = int(a.split("=")[1])
    watch = "--watch" in sys.argv
    if watch:
        print(f"[📡] SENTINELA (3 cabeças, L1={l1}): {checkpoints}")
        ultimo = -1
        while True:
            f, num = encontrar_ultimo_checkpoint(checkpoints)
            if f and num > ultimo:
                time.sleep(2.0)
                try:
                    executar_conversao(f, output, l1); ultimo = num
                    print("[💤] À espera do próximo superbatch...")
                except Exception as e:
                    print(f"[-] erro: {e}")
            time.sleep(5.0)
    else:
        f, num = encontrar_ultimo_checkpoint(checkpoints)
        if not f:
            print(f"[-] sem checkpoints em {checkpoints}"); sys.exit(1)
        executar_conversao(f, output, l1)

if __name__ == "__main__":
    main()
