#!/usr/bin/env python3
# 🦅 auto_converter_li11.py — converte o checkpoint do napk9_train_li11.rs (raw.bin) num
#   ficheiro .li11 (magic "NAPKLI11"), o formato novo lido por loadLI11() em nnue_net.cpp.
#
#   Layout do raw.bin (ordem = save_ids_li11() em napk9_train_li11.rs, n_in=31745 p/ full
#   threats — o "+1" é o mesmo padrão do NapkInputV10 do V10/juntas, não um erro):
#     accw(n_in,L1) accb(L1) psqtw(n_in,8) psqtb(8) fc0w(33,L1) fc0b(33)
#     fc1w(32,64) fc1b(32) fc2w(8,32) fc2b(8)
#
#   accw/psqtw são UMA matriz no treino mas o MOTOR quer peças e threats em chunks
#   SEPARADOS (mesma convenção do auto_converter_juntas.py para V10): split em
#   accw[:22528] (peças) e accw[22528:22528+9216] (threats).
#
#   Ficheiro de saída:
#     magic"NAPKLI11"(8) L1(u32) QA(u32) QB_FC(u32) QB_FC2(u32)
#     [accBias i16] [accWeight(peças) i16, 22528*L1] [psqt(peças) i32, 22528*8]
#     [fc0w i16, 33*L1] [fc0b i32, 33]
#     [fc1w i16, 32*64] [fc1b i32, 32]
#     [fc2w i16, 8*32]  [fc2b i32, 8]
#     [threatWeight i16, 9216*L1]   (opcional — omitido se NAPK_THREATS=none no treino)
#   Cada [] é um chunk LEB128 (mesmo formato/magic "COMPRESSED_LEB128" do .napk9 antigo).
#   psqt dos threats (linhas 22528:31744 de psqtw) é DESCARTADO — o motor só lê psqt das
#   peças (22528 linhas), o mesmo que o V10/juntas já faz (ver auto_converter_juntas.py).
#
#   Uso:
#     python3 auto_converter_li11.py --checkpoint checkpoints_napk/NAPKa0s_li11_512_<tag>-<sb> \
#         --l1 512 --train-scale 400 --out nets/li11_512.li11

import os, sys, struct, argparse
import numpy as np

MAGIC = b"NAPKLI11"
LEB128_CHUNK_MAGIC = b"COMPRESSED_LEB128"
TOTAL_FEATURES = 22528
THREAT_FEATURES_FULL = 9216
MATERIAL_BUCKETS = 8
QA = 255
QB_FC = 64
QB_FC2 = QB_FC * QB_FC
FC0_REAL = 32
FC0_TOTAL = FC0_REAL + 1
FC1_OUT = 32

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

def qround(arr, scale, dtype=np.int16):
    return np.clip(np.round(arr * scale), -32768, 32767).astype(dtype)

def main():
    ap = argparse.ArgumentParser(description="Converte o checkpoint LI11 (raw.bin) -> .li11 (NAPKLI11)")
    ap.add_argument("--checkpoint", required=True, help="pasta do checkpoint (com raw.bin) ou caminho direto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--l1", type=int, default=512)
    ap.add_argument("--train-scale", type=float, default=400.0,
                    help="eval_scale do treino. Motor usa OUTPUT_SCALE_CP=400 agora — se baterem "
                         "(o normal), RESCALE=1.0, sem ajuste nenhum.")
    args = ap.parse_args()

    raw_path = os.path.join(args.checkpoint, "raw.bin") if os.path.isdir(args.checkpoint) else args.checkpoint
    if not os.path.isfile(raw_path):
        print(f"🔴 não encontrado: {raw_path}"); sys.exit(1)

    raw = np.fromfile(raw_path, dtype=np.float32)
    l1 = args.l1

    # 🦅 detetar n_in pela contagem total (mesmo espírito do auto_converter_juntas.py):
    #   tenta full-threats (31745) primeiro, cai p/ sem-threats (22528) se não bater certo.
    def total_exp(n_in):
        return n_in*l1 + l1 + n_in*8 + 8 + FC0_TOTAL*l1 + FC0_TOTAL + FC1_OUT*64 + FC1_OUT \
             + MATERIAL_BUCKETS*FC1_OUT + MATERIAL_BUCKETS

    n_in = None
    for cand in (31745, 22528):
        if total_exp(cand) == len(raw):
            n_in = cand; break
    if n_in is None:
        print(f"🔴 raw.bin tem {len(raw)} floats; não bate com L1={l1} p/ n_in=31745 (full threats, "
              f"esperava {total_exp(31745)}) nem n_in=22528 (sem threats, esperava {total_exp(22528)}).")
        sys.exit(1)
    has_threats = (n_in == 31745)
    print(f"🦅 geometria: n_in={n_in} ({'FULL THREATS 9216' if has_threats else 'sem threats'}) × L1={l1}")

    RESCALE = args.train_scale / 400.0
    if RESCALE != 1.0:
        print(f"  🦅 reconciliação de escala: rede treinada a {args.train_scale}, motor usa 400 → psqt/fc2 × {RESCALE:.6f}")

    ptr = [0]
    def take(shape):
        size = int(np.prod(shape))
        seg = raw[ptr[0]:ptr[0]+size]; ptr[0] += size
        return seg.reshape(shape)

    accw  = take((n_in, l1))
    accb  = take((l1,))
    psqtw = take((n_in, MATERIAL_BUCKETS))
    psqtb = take((MATERIAL_BUCKETS,))   # treinado mas NÃO escrito -- motor não o lê (cancela em psqtUs-psqtThem)
    fc0w  = take((FC0_TOTAL, l1))
    fc0b  = take((FC0_TOTAL,))
    fc1w  = take((FC1_OUT, 64))
    fc1b  = take((FC1_OUT,))
    fc2w  = take((MATERIAL_BUCKETS, FC1_OUT))
    fc2b  = take((MATERIAL_BUCKETS,))
    _ = psqtb

    leftover = len(raw) - ptr[0]
    if leftover != 0:
        print(f"  ⚠️ sobram {leftover} float32 no raw.bin — geometria errada?")
    else:
        print(f"  ✅ raw.bin lido inteiro (L1={l1}): acc+psqt+fc0+fc1+fc2, geometria certa")

    # split peças / threats (mesma convenção do auto_converter_juntas.py p/ V10)
    pieces_w = accw[:TOTAL_FEATURES, :]
    pieces_psqtw = psqtw[:TOTAL_FEATURES, :]
    threat_w = accw[TOTAL_FEATURES:TOTAL_FEATURES + THREAT_FEATURES_FULL, :] if has_threats else None

    accb_q  = qround(accb, QA)
    accw_q  = qround(pieces_w, QA)
    psqtw_q = qround(pieces_psqtw * RESCALE, QA, np.int32)
    fc0w_q  = qround(fc0w, QB_FC)
    fc0b_q  = qround(fc0b, QB_FC, np.int32)
    fc1w_q  = qround(fc1w, QB_FC)
    fc1b_q  = qround(fc1b, QB_FC, np.int32)
    fc2w_q  = qround(fc2w * RESCALE, QB_FC2)
    fc2b_q  = qround(fc2b * RESCALE, QB_FC2, np.int32)

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack('<IIII', l1, QA, QB_FC, QB_FC2))
        f.write(encode_chunk(accb_q))
        f.write(encode_chunk(accw_q.flatten()))
        f.write(encode_chunk(psqtw_q.flatten()))
        f.write(encode_chunk(fc0w_q.flatten()))
        f.write(encode_chunk(fc0b_q))
        f.write(encode_chunk(fc1w_q.flatten()))
        f.write(encode_chunk(fc1b_q))
        f.write(encode_chunk(fc2w_q.flatten()))
        f.write(encode_chunk(fc2b_q))
        if has_threats:
            threat_w_q = qround(threat_w, QA)
            f.write(encode_chunk(threat_w_q.flatten()))
            print("  🔥 threats incluídos (9216 × L1)")
        else:
            print("  ⚪ sem threats (treino com NAPK_THREATS=none) — chunk omitido")

    print(f"[✓] .li11 escrito: {args.out} ({os.path.getsize(args.out)} bytes)")

if __name__ == "__main__":
    main()
