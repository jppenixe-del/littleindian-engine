#!/usr/bin/env python3
# 🦅 calibra_li11_fc.py — verifica o forward pass FC0->skip->FC1->FC2->psqt da LI11
#   comparando: (a) pesos NÃO quantizados lidos diretamente do raw.bin (float puro,
#   mesma semântica do grafo do treino: sqrrelu=relu(x)^2 SEM clamp, crelu=clamp(x,0,1))
#   vs (b) o output já quantizado/dequantizado do motor C++ (dump via LI11_DEBUG_DUMP).
#
#   Uso:
#     LI11_DEBUG=1 LI11_DEBUG_DUMP=/tmp/dump.txt ./littleindian <<< 'position startpos\neval\nquit'
#     python3 training/calibra_li11_fc.py --checkpoint <pasta com raw.bin> --l1 512 --dump /tmp/dump.txt

import argparse, os, sys
import numpy as np

FC0_REAL = 32
FC0_TOTAL = 33
FC1_OUT = 32
MATERIAL_BUCKETS = 8
TOTAL_FEATURES = 22528
THREAT_FEATURES_FULL = 9216

def load_raw(checkpoint, l1):
    raw_path = os.path.join(checkpoint, "raw.bin") if os.path.isdir(checkpoint) else checkpoint
    raw = np.fromfile(raw_path, dtype=np.float32)

    def total_exp(n_in):
        return n_in*l1 + l1 + n_in*8 + 8 + FC0_TOTAL*l1 + FC0_TOTAL + FC1_OUT*64 + FC1_OUT \
             + MATERIAL_BUCKETS*FC1_OUT + MATERIAL_BUCKETS
    n_in = None
    for cand in (31745, 22528):
        if total_exp(cand) == len(raw):
            n_in = cand; break
    if n_in is None:
        print(f"geometria não bate (len={len(raw)})"); sys.exit(1)

    ptr = [0]
    def take(shape):
        size = int(np.prod(shape))
        seg = raw[ptr[0]:ptr[0]+size]; ptr[0] += size
        return seg.reshape(shape)
    take((n_in, l1))       # accw -- não precisamos aqui (concat já vem do dump)
    take((l1,))            # accb
    take((n_in, MATERIAL_BUCKETS))  # psqtw
    take((MATERIAL_BUCKETS,))       # psqtb
    fc0w = take((FC0_TOTAL, l1))
    fc0b = take((FC0_TOTAL,))
    fc1w = take((FC1_OUT, 64))
    fc1b = take((FC1_OUT,))
    fc2w = take((MATERIAL_BUCKETS, FC1_OUT))
    fc2b = take((MATERIAL_BUCKETS,))
    return fc0w, fc0b, fc1w, fc1b, fc2w, fc2b

def load_dump(path):
    d = {}
    with open(path) as f:
        lines = f.read().strip().split("\n")
    header = lines[0].split()
    d['L1'] = int(header[0].split('=')[1])
    d['bucket'] = int(header[1].split('=')[1])
    for line in lines[1:]:
        key, vals = line.split('=', 1)
        arr = np.array([float(x) for x in vals.split(',') if x], dtype=np.float64)
        d[key] = arr
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--l1", type=int, default=512)
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    fc0w, fc0b, fc1w, fc1b, fc2w, fc2b = load_raw(args.checkpoint, args.l1)
    d = load_dump(args.dump)
    L1 = d['L1']; bucket = d['bucket']
    concat = d['concat'][:L1]  # uint8 [0,127], MESMO valor que o C++ usou no dot int8xuint8

    # 1) FC0 em float puro, a partir do concat JÁ EM [0,127] (não voltamos a derivar do
    #    acumulador -- isolamos só FC0 em diante, que é o código mais novo/menos provado).
    #    concat[i]/127 == "valor natural" (crelu().pairwise_mul(), em [0,1]) que o treino usa.
    natural = concat / 127.0
    fc0_out_float = fc0w @ natural + fc0b   # (33,)

    print("=== FC0 ===")
    print("python (float puro):", fc0_out_float[:4], " skip=", fc0_out_float[FC0_REAL])
    print("C++ (dequant):      ", d['fc0_out'][:4], " skip=", d['fc0_out'][FC0_REAL])
    diff0 = np.abs(fc0_out_float - d['fc0_out'])
    print(f"diff max FC0: {diff0.max():.6f}  (esperado: pequeno, só ruído de quantização)")

    # 2) sqrrelu (relu(x)^2, SEM clamp) + crelu (clamp 0..1) nos 32 reais, usando o
    #    fc0_out do C++ como entrada (para isolar SÓ a parte FC1/FC2 a seguir).
    real = d['fc0_out'][:FC0_REAL]
    r = np.maximum(real, 0.0)
    sqr_path = r * r
    lin_path = np.minimum(r, 1.0)
    concat64 = np.concatenate([sqr_path, lin_path])

    fc1_out_float = np.clip(fc1w @ concat64 + fc1b, 0.0, 1.0)
    print("\n=== FC1 ===")
    print("python (float puro):", fc1_out_float[:4])
    print("C++ (dequant):      ", d['fc1_out'][:4])
    diff1 = np.abs(fc1_out_float - d['fc1_out'])
    print(f"diff max FC1: {diff1.max():.6f}")

    fc2_out_float = fc2w @ d['fc1_out'] + fc2b
    print("\n=== FC2 ===")
    print("python (float puro):", fc2_out_float)
    print("C++ (dequant):      ", d['fc2_out'])
    diff2 = np.abs(fc2_out_float - d['fc2_out'])
    print(f"diff max FC2: {diff2.max():.6f}")

    final_python = fc2_out_float[bucket] + d['fc0_out'][FC0_REAL]
    print(f"\nfinal (python, bucket={bucket}): {final_python:.6f}")
    print(f"final (C++):                    {d['final_out'][0]:.6f}")

if __name__ == "__main__":
    main()
