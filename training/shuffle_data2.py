#!/usr/bin/env python3
# 🦅 shuffle_data2.py — baralha um .data2 (NapkRecordV2 112B) NO DISCO, em 2 passes.
#
#   PORQUÊ (lição s29, a causa dos 2 desastres — syzygy e aberturas): o nosso .data2 sai do
#   converter ORDENADO POR JOGOS (posições consecutivas quase idênticas, aberturas em clusters,
#   finais TB em blocos). O DirectSequentialDataLoader do bullet é sequencial PURO (131 linhas,
#   zero shuffle — verificado na fonte). Batches correlacionados → gradientes dominados por
#   posições repetidas → evals explosivas nas heads dos buckets concentrados (b7 aberturas,
#   b1 finais). O ecossistema bullet baralha SEMPRE (o repo tem crates/utils/src/shuffle.rs;
#   os loaders streaming têm shuffle_buffer). Este script dá-nos o mesmo, para o nosso formato.
#
#   USO: python3 training/shuffle_data2.py in.data2 out_shuffled.data2 [--shards 64] [--seed 7]
#   2 passes: (1) stream → shard aleatório; (2) cada shard baralhado em RAM → concatena.
#   RAM: ~ tamanho_ficheiro/shards por shard (14.6GB/64 ≈ 230MB ✅ p/ 16GB).
#   ⚠️ usa um diretório temporário no MESMO filesystem do output (rápido); em WSL preferir ext4.

import argparse, os, sys, tempfile
import numpy as np

REC = 112

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tmpdir", default=None, help="default: pasta do output")
    a = ap.parse_args()

    size = os.path.getsize(a.inp)
    n = size // REC
    if size % REC:
        print(f"🔴 {a.inp}: {size} bytes não é múltiplo de {REC} — ficheiro cortado a meio de um registo!")
        sys.exit(1)
    print(f"🦅 shuffle: {n/1e6:.1f}M registos, {a.shards} shards, seed {a.seed}")

    tmpdir = tempfile.mkdtemp(prefix="napk_shuf_", dir=a.tmpdir or os.path.dirname(os.path.abspath(a.out)) or ".")
    shards = [open(os.path.join(tmpdir, f"s{i:03d}"), "wb", buffering=1 << 22) for i in range(a.shards)]
    rng = np.random.default_rng(a.seed)

    # ── passo 1: distribuir por shards (streaming, 8M registos por bloco ≈ 900MB) ──
    CHUNK = 8_000_000
    feito = 0
    with open(a.inp, "rb", buffering=1 << 24) as f:
        while True:
            blk = f.read(CHUNK * REC)
            if not blk: break
            m = len(blk) // REC
            arr = np.frombuffer(blk, dtype=np.uint8).reshape(m, REC)
            dest = rng.integers(0, a.shards, m)
            for s in range(a.shards):
                sel = arr[dest == s]
                if len(sel): shards[s].write(sel.tobytes())
            feito += m
            print(f"   passo 1: {feito/1e6:.0f}M / {n/1e6:.0f}M", end="\r")
    for s in shards: s.close()
    print(f"\n   passo 1 ✅ ({a.shards} shards em {tmpdir})")

    # ── passo 2: baralhar cada shard em RAM e concatenar ──
    escrito = 0
    with open(a.out, "wb", buffering=1 << 24) as out:
        for i in range(a.shards):
            p = os.path.join(tmpdir, f"s{i:03d}")
            arr = np.fromfile(p, dtype=np.uint8)
            m = len(arr) // REC
            arr = arr.reshape(m, REC)
            arr = arr[rng.permutation(m)]
            out.write(arr.tobytes())
            escrito += m
            os.remove(p)
            print(f"   passo 2: shard {i+1}/{a.shards} ({escrito/1e6:.0f}M)", end="\r")
    os.rmdir(tmpdir)
    print()
    if escrito != n:
        print(f"🔴 PERDA: {n} → {escrito} registos!"); sys.exit(1)
    print(f"✅ {a.out}: {escrito/1e6:.1f}M registos baralhados (mesmo conteúdo, ordem nova)")
    print(f"   treina com: NAPK_DATA={a.out} ...")

if __name__ == "__main__":
    main()
