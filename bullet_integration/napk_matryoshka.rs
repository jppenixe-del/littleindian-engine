// ════════════════════════════════════════════════════════════════════════════
// 🦅 napk_matryoshka.rs — Napoleon / Nap2Siriux
// REDE ÚNICA "Matryoshka" — substitui o monstro multi-cabeça (3 caminhos paralelos que
// divergiam e perdiam até ao 256 do trainsiriux).
//
// IDEIA (Maréchal): low = PREFIXO do high. Uma só feature transformer (accumulator de ACC_BIG=768).
//   - HIGH (raiz, preciso): lê o accumulator TODO (768) → 256 → 32 → 1×8 + psqt
//   - LOW  (folhas/bullet, rápido): lê SÓ os primeiros ACC_LOW=128 do MESMO accumulator
//                                   → 128*2 → 32 → 1×8 + psqt
//   → o low é literalmente parte do high (mesmos pesos no feature transformer). Coerência por
//     construção: nunca divergem. É a versão NNUE correta de "1 só cabeça, low até 128, high 768".
//
// TREINO: deep supervision → loss = loss(high) + LAMBDA_LOW * loss(low). Ambas contra o mesmo
//   alvo (cp+wdl). O low aprende a aproximar barato; o high refina; partilham o acc.
//
// QUANTIZAÇÃO: QA=255 (acc), QB=64 (camadas). OUTPUT_SCALE 408 (como o NapK9). 8 buckets material.
// ════════════════════════════════════════════════════════════════════════════
use bullet_lib::{
    nn::{InitSettings, Shape},
    trainer::default::{
        builder::ModelBuilder,
        ModelNode,
    },
};

pub const FEATURES: usize = 23169;   // 22528 peças + 640 threats + 1 (como o NapK9)
pub const ACC_BIG: usize = 768;      // accumulator grande (a força toda) — HIGH lê isto
pub const ACC_LOW: usize = 128;      // prefixo que o LOW lê (low = primeiros 128 do acc)
pub const BUCKETS: usize = 8;        // buckets material (mb 0..7)

// camadas finais (afunilam, estilo NNUE clássico)
pub const HIGH_L1: usize = 256;      // high: ACC_BIG*2 → 256 → 32 → 1×8
pub const HIGH_L2: usize = 32;
pub const LOW_L1:  usize = 32;       // low: ACC_LOW*2 → 32 → 1×8  (barato)

pub struct NapKMatryoshka;

impl NapKMatryoshka {
    /// Constrói o grafo. Devolve (out_high, out_low) — duas saídas da MESMA rede.
    pub fn build_graph<'a>(
        &self,
        builder: &'a ModelBuilder,
        stm: ModelNode<'a>,
        ntm: ModelNode<'a>,
        buckets: ModelNode<'a>,
    ) -> (ModelNode<'a>, ModelNode<'a>) {
        // ── Feature transformer ÚNICO (accumulator de 768) ──
        let acc_layer  = builder.new_affine("acc",  FEATURES, ACC_BIG);
        let psqt_layer = builder.new_affine("psqt", FEATURES, BUCKETS);

        // ── Camadas finais HIGH (lê o acc todo, 768) ──
        let high_l1 = builder.new_affine("high_l1", ACC_BIG * 2, HIGH_L1);
        let high_l2 = builder.new_affine("high_l2", HIGH_L1, HIGH_L2);
        let high_l3 = builder.new_affine("high_l3", HIGH_L2, 1 * BUCKETS);

        // ── Camadas finais LOW (lê só o prefixo 128 do acc) ──
        let low_l1 = builder.new_affine("low_l1", ACC_LOW * 2, LOW_L1);
        let low_l2 = builder.new_affine("low_l2", LOW_L1, 1 * BUCKETS);

        // ── Forward do accumulator (partilhado) ──
        let acc_us   = acc_layer.forward(stm).screlu();   // [768]
        let acc_them = acc_layer.forward(ntm).screlu();   // [768]

        // HIGH: concatena os 768 de ambas as perspetivas
        let x_high = acc_us.concat(acc_them);              // [1536]

        // LOW: lê SÓ os primeiros ACC_LOW (128) de cada perspetiva = prefixo do mesmo acc.
        //   slice(0, ACC_LOW) extrai os primeiros 128 neurónios → o low é parte do high.
        let low_us   = acc_us.slice_rows(0, ACC_LOW);      // [128]
        let low_them = acc_them.slice_rows(0, ACC_LOW);    // [128]
        let x_low = low_us.concat(low_them);               // [256]

        // PSQT (forward separado por saída p/ evitar "Cycle found" no otimizador de grafos)
        let psqt_h = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);
        let psqt_l = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);

        // HIGH: 1536 → 256 → 32 → 1×8 + psqt
        let h1 = high_l1.forward(x_high).screlu();
        let h2 = high_l2.forward(h1).screlu();
        let out_high = high_l3.forward(h2).select(buckets) + psqt_h;

        // LOW: 256 → 32 → 1×8 + psqt
        let l1 = low_l1.forward(x_low).screlu();
        let out_low = low_l2.forward(l1).select(buckets) + psqt_l;

        (out_high, out_low)
    }
}
