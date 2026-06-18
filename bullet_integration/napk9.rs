use crate::nn::{ModelBuilder, ModelNode};
use crate::value::loader::{GameResult, CanBeDirectlySequentiallyLoaded};

/// 🦅 Estrutura binária de 268 bytes equivalente ao Struct.pack do Python
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct NapkRecord {
    pub us_feats: [u16; 64],
    pub them_feats: [u16; 64],
    pub us_count: u8,
    pub them_count: u8,
    pub bucket: u8,
    pub padding: u8,
    pub score: f32,
    pub wdl: f32,
}

impl Default for NapkRecord {
    fn default() -> Self {
        Self {
            us_feats: [0; 64],
            them_feats: [0; 64],
            us_count: 0,
            them_count: 0,
            bucket: 0,
            padding: 0,
            score: 0.0,
            wdl: 0.0,
        }
    }
}

unsafe impl CanBeDirectlySequentiallyLoaded for NapkRecord {}

impl crate::value::loader::LoadableDataType for NapkRecord {
    #[inline]
    fn score(&self) -> i16 { self.score as i16 }

    #[inline]
    fn result(&self) -> GameResult {
        if self.wdl > 0.75 { GameResult::Win }
        else if self.wdl < 0.25 { GameResult::Loss }
        else { GameResult::Draw }
    }
}

#[derive(Clone, Copy, Default)]
pub struct NapkMaterialBuckets<const N: usize>;

impl<const N: usize> crate::game::outputs::OutputBuckets<NapkRecord> for NapkMaterialBuckets<N> {
    const BUCKETS: usize = N;
    #[inline]
    fn bucket(&self, pos: &NapkRecord) -> u8 { pos.bucket }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NapkInput23168;

impl crate::game::inputs::SparseInputType for NapkInput23168 {
    type RequiredDataType = NapkRecord;

    fn num_inputs(&self) -> usize { 23169 }
    fn max_active(&self) -> usize { 64 }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let max_iterations = std::cmp::max(pos.us_count, pos.them_count) as usize;
        let dummy_index = 23168;
        for i in 0..max_iterations {
            let stm = if i < pos.us_count as usize { pos.us_feats[i] as usize } else { dummy_index };
            let ntm = if i < pos.them_count as usize { pos.them_feats[i] as usize } else { dummy_index };
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String { "23169".to_string() }
    fn description(&self) -> String { "Napk V9 Asymmetric Feature Space".to_string() }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 🦅 NapKa0sV9 — AGORA COM AS 3 CABEÇAS (bullet 16 / small 32 / big) + Chaos
// ═══════════════════════════════════════════════════════════════════════════════
//   Alinhado À GEOMETRIA DO MOTOR (nnue_net.cpp linhas 233-235 + detectBigL2):
//     BULLET: L1*2 → 16 → 32 → 1×8 buckets   (BULLET_L1=16, BULLET_L2=32)
//     SMALL : L1*2 → 32 → 32 → 1×8 buckets   (SMALL_L1=32,  SMALL_L2=32)
//     BIG   : L1*2 → L1 → bigL2 → 1×8 buckets (bigL1=L1, bigL2=detectBigL2(L1))
//     CHAOS : L1*2 → 32 → 16 → 1
//   ⚠️ MUDANÇA vs versão anterior: a BIG já NÃO usa big_size=512. A 1ª camada da big tem
//      l1_out = L1 (igual ao motor), e a 2ª tem bigL2. Assim os pesos exportados batem
//      EXATAMENTE com o que o motor carrega (sem zero-padding de geometria falsa).
//   As 3 cabeças partilham o MESMO acumulador (acc) e o MESMO PSQT — só os batalhões
//   densos diferem. Treino: a loss soma as 3 cabeças de avaliação + a chaos.
pub struct NapKa0sV9 {
    pub l1_size: usize,
}

impl NapKa0sV9 {
    // bigL2 idêntico ao detectBigL2 do motor (nnue_net.cpp:136)
    pub fn detect_big_l2(l1: usize) -> usize {
        if l1 <= 128 { 16 }
        else if l1 <= 256 { 32 }
        else if l1 <= 512 { 32 }
        else if l1 <= 768 { 48 }
        else if l1 <= 1024 { 64 }
        else { 96 }
    }

    pub fn build_graph<'a>(
        &self,
        builder: &'a ModelBuilder,
        stm: ModelNode<'a>,
        ntm: ModelNode<'a>,
        buckets: ModelNode<'a>,
    ) -> (ModelNode<'a>, ModelNode<'a>, ModelNode<'a>, ModelNode<'a>) {
        let l1 = self.l1_size;
        let big_l2 = Self::detect_big_l2(l1);

        // ── Factories ────────────────────────────────────────────────────────────
        let acc_layer  = builder.new_affine("acc",  23169, l1);
        let psqt_layer = builder.new_affine("psqt", 23169, 8);

        // BULLET (g = "bullet"): L1*2 → 16 → 32 → 1×8
        let bullet_l1 = builder.new_affine("bullet_l1", l1 * 2, 16);
        let bullet_l2 = builder.new_affine("bullet_l2", 16, 32);
        let bullet_l3 = builder.new_affine("bullet_l3", 32, 1 * 8);

        // SMALL: L1*2 → 32 → 32 → 1×8
        let small_l1 = builder.new_affine("small_l1", l1 * 2, 32);
        let small_l2 = builder.new_affine("small_l2", 32, 32);
        let small_l3 = builder.new_affine("small_l3", 32, 1 * 8);

        // BIG / Austerlitz: L1*2 → L1 → bigL2 → 1×8  (alinhado ao motor!)
        let big_l1 = builder.new_affine("g_l1", l1 * 2, l1);
        let big_l2_layer = builder.new_affine("g_l2", l1, big_l2);
        let big_l3 = builder.new_affine("g_l3", big_l2, 1 * 8);

        // CHAOS: L1*2 → 32 → 16 → 1
        let chaos_l1 = builder.new_affine("chaos_l1", l1 * 2, 32);
        let chaos_l2 = builder.new_affine("chaos_l2", 32, 16);
        let chaos_l3 = builder.new_affine("chaos_l3", 16, 1);

        // ── Forward (acumulador partilhado; PSQT com forward SEPARADO por cabeça) ──
        let acc_us = acc_layer.forward(stm).screlu();
        let acc_them = acc_layer.forward(ntm).screlu();
        let x = acc_us.concat(acc_them);

        // 🦅 FIX "Cycle found": cada cabeça faz o SEU próprio forward do psqt_layer (nós
        //   distintos no grafo), em vez de partilharem UM psqt_bias. O otimizador de grafos
        //   do bullet (FusePointwise/SwapOutputs) entrava em ciclo quando 3 saídas somavam
        //   o MESMO nó psqt_bias. Os PESOS do psqt_layer são os mesmos (1 só psqt no save),
        //   só o nó do forward é que se duplica → grafo linearizável, sem ciclo.
        let psqt_bias_b = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);
        let psqt_bias_s = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);
        let psqt_bias_g = psqt_layer.forward(stm).select(buckets) - psqt_layer.forward(ntm).select(buckets);

        // BULLET
        let b1 = bullet_l1.forward(x).screlu();
        let b2 = bullet_l2.forward(b1).screlu();
        let out_bullet = bullet_l3.forward(b2).select(buckets) + psqt_bias_b;

        // SMALL
        let s1 = small_l1.forward(x).screlu();
        let s2 = small_l2.forward(s1).screlu();
        let out_small = small_l3.forward(s2).select(buckets) + psqt_bias_s;

        // BIG
        let g1 = big_l1.forward(x).screlu();
        let g2 = big_l2_layer.forward(g1).screlu();
        let out_big = big_l3.forward(g2).select(buckets) + psqt_bias_g;

        // CHAOS (não usa psqt)
        let c1 = chaos_l1.forward(x).screlu();
        let c2 = chaos_l2.forward(c1).screlu();
        let out_chaos = chaos_l3.forward(c2);

        (out_bullet, out_small, out_big, out_chaos)
    }
}
