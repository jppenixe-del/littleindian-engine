// 🦅 napk9.rs — versão SAÍDA ÚNICA (Opção B): treina UMA cabeça de cada vez.
//   EVITA o "Cycle found" (que vem de 3 saídas partilharem nós). Cada treino produz 1
//   cabeça (+ chaos opcional). Corre 3× (bullet/small/big) e junta no auto_converter.
//   A cabeça é escolhida por env NAPK_HEAD = "bullet" | "small" | "big" (default big).
use crate::nn::{ModelBuilder, ModelNode};
use crate::value::loader::{GameResult, CanBeDirectlySequentiallyLoaded};

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct NapkRecord {
    pub us_feats: [u16; 64], pub them_feats: [u16; 64],
    pub us_count: u8, pub them_count: u8, pub bucket: u8, pub padding: u8,
    pub score: f32, pub wdl: f32,
}
impl Default for NapkRecord {
    fn default() -> Self { Self { us_feats:[0;64], them_feats:[0;64], us_count:0, them_count:0, bucket:0, padding:0, score:0.0, wdl:0.0 } }
}
unsafe impl CanBeDirectlySequentiallyLoaded for NapkRecord {}
impl crate::value::loader::LoadableDataType for NapkRecord {
    #[inline] fn score(&self) -> i16 { self.score as i16 }
    #[inline] fn result(&self) -> GameResult {
        if self.wdl > 0.75 { GameResult::Win } else if self.wdl < 0.25 { GameResult::Loss } else { GameResult::Draw }
    }
}
#[derive(Clone, Copy, Default)]
pub struct NapkMaterialBuckets<const N: usize>;
impl<const N: usize> crate::game::outputs::OutputBuckets<NapkRecord> for NapkMaterialBuckets<N> {
    const BUCKETS: usize = N;
    #[inline] fn bucket(&self, pos: &NapkRecord) -> u8 { pos.bucket }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct NapkInput23168;
impl crate::game::inputs::SparseInputType for NapkInput23168 {
    type RequiredDataType = NapkRecord;
    fn num_inputs(&self) -> usize { 23169 }
    fn max_active(&self) -> usize { 64 }
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let mx = std::cmp::max(pos.us_count, pos.them_count) as usize;
        let d = 23168;
        for i in 0..mx {
            let s = if i < pos.us_count as usize { pos.us_feats[i] as usize } else { d };
            let n = if i < pos.them_count as usize { pos.them_feats[i] as usize } else { d };
            f(s, n);
        }
    }
    fn shorthand(&self) -> String { "23169".to_string() }
    fn description(&self) -> String { "Napk V9 Asymmetric".to_string() }
}

pub struct NapKa0sV9 { pub l1_size: usize, pub head: &'static str }

impl NapKa0sV9 {
    pub fn detect_big_l2(l1: usize) -> usize {
        if l1 <= 128 {16} else if l1 <= 256 {32} else if l1 <= 512 {32}
        else if l1 <= 768 {48} else if l1 <= 1024 {64} else {96}
    }
    // 1 cabeça de avaliação + chaos. Sem 3 saídas → sem ciclo.
    pub fn build_graph<'a>(&self, builder: &'a ModelBuilder, stm: ModelNode<'a>,
                           ntm: ModelNode<'a>, buckets: ModelNode<'a>)
                           -> (ModelNode<'a>, ModelNode<'a>) {
        let l1 = self.l1_size;
        let (h1_out, h2_out, prefix) = match self.head {
            "bullet" => (16usize, 32usize, "bullet"),
            "small"  => (32usize, 32usize, "small"),
            _         => (l1, Self::detect_big_l2(l1), "g"),   // big → ids g_l1/g_l2/g_l3
        };
        let acc  = builder.new_affine("acc",  23169, l1);
        let psqt = builder.new_affine("psqt", 23169, 8);
        let h_l1 = builder.new_affine(&format!("{}_l1", prefix), l1 * 2, h1_out);
        let h_l2 = builder.new_affine(&format!("{}_l2", prefix), h1_out, h2_out);
        let h_l3 = builder.new_affine(&format!("{}_l3", prefix), h2_out, 1 * 8);
        let chaos_l1 = builder.new_affine("chaos_l1", l1 * 2, 32);
        let chaos_l2 = builder.new_affine("chaos_l2", 32, 16);
        let chaos_l3 = builder.new_affine("chaos_l3", 16, 1);

        let acc_us = acc.forward(stm).screlu();
        let acc_them = acc.forward(ntm).screlu();
        let x = acc_us.concat(acc_them);
        let psqt_bias = psqt.forward(stm).select(buckets) - psqt.forward(ntm).select(buckets);

        let a1 = h_l1.forward(x).screlu();
        let a2 = h_l2.forward(a1).screlu();
        let out_eval = h_l3.forward(a2).select(buckets) + psqt_bias;

        let c1 = chaos_l1.forward(x).screlu();
        let c2 = chaos_l2.forward(c1).screlu();
        let out_chaos = chaos_l3.forward(c2);

        (out_eval, out_chaos)
    }
}
