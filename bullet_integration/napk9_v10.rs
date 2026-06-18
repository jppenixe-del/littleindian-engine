// 🦅 napk9_v10.rs — módulo BULLET do NapK9 V10 (record V2 + features on-the-fly).
//
//   FILOSOFIA (decisão do Maréchal): o .data2 guarda a POSIÇÃO (bitboards, 112B), não as
//   features. O loader gera peças+FULL THREATS on-the-fly com o MESMO código calibrado que o
//   motor usa em jogo (napk9_v10_features.rs, harness 12/12). Muda-se o feature set → recompila
//   o loader, SEM reconverter os dados.
//
//   INSTALAÇÃO no bullet (PC): copiar este ficheiro E o napk9_v10_features.rs para src/ do
//   bullet (lado a lado — o include! da linha 18 apanha o features).
//   ⚠️⚠️ NO lib.rs REGISTAR APENAS `pub mod napk9_v10;`. NUNCA `mod napk9_v10_features;` —
//      o features ENTRA pelo include!, NÃO é um módulo. Os dois (mod + include) do MESMO
//      ficheiro = dupla compilação → "NapkInputV10 has no field mode" (erro enganador, s29).
//   trainer: .inputs(NapkInputV10 { mode }) com new_affine("acc", num_inputs, l1).
//
//   ⚠️ NUNCA editar as funções core aqui — vivem no napk9_v10_features.rs (1 só verdade).

use crate::value::loader::{GameResult, CanBeDirectlySequentiallyLoaded};

// o CORE calibrado (Pos, gathers, map_features_pairs, map_features_pairs_mode, TOTAL_INPUTS_NONE, TOTAL_INPUTS_640, constantes) — 1 só ficheiro de verdade.
// O main() do ficheiro incluído fica morto aqui (não é o crate root), sem conflito.
include!("napk9_v10_features.rs");

/// NapkRecordV2 — 112 bytes: a POSIÇÃO + meta. [cor][tipo], cor 0=BRANCA, tipos P,N,B,R,Q,K.
/// score/wdl no POV do STM (como o NapkRecord clássico). bucket = material bucket pré-computado.
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct NapkRecordV2 {
    pub bb: [[u64; 6]; 2],   // 96 B — alimenta DIRETAMENTE o Pos do gerador calibrado
    pub stm: u8,             // 0 = brancas a jogar
    pub bucket: u8,          // mb = ((npieces-1)/4).clamp(0,7)
    pub pad: [u8; 2],
    pub score: f32,          // cp, POV stm
    pub wdl: f32,            // POV stm
    pub pad2: [u8; 4]        // → 112 bytes exatos (= o que o converter --v2 escreve; alinhado a 8)
}

impl Default for NapkRecordV2 {
    fn default() -> Self {
        Self { bb: [[0; 6]; 2], stm: 0, bucket: 0, pad: [0; 2], score: 0.0, wdl: 0.0, pad2: [0; 4]}
    }
}

unsafe impl CanBeDirectlySequentiallyLoaded for NapkRecordV2 {}

impl crate::value::loader::LoadableDataType for NapkRecordV2 {
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
pub struct NapkV2MaterialBuckets<const N: usize>;

impl<const N: usize> crate::game::outputs::OutputBuckets<NapkRecordV2> for NapkV2MaterialBuckets<N> {
    const BUCKETS: usize = N;
    #[inline]
    fn bucket(&self, pos: &NapkRecordV2) -> u8 { pos.bucket }
}

/// Input V10: 31744 features (22528 peças + 9216 full threats) + 1 dummy = 31745.
/// max_active: 31 pares de peças + threats (típico 40-60, raro >100) → 192 com folga.
#[derive(Clone, Copy, Debug)]
pub struct NapkInputV10 { pub mode: u8 }   // 🦅 s29: 0=none, 1=threats 640, 2=full 9216
impl Default for NapkInputV10 { fn default() -> Self { Self { mode: 2 } } }

impl crate::game::inputs::SparseInputType for NapkInputV10 {
    type RequiredDataType = NapkRecordV2;

    fn num_inputs(&self) -> usize {
        1 + match self.mode { 0 => TOTAL_INPUTS_NONE, 1 => TOTAL_INPUTS_640, _ => TOTAL_INPUTS_V10 }
    }
    fn max_active(&self) -> usize { match self.mode { 0 => 64, 1 => 96, _ => 192 } }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let p = Pos { pieces: pos.bb };
        let cap = self.max_active();
        let mut n = 0usize;
        map_features_pairs_mode(&p, pos.stm as usize, self.mode, &mut |a, b| {
            if n < cap { f(a, b); n += 1; }   // clamp de segurança (posições patológicas)
        });
    }

    fn shorthand(&self) -> String { format!("napk9v10-m{}-{}", self.mode, self.num_inputs()) }
    fn description(&self) -> String {
        "NapK9 V10: HalfK2 22528 + full threats 9216, geradas on-the-fly dos bitboards".to_string()
    }
}
