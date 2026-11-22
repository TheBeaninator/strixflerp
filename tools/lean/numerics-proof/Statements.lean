/-
NUMERICS-PROOF. Statements: the file a reviewer must trust.

Every theorem below ends in "sorry" here. Proofs.lean repeats each definition and statement verbatim and proves it;
"lake comparator" checks that the two match and that the proofs use only propext, Classical.choice and Quot.sound.

Sources the statements are checked against:
* numerics.json 1.7.0 global.ambiguity_rule.topk_sets / argmax and router.noaux_tc procedure step 4 (the old rule);
* a fork branch's tools/dsv41-hip/tri/band_rule.py (the exact rule: set_ok, classify, argmax_ok);
* fork pod-main-2026-11 a7dfc277 tools/dsv41-hip/tri/t_route.py selection / set_ok (the old rule as run);
* numerics.json Engram lookup "e4m3 rows x UE8M0 scales (exact; also exact in bf16)", quant.fp8 act_quant
  "s = 2^ceil(log2(amax/448)) (UE8M0)", quant.fp4 "e2m1 ... UE8M0 scale".
-/
import Mathlib

namespace NumericsProof

open Finset

/-! ## Part 1. The exact top-k band rule (band_rule.py)

Positions form a finite type ι (the n experts / tokens of one row). s = reference scores, b = per-position error
bands (b t bounds |s_gpu t - s_ref t|). -/

section TopK

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- g is a GPU score vector inside the bands: |g t - s t| ≤ b t for every position t. -/
def InBand (s b g : ι → ℝ) : Prop := ∀ t, |g t - s t| ≤ b t

/-- S is an output torch.topk(g, k) may return: k distinct positions, none of them below a position left out
(torch.topk sets no tie order, so every tie order is allowed). -/
def IsTopK (g : ι → ℝ) (k : ℕ) (S : Finset ι) : Prop :=
  S.card = k ∧ ∀ t ∈ S, ∀ u ∉ S, g u ≤ g t

/-- S is the top-k of some in-band score vector. -/
def Reachable (s b : ι → ℝ) (k : ℕ) (S : Finset ι) : Prop :=
  ∃ g, InBand s b g ∧ IsTopK g k S

/-- band_rule.set_ok: accept the k picks S iff min over S of (s + b) ≥ max over the rest of (s - b), written
pairwise (theorem exact_accept_iff_inf_sup gives the min/max form). The card condition is set_ok's
"k distinct ids in range" check (a Finset has distinct members, all in range). -/
def ExactAccept (s b : ι → ℝ) (k : ℕ) (S : Finset ι) : Prop :=
  S.card = k ∧ ∀ t ∈ S, ∀ u ∉ S, s u - b u ≤ s t + b t

/-! ## Part 2. The old rule (numerics.json 1.5.0-1.7.0 topk_sets text, t_route.py selection + set_ok)

c = the k-th largest reference score, held by position p (t_route: p = ref_picks[:, k-1]), b_c = b p.
MUST = {e : s e > c + b e + b_c}, FREE = {e : |s e - c| ≤ b e + b_c} minus MUST; accept S iff |S| = k,
MUST ⊆ S and S ⊆ MUST ∪ FREE. -/

/-- p holds a k-th largest reference score: fewer than k scores are strictly above it, at least k are ≥ it. -/
def IsKthPos (s : ι → ℝ) (k : ℕ) (p : ι) : Prop :=
  (univ.filter fun u => s p < s u).card < k ∧ k ≤ (univ.filter fun u => s p ≤ s u).card

def OldMust (s b : ι → ℝ) (p e : ι) : Prop := s p + b e + b p < s e

def OldFree (s b : ι → ℝ) (p e : ι) : Prop := |s e - s p| ≤ b e + b p ∧ ¬ OldMust s b p e

def OldAccept (s b : ι → ℝ) (k : ℕ) (p : ι) (S : Finset ι) : Prop :=
  S.card = k ∧ (∀ e, OldMust s b p e → e ∈ S) ∧ (∀ e ∈ S, OldMust s b p e ∨ OldFree s b p e)

end TopK

/-! Concrete rows (numerics-topk.md table "The two counterexamples through the real code"; positions are
Fin n, position 0 = A, 1 = B, ...). -/

/-- CE1, must-select side, k = 2: A 10 (0.1), B 8 (0.1), C 7.5 (2.5), D 7.6 (2.5); GPU 9.9, 7.9, 10.0, 10.1. -/
def ce1_s : Fin 4 → ℝ := ![10, 8, 7.5, 7.6]
def ce1_b : Fin 4 → ℝ := ![0.1, 0.1, 2.5, 2.5]
def ce1_g : Fin 4 → ℝ := ![9.9, 7.9, 10.0, 10.1]

/-- CE2, must-not side, k = 2: A 10 (3), B 8 (0.1), E 7.5 (0.2); GPU 7.0, 8.0, 7.7. -/
def ce2_s : Fin 3 → ℝ := ![10, 8, 7.5]
def ce2_b : Fin 3 → ℝ := ![3, 0.1, 0.2]
def ce2_g : Fin 3 → ℝ := ![7.0, 8.0, 7.7]

/-- FP1, equal bands 1, k = 2: A 12, K 10, X 8. -/
def fp1_s : Fin 3 → ℝ := ![12, 10, 8]

/-! ## Part 3. Number formats, bit for bit

Each decoder maps a code (a natural number below 2^width, read as the bit pattern) to its finite value as an
exact rational, or none for NaN / infinity. -/

/-- IEEE 754 binary format with eb exponent bits and mb stored mantissa bits: sign = bit (eb + mb), exponent
field = the next eb bits, bias 2^(eb-1) - 1; exponent field all ones = infinity or NaN (none); field 0 =
subnormal (0.m × 2^(1-bias)); otherwise 1.m × 2^(e-bias). bf16 = (8, 7), fp32 = (8, 23). -/
def ieeeDecode (eb mb c : ℕ) : Option ℚ :=
  let sgn : ℚ := if c / 2 ^ (eb + mb) % 2 = 1 then -1 else 1
  let e := c / 2 ^ mb % 2 ^ eb
  let m := c % 2 ^ mb
  let bias : ℤ := 2 ^ (eb - 1) - 1
  if e = 2 ^ eb - 1 then none
  else if e = 0 then some (sgn * ((m : ℚ) / 2 ^ mb) * (2 : ℚ) ^ (1 - bias))
  else some (sgn * (1 + (m : ℚ) / 2 ^ mb) * (2 : ℚ) ^ ((e : ℤ) - bias))

/-- The finite bf16 values. -/
def bf16Values : Set ℚ := {q | ∃ c < 2 ^ 16, ieeeDecode 8 7 c = some q}

/-- The finite fp32 values. -/
def fp32Values : Set ℚ := {q | ∃ c < 2 ^ 32, ieeeDecode 8 23 c = some q}

/-- OCP FP8 E4M3 (torch float8_e4m3fn): sign = bit 7, exponent = bits 6..3 (bias 7), mantissa = bits 2..0.
No infinities; S.1111.111 is NaN; everything else is finite (S.1111.110 = ±448). -/
def e4m3Decode (c : ℕ) : Option ℚ :=
  let sgn : ℚ := if c / 128 % 2 = 1 then -1 else 1
  let e := c / 8 % 16
  let m := c % 8
  if e = 15 ∧ m = 7 then none
  else if e = 0 then some (sgn * ((m : ℚ) / 8) * (2 : ℚ) ^ (-6 : ℤ))
  else some (sgn * (1 + (m : ℚ) / 8) * (2 : ℚ) ^ ((e : ℤ) - 7))

def e4m3Values : Set ℚ := {q | ∃ c < 256, e4m3Decode c = some q}

/-- OCP FP4 E2M1: sign = bit 3, exponent = bits 2..1 (bias 1), mantissa = bit 0; no infinities, no NaN.
Values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. -/
def e2m1Decode (c : ℕ) : ℚ :=
  let sgn : ℚ := if c / 8 % 2 = 1 then -1 else 1
  let e := c / 2 % 4
  let m := c % 2
  if e = 0 then sgn * ((m : ℚ) / 2)
  else sgn * (1 + (m : ℚ) / 2) * (2 : ℚ) ^ ((e : ℤ) - 1)

def e2m1Values : Set ℚ := {q | ∃ c < 16, e2m1Decode c = q}

/-- OCP UE8M0 scale: an 8-bit biased exponent, code c < 255 is 2^(c - 127); code 255 is NaN. -/
def ue8m0Values : Set ℚ := {q | ∃ c < 255, q = (2 : ℚ) ^ ((c : ℤ) - 127)}

/-- The act_quant / fp4_act_quant scale over the reals: s = 2^ceil(log2(amax / qmax)) (qmax = 448 for E4M3,
6 for E2M1). -/
noncomputable def ue8m0Scale (qmax amax : ℝ) : ℝ := (2 : ℝ) ^ ⌈Real.logb 2 (amax / qmax)⌉

/-! ## Statement 1: the exact rule decides reachability exactly -/

section TopKThms

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- 1a. No false fail: every pick set some in-band score vector produces is accepted (no band sign needed). -/
theorem exact_no_false_fail (s b : ι → ℝ) (k : ℕ) (S : Finset ι) :
    Reachable s b k S → ExactAccept s b k S := by
  sorry

/-- 1b. No false pass: every accepted pick set is produced by some in-band score vector (bands ≥ 0). -/
theorem exact_no_false_pass (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (k : ℕ) (S : Finset ι) :
    ExactAccept s b k S → Reachable s b k S := by
  sorry

/-- 1c. The pairwise form is set_ok's min/max form whenever both S and its complement are nonempty
(set_ok accepts outright when S is everything). -/
theorem exact_accept_iff_inf_sup (s b : ι → ℝ) (S : Finset ι) (hS : S.Nonempty) (hT : Sᶜ.Nonempty) :
    (∀ t ∈ S, ∀ u ∉ S, s u - b u ≤ s t + b t) ↔
      Sᶜ.sup' hT (fun u => s u - b u) ≤ S.inf' hS (fun t => s t + b t) := by
  sorry

/-- 1d. band_rule.classify's may (= must or free): t is in some reachable pick set iff fewer than k positions have
a lower end above t's upper end (numpy: s t + b t ≥ the k-th largest lower end). -/
theorem classify_may (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (k : ℕ) (hk : k ≤ Fintype.card ι) (t : ι) :
    (∃ S, Reachable s b k S ∧ t ∈ S) ↔
      (univ.filter fun u => s t + b t < s u - b u).card < k := by
  sorry

/-- 1e. band_rule.classify's must: t is in every reachable pick set iff fewer than k other positions have an
upper end at or above t's lower end (numpy: s t - b t > the (k+1)-th largest upper end). -/
theorem classify_must (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (k : ℕ) (t : ι) :
    (∀ S, Reachable s b k S → t ∈ S) ↔
      (univ.filter fun u => u ≠ t ∧ s t - b t ≤ s u + b u).card < k := by
  sorry

/-- 1f. band_rule.argmax_ok (k = 1): token a is the argmax of some in-band score vector iff its upper end is at
least every other token's lower end. -/
theorem argmax_exact (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (a : ι) :
    (∀ u, u ≠ a → s u - b u ≤ s a + b a) ↔ ∃ g, InBand s b g ∧ ∀ u, g u ≤ g a := by
  sorry

/-! ## Statement 2: the old summed-band rule -/

/-- 2d. With one band β for every position the old rule never fails a reachable pick set. -/
theorem old_rule_sound_equal_bands (s : ι → ℝ) (β : ℝ) (k : ℕ) (p : ι) (S : Finset ι)
    (hp : IsKthPos s k p) :
    Reachable s (fun _ => β) k S → OldAccept s (fun _ => β) k p S := by
  sorry

end TopKThms

/-- 2a. CE1 (lean-spike section 2.3): B is the 2nd largest, the GPU vector is in band, its top 2 is {C, D},
and the old rule rejects {C, D} (A is MUST). -/
theorem old_rule_false_fail_ce1 :
    IsKthPos ce1_s 2 1 ∧ InBand ce1_s ce1_b ce1_g ∧ IsTopK ce1_g 2 {2, 3} ∧
      ¬ OldAccept ce1_s ce1_b 2 1 {2, 3} := by
  sorry

/-- 2b. CE2: the GPU vector is in band, its top 2 is {B, E}, and the old rule rejects it (E is neither MUST
nor FREE). -/
theorem old_rule_false_fail_ce2 :
    IsKthPos ce2_s 2 1 ∧ InBand ce2_s ce2_b ce2_g ∧ IsTopK ce2_g 2 {1, 2} ∧
      ¬ OldAccept ce2_s ce2_b 2 1 {1, 2} := by
  sorry

/-- 2c. Hence the old rule is not sound once bands differ: "every reachable pick set is accepted" is false. -/
theorem old_rule_unsound :
    ¬ ∀ (s b : Fin 4 → ℝ) (k : ℕ) (p : Fin 4) (S : Finset (Fin 4)),
        (∀ t, 0 ≤ b t) → IsKthPos s k p → Reachable s b k S → OldAccept s b k p S := by
  sorry

/-- 2e. FP1: even with equal bands the old rule accepts {K, X}, which no in-band vector produces. -/
theorem old_rule_false_pass_fp1 :
    IsKthPos fp1_s 2 1 ∧ OldAccept fp1_s (fun _ => 1) 2 1 {1, 2} ∧
      ¬ Reachable fp1_s (fun _ => 1) 2 {1, 2} := by
  sorry

/-! ## Statement 3: format lemmas -/

/-- 3a, as written ("e4m3 x UE8M0 ... also exact in bf16" for every scale): false. -/
theorem e4m3_ue8m0_bf16_literal_false :
    ¬ ∀ x ∈ e4m3Values, ∀ s ∈ ue8m0Values, x * s ∈ bf16Values := by
  sorry

/-- 3a, exact range: every e4m3 value times the UE8M0 scale with code c (2^(c-127)) is a bf16 value iff
3 ≤ c ≤ 246, i.e. scale exponents -124 .. 119. -/
theorem e4m3_ue8m0_bf16_iff (c : ℕ) (hc : c < 255) :
    (∀ x ∈ e4m3Values, x * (2 : ℚ) ^ ((c : ℤ) - 127) ∈ bf16Values) ↔ 3 ≤ c ∧ c ≤ 246 := by
  sorry

/-- 3a, the two edges: 448 × 2^120 overflows bf16 and 2^-9 × 2^-125 = 2^-134 is below its smallest subnormal. -/
theorem e4m3_ue8m0_bf16_edges :
    (448 : ℚ) ∈ e4m3Values ∧ (2 : ℚ) ^ (120 : ℤ) ∈ ue8m0Values ∧ (448 : ℚ) * 2 ^ (120 : ℤ) ∉ bf16Values ∧
    (2 : ℚ) ^ (-9 : ℤ) ∈ e4m3Values ∧ (2 : ℚ) ^ (-125 : ℤ) ∈ ue8m0Values ∧
      (2 : ℚ) ^ (-9 : ℤ) * 2 ^ (-125 : ℤ) ∉ bf16Values := by
  sorry

/-- 3b. The UE8M0 scale keeps every element of the block within qmax: |x| ≤ amax implies |x / s| ≤ qmax
(qmax = 448 for act_quant). -/
theorem ue8m0Scale_bound (qmax amax x : ℝ) (hq : 0 < qmax) (ha : 0 < amax) (hx : |x| ≤ amax) :
    |x / ue8m0Scale qmax amax| ≤ qmax := by
  sorry

/-- 3b'. It is the smallest power of two that does: amax / s > qmax / 2. -/
theorem ue8m0Scale_tight (qmax amax : ℝ) (hq : 0 < qmax) (ha : 0 < amax) :
    qmax / 2 < amax / ue8m0Scale qmax amax := by
  sorry

/-- 3b''. For act_quant's amax (floored at 1e-4, at most 2^128) the scale exponent lies in -22 .. 120, so the
scale is a UE8M0 value (codes 105 .. 247). -/
theorem ue8m0Scale_e4m3_exponent (amax : ℝ) (h1 : 1 / 10000 ≤ amax) (h2 : amax ≤ 2 ^ 128) :
    -22 ≤ ⌈Real.logb 2 (amax / 448)⌉ ∧ ⌈Real.logb 2 (amax / 448)⌉ ≤ 120 := by
  sorry

/-- 3c, as written ("E2M1 x UE8M0 exact in fp32" for every scale): false. -/
theorem e2m1_ue8m0_fp32_literal_false :
    ¬ ∀ x ∈ e2m1Values, ∀ s ∈ ue8m0Values, x * s ∈ fp32Values := by
  sorry

/-- 3c, exact range: every E2M1 value times the UE8M0 scale with code c is an fp32 value iff c ≤ 252
(scale exponent ≤ 125); the edge is 6 × 2^126, which overflows fp32. -/
theorem e2m1_ue8m0_fp32_iff (c : ℕ) (hc : c < 255) :
    (∀ x ∈ e2m1Values, x * (2 : ℚ) ^ ((c : ℤ) - 127) ∈ fp32Values) ↔ c ≤ 252 := by
  sorry

end NumericsProof
