/-
NUMERICS-PROOF: the proofs. Definitions and theorem statements are copied verbatim from
Statements.lean (lake comparator checks they match); the helper lemmas are new.
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

set_option linter.deprecated false
set_option linter.unusedSectionVars false
set_option linter.style.longLine false

/-! ## Proofs, Part 1 -/

section TopKThms

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

theorem exact_no_false_fail (s b : ι → ℝ) (k : ℕ) (S : Finset ι) :
    Reachable s b k S → ExactAccept s b k S := by
  rintro ⟨g, hg, hcard, htop⟩
  refine ⟨hcard, fun t ht u hu => ?_⟩
  have h1 := htop t ht u hu
  have h2 := abs_le.mp (hg t)
  have h3 := abs_le.mp (hg u)
  linarith [h2.1, h2.2, h3.1, h3.2]

theorem exact_no_false_pass (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (k : ℕ) (S : Finset ι) :
    ExactAccept s b k S → Reachable s b k S := by
  rintro ⟨hcard, h⟩
  refine ⟨fun t => if t ∈ S then s t + b t else s t - b t, ?_, hcard, ?_⟩
  · intro t
    by_cases ht : t ∈ S
    · simp [ht, abs_of_nonneg (hb t)]
    · simp [ht, abs_of_nonneg (hb t)]
  · intro t ht u hu
    simp only [ht, hu, if_true, if_false]
    exact h t ht u hu

theorem exact_accept_iff_inf_sup (s b : ι → ℝ) (S : Finset ι) (hS : S.Nonempty) (hT : Sᶜ.Nonempty) :
    (∀ t ∈ S, ∀ u ∉ S, s u - b u ≤ s t + b t) ↔
      Sᶜ.sup' hT (fun u => s u - b u) ≤ S.inf' hS (fun t => s t + b t) := by
  rw [Finset.sup'_le_iff]
  constructor
  · intro h u hu
    rw [Finset.le_inf'_iff]
    intro t ht
    exact h t ht u (Finset.mem_compl.mp hu)
  · intro h t ht u hu
    have := h u (Finset.mem_compl.mpr hu)
    rw [Finset.le_inf'_iff] at this
    exact this t ht

/-- a threshold θ with fewer than k values above it and at least k at or above it -/
lemma exists_threshold (g : ι → ℝ) (k : ℕ) (hk0 : 0 < k) (hk : k ≤ Fintype.card ι) :
    ∃ θ, (univ.filter fun u => θ < g u).card < k ∧ k ≤ (univ.filter fun u => θ ≤ g u).card := by
  classical
  have hne : (univ : Finset ι).Nonempty := by
    rw [Finset.univ_nonempty_iff, ← Fintype.card_pos_iff]; omega
  obtain ⟨u0, -, hu0⟩ := Finset.exists_min_image univ g hne
  set T := (univ.image g).filter (fun θ => k ≤ (univ.filter fun u => θ ≤ g u).card) with hTdef
  have hT : T.Nonempty := by
    refine ⟨g u0, ?_⟩
    rw [hTdef, Finset.mem_filter]
    refine ⟨Finset.mem_image_of_mem g (Finset.mem_univ _), ?_⟩
    have : (univ.filter fun u => g u0 ≤ g u) = univ := by
      ext u; simp [hu0 u (Finset.mem_univ u)]
    rw [this, Finset.card_univ]; exact hk
  refine ⟨T.max' hT, ?_, ?_⟩
  · by_contra hlt
    push_neg at hlt
    set A := univ.filter fun u => T.max' hT < g u with hA
    have hAne : A.Nonempty := by
      rw [← Finset.card_pos]; omega
    obtain ⟨u1, hu1A, hu1⟩ := Finset.exists_min_image A g hAne
    have hmem : g u1 ∈ T := by
      rw [hTdef, Finset.mem_filter]
      refine ⟨Finset.mem_image_of_mem g (Finset.mem_univ _), le_trans hlt (Finset.card_le_card ?_)⟩
      intro u hu
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ _, hu1 u hu⟩
    have h1 := T.le_max' (g u1) hmem
    rw [hA, Finset.mem_filter] at hu1A
    linarith [hu1A.2]
  · have := T.max'_mem hT
    exact (Finset.mem_filter.mp this).2

lemma exists_topk_mem (g : ι → ℝ) (k : ℕ) (hk : k ≤ Fintype.card ι) (t : ι)
    (ht : (univ.filter fun u => g t < g u).card < k) : ∃ S, IsTopK g k S ∧ t ∈ S := by
  have hk0 : 0 < k := by omega
  obtain ⟨θ, hP, hQ⟩ := exists_threshold g k hk0 hk
  rcases lt_or_ge (g t) θ with hlt | hle
  · exfalso
    have : (univ.filter fun u => θ ≤ g u) ⊆ (univ.filter fun u => g t < g u) := by
      intro u hu
      rw [Finset.mem_filter] at hu ⊢
      exact ⟨hu.1, lt_of_lt_of_le hlt hu.2⟩
    have := Finset.card_le_card this
    omega
  · have hsub : insert t (univ.filter fun u => θ < g u) ⊆ (univ.filter fun u => θ ≤ g u) := by
      intro u hu
      rw [Finset.mem_insert] at hu
      rw [Finset.mem_filter]
      rcases hu with rfl | hu
      · exact ⟨Finset.mem_univ _, hle⟩
      · rw [Finset.mem_filter] at hu; exact ⟨hu.1, le_of_lt hu.2⟩
    have hcard : (insert t (univ.filter fun u => θ < g u)).card ≤ k := by
      have := Finset.card_insert_le t (univ.filter fun u => θ < g u)
      omega
    obtain ⟨S, hPS, hSQ, hSc⟩ := Finset.exists_subsuperset_card_eq hsub hcard hQ
    refine ⟨S, ⟨hSc, fun x hx y hy => ?_⟩, hPS (Finset.mem_insert_self _ _)⟩
    have hx' := hSQ hx
    rw [Finset.mem_filter] at hx'
    have hy' : y ∉ univ.filter fun u => θ < g u := fun h => hy (hPS (Finset.mem_insert_of_mem h))
    rw [Finset.mem_filter] at hy'
    push_neg at hy'
    linarith [hx'.2, hy' (Finset.mem_univ _)]

lemma exists_topk_not_mem (g : ι → ℝ) (k : ℕ) (hk0 : 0 < k) (t : ι)
    (ht : k ≤ (univ.filter fun u => u ≠ t ∧ g t ≤ g u).card) : ∃ S, IsTopK g k S ∧ t ∉ S := by
  have hk : k ≤ Fintype.card ι := le_trans ht (by rw [← Finset.card_univ]; exact Finset.card_le_card (Finset.filter_subset _ _))
  obtain ⟨θ, hP, hQ⟩ := exists_threshold g k hk0 hk
  set P := univ.filter fun u => θ < g u with hPdef
  set Q := univ.filter fun u => θ ≤ g u with hQdef
  rcases lt_or_ge θ (g t) with hlt | hle
  · exfalso
    have hsub : insert t (univ.filter fun u => u ≠ t ∧ g t ≤ g u) ⊆ P := by
      intro u hu
      rw [Finset.mem_insert] at hu
      rw [hPdef, Finset.mem_filter]
      rcases hu with rfl | hu
      · exact ⟨Finset.mem_univ _, hlt⟩
      · rw [Finset.mem_filter] at hu; exact ⟨hu.1, lt_of_lt_of_le hlt hu.2.2⟩
    have hnot : t ∉ univ.filter fun u => u ≠ t ∧ g t ≤ g u := by simp
    have := Finset.card_le_card hsub
    rw [Finset.card_insert_of_notMem hnot] at this
    omega
  · have htP : t ∉ P := by rw [hPdef, Finset.mem_filter]; push_neg; intro _; exact hle
    have hsub : P ⊆ Q.erase t := by
      intro u hu
      rw [Finset.mem_erase]
      refine ⟨fun h => htP (h ▸ hu), ?_⟩
      rw [hPdef, Finset.mem_filter] at hu
      rw [hQdef, Finset.mem_filter]
      exact ⟨hu.1, le_of_lt hu.2⟩
    have hQe : k ≤ (Q.erase t).card := by
      rcases lt_or_eq_of_le hle with hlt2 | heq
      · have htQ : t ∉ Q := by rw [hQdef, Finset.mem_filter]; push_neg; intro _; exact hlt2
        rw [Finset.erase_eq_of_notMem htQ]; exact hQ
      · refine le_trans ht (Finset.card_le_card ?_)
        intro u hu
        rw [Finset.mem_filter] at hu
        rw [Finset.mem_erase, hQdef, Finset.mem_filter]
        exact ⟨hu.2.1, Finset.mem_univ _, heq ▸ hu.2.2⟩
    obtain ⟨S, hPS, hSQ, hSc⟩ := Finset.exists_subsuperset_card_eq hsub (le_of_lt hP) hQe
    refine ⟨S, ⟨hSc, fun x hx y hy => ?_⟩, fun h => (Finset.mem_erase.mp (hSQ h)).1 rfl⟩
    have hx' := (Finset.mem_erase.mp (hSQ hx)).2
    rw [hQdef, Finset.mem_filter] at hx'
    have hy' : y ∉ P := fun h => hy (hPS h)
    rw [hPdef, Finset.mem_filter] at hy'
    push_neg at hy'
    linarith [hx'.2, hy' (Finset.mem_univ _)]

theorem classify_may (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (k : ℕ) (hk : k ≤ Fintype.card ι) (t : ι) :
    (∃ S, Reachable s b k S ∧ t ∈ S) ↔
      (univ.filter fun u => s t + b t < s u - b u).card < k := by
  constructor
  · rintro ⟨S, hR, htS⟩
    obtain ⟨hcard, hacc⟩ := exact_no_false_fail s b k S hR
    have hsub : (univ.filter fun u => s t + b t < s u - b u) ⊆ S.erase t := by
      intro u hu
      rw [Finset.mem_filter] at hu
      rw [Finset.mem_erase]
      constructor
      · rintro rfl; linarith [hb u, hu.2]
      · by_contra hS
        linarith [hacc t htS u hS, hu.2]
    have h1 := Finset.card_le_card hsub
    rw [Finset.card_erase_of_mem htS, hcard] at h1
    have : 0 < k := by rw [← hcard]; exact Finset.card_pos.mpr ⟨t, htS⟩
    omega
  · intro h
    let g : ι → ℝ := fun u => if u = t then s u + b u else s u - b u
    have hg : InBand s b g := by
      intro u
      by_cases hu : u = t
      · simp [g, hu, abs_of_nonneg (hb t)]
      · simp [g, hu, abs_of_nonneg (hb u)]
    have heq : (univ.filter fun u => g t < g u) = (univ.filter fun u => s t + b t < s u - b u) := by
      ext u
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, g, if_true]
      by_cases hu : u = t
      · subst hu; simp only [if_true]; constructor <;> intro h <;> linarith [hb u]
      · simp only [hu, if_false]
    obtain ⟨S, hS, htS⟩ := exists_topk_mem g k hk t (by rw [heq]; exact h)
    exact ⟨S, ⟨g, hg, hS⟩, htS⟩

theorem classify_must (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (k : ℕ) (t : ι) :
    (∀ S, Reachable s b k S → t ∈ S) ↔
      (univ.filter fun u => u ≠ t ∧ s t - b t ≤ s u + b u).card < k := by
  constructor
  · intro h
    by_contra hge
    push_neg at hge
    rcases Nat.eq_zero_or_pos k with hk0 | hk0
    · subst hk0
      have hR : Reachable s b 0 ∅ := ⟨s, fun u => by simp [hb u], by simp, by simp⟩
      exact Finset.notMem_empty t (h ∅ hR)
    · let g : ι → ℝ := fun u => if u = t then s u - b u else s u + b u
      have hg : InBand s b g := by
        intro u
        by_cases hu : u = t
        · simp [g, hu, abs_of_nonneg (hb t)]
        · simp [g, hu, abs_of_nonneg (hb u)]
      have heq : (univ.filter fun u => u ≠ t ∧ g t ≤ g u) =
          (univ.filter fun u => u ≠ t ∧ s t - b t ≤ s u + b u) := by
        ext u
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, g, if_true]
        by_cases hu : u = t
        · simp [hu]
        · simp only [hu, if_false, ne_eq, not_false_eq_true, true_and]
      obtain ⟨S, hS, htS⟩ := exists_topk_not_mem g k hk0 t (by rw [heq]; exact hge)
      exact htS (h S ⟨g, hg, hS⟩)
  · intro h S hR
    by_contra htS
    obtain ⟨hcard, hacc⟩ := exact_no_false_fail s b k S hR
    have hsub : S ⊆ (univ.filter fun u => u ≠ t ∧ s t - b t ≤ s u + b u) := by
      intro x hx
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ _, fun hxt => htS (hxt ▸ hx), hacc x hx t htS⟩
    have := Finset.card_le_card hsub
    omega

theorem argmax_exact (s b : ι → ℝ) (hb : ∀ t, 0 ≤ b t) (a : ι) :
    (∀ u, u ≠ a → s u - b u ≤ s a + b a) ↔ ∃ g, InBand s b g ∧ ∀ u, g u ≤ g a := by
  constructor
  · intro h
    obtain ⟨g, hg, -, htop⟩ := exact_no_false_pass s b hb 1 {a}
      ⟨Finset.card_singleton a, fun t ht u hu => by
        rw [Finset.mem_singleton] at ht hu; subst ht; exact h u hu⟩
    refine ⟨g, hg, fun u => ?_⟩
    by_cases hu : u = a
    · rw [hu]
    · exact htop a (Finset.mem_singleton_self a) u (by rwa [Finset.mem_singleton])
  · rintro ⟨g, hg, hmax⟩ u hu
    have hacc := (exact_no_false_fail s b 1 {a} ⟨g, hg, Finset.card_singleton a, fun t ht v _ => by
      rw [Finset.mem_singleton] at ht; subst ht; exact hmax v⟩).2
    exact hacc a (Finset.mem_singleton_self a) u (by rwa [Finset.mem_singleton])

/-! ## Proofs, Part 2 -/

theorem old_rule_sound_equal_bands (s : ι → ℝ) (β : ℝ) (k : ℕ) (p : ι) (S : Finset ι)
    (hp : IsKthPos s k p) :
    Reachable s (fun _ => β) k S → OldAccept s (fun _ => β) k p S := by
  intro hR
  obtain ⟨hcard, hacc⟩ := exact_no_false_fail s (fun _ => β) k S hR
  obtain ⟨hP, hQ⟩ := hp
  refine ⟨hcard, ?_, ?_⟩
  · intro e he
    by_contra heS
    simp only [OldMust] at he
    have hsub : S ⊆ univ.filter fun u => s p < s u := by
      intro x hx
      rw [Finset.mem_filter]
      have := hacc x hx e heS
      exact ⟨Finset.mem_univ _, by linarith⟩
    have := Finset.card_le_card hsub
    omega
  · intro e heS
    by_cases hm : OldMust s (fun _ => β) p e
    · exact Or.inl hm
    · right
      refine ⟨?_, hm⟩
      simp only [OldMust, not_lt] at hm
      obtain ⟨g, hg, -⟩ := hR
      have hβ : 0 ≤ β := le_trans (abs_nonneg _) (hg e)
      rw [abs_le]
      refine ⟨?_, by linarith⟩
      by_contra hlt
      push_neg at hlt
      have hsub : insert e (univ.filter fun u => s p ≤ s u) ⊆ S := by
        intro x hx
        rw [Finset.mem_insert] at hx
        rcases hx with rfl | hx
        · exact heS
        · rw [Finset.mem_filter] at hx
          by_contra hxS
          have := hacc e heS x hxS
          linarith [hx.2]
      have hnot : e ∉ univ.filter fun u => s p ≤ s u := by
        rw [Finset.mem_filter]; push_neg; intro _; linarith
      have := Finset.card_le_card hsub
      rw [Finset.card_insert_of_notMem hnot] at this
      omega

end TopKThms

theorem old_rule_false_fail_ce1 :
    IsKthPos ce1_s 2 1 ∧ InBand ce1_s ce1_b ce1_g ∧ IsTopK ce1_g 2 {2, 3} ∧
      ¬ OldAccept ce1_s ce1_b 2 1 {2, 3} := by
  have hA : (univ.filter fun u : Fin 4 => ce1_s 1 < ce1_s u) = {0} := by
    ext u; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    fin_cases u <;> norm_num [ce1_s]
  have hB : (univ.filter fun u : Fin 4 => ce1_s 1 ≤ ce1_s u) = {0, 1} := by
    ext u; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    fin_cases u <;> norm_num [ce1_s]
  refine ⟨⟨by rw [hA]; decide, by rw [hB]; decide⟩, ?_, ⟨by decide, ?_⟩, ?_⟩
  · intro t; fin_cases t <;> simp [ce1_s, ce1_b, ce1_g, abs_le] <;> norm_num
  · intro t ht u hu
    fin_cases t <;> fin_cases u <;> simp_all [ce1_g] <;> norm_num
  · rintro ⟨-, hmust, -⟩
    have : OldMust ce1_s ce1_b 1 0 := by simp [OldMust, ce1_s, ce1_b]; norm_num
    have := hmust 0 this
    simp at this

theorem old_rule_false_fail_ce2 :
    IsKthPos ce2_s 2 1 ∧ InBand ce2_s ce2_b ce2_g ∧ IsTopK ce2_g 2 {1, 2} ∧
      ¬ OldAccept ce2_s ce2_b 2 1 {1, 2} := by
  have hA : (univ.filter fun u : Fin 3 => ce2_s 1 < ce2_s u) = {0} := by
    ext u; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    fin_cases u <;> norm_num [ce2_s]
  have hB : (univ.filter fun u : Fin 3 => ce2_s 1 ≤ ce2_s u) = {0, 1} := by
    ext u; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    fin_cases u <;> norm_num [ce2_s]
  refine ⟨⟨by rw [hA]; decide, by rw [hB]; decide⟩, ?_, ⟨by decide, ?_⟩, ?_⟩
  · intro t; fin_cases t <;> simp [ce2_s, ce2_b, ce2_g, abs_le] <;> norm_num
  · intro t ht u hu
    fin_cases t <;> fin_cases u <;> simp_all [ce2_g] <;> norm_num
  · rintro ⟨-, -, hin⟩
    have := hin 2 (by decide)
    simp [OldMust, OldFree, ce2_s, ce2_b, abs_le] at this
    norm_num at this

theorem old_rule_unsound :
    ¬ ∀ (s b : Fin 4 → ℝ) (k : ℕ) (p : Fin 4) (S : Finset (Fin 4)),
        (∀ t, 0 ≤ b t) → IsKthPos s k p → Reachable s b k S → OldAccept s b k p S := by
  intro h
  obtain ⟨hp, hg, htop, hno⟩ := old_rule_false_fail_ce1
  refine hno (h ce1_s ce1_b 2 1 {2, 3} ?_ hp ⟨ce1_g, hg, htop⟩)
  intro t; fin_cases t <;> simp [ce1_b] <;> norm_num

theorem old_rule_false_pass_fp1 :
    IsKthPos fp1_s 2 1 ∧ OldAccept fp1_s (fun _ => 1) 2 1 {1, 2} ∧
      ¬ Reachable fp1_s (fun _ => 1) 2 {1, 2} := by
  have hA : (univ.filter fun u : Fin 3 => fp1_s 1 < fp1_s u) = {0} := by
    ext u; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    fin_cases u <;> norm_num [fp1_s]
  have hB : (univ.filter fun u : Fin 3 => fp1_s 1 ≤ fp1_s u) = {0, 1} := by
    ext u; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Finset.mem_singleton]
    fin_cases u <;> norm_num [fp1_s]
  refine ⟨⟨by rw [hA]; decide, by rw [hB]; decide⟩, ⟨by decide, ?_, ?_⟩, ?_⟩
  · intro e he
    fin_cases e <;> simp [OldMust, fp1_s] at he <;> norm_num at he
  · intro e he
    fin_cases e <;> simp at he <;> right <;> simp [OldFree, OldMust, fp1_s, abs_le] <;> norm_num
  · intro hR
    have := (exact_no_false_fail _ _ _ _ hR).2 2 (by decide) 0 (by decide)
    simp [fp1_s] at this
    norm_num at this

/-! ## Proofs, Part 3: formats -/

lemma two_zpow_pos (n : ℤ) : (0 : ℚ) < 2 ^ n := zpow_pos (by norm_num) n

/-- a bf16 code with sign bit σ, exponent field e (1..254) and mantissa m decodes to ±1.m × 2^(e-127) -/
lemma bf16_normal (σ e m : ℕ) (hσ : σ < 2) (he1 : 1 ≤ e) (he : e ≤ 254) (hm : m < 128) :
    ieeeDecode 8 7 (σ * 32768 + e * 128 + m) =
      some ((if σ = 1 then -1 else 1) * (1 + (m : ℚ) / 128) * (2 : ℚ) ^ ((e : ℤ) - 127)) := by
  have h1 : (σ * 32768 + e * 128 + m) / 2 ^ (8 + 7) % 2 = σ := by norm_num; omega
  have h2 : (σ * 32768 + e * 128 + m) / 2 ^ 7 % 2 ^ 8 = e := by norm_num; omega
  have h3 : (σ * 32768 + e * 128 + m) % 2 ^ 7 = m := by norm_num; omega
  have h4 : e ≠ 2 ^ 8 - 1 := by norm_num; omega
  have h5 : e ≠ 0 := by omega
  simp only [ieeeDecode, h1, h2, h3, h4, h5, if_false]
  norm_num

/-- a bf16 code with exponent field 0 decodes to ±0.m × 2^-126 = ±m × 2^-133 -/
lemma bf16_sub (σ m : ℕ) (hσ : σ < 2) (hm : m < 128) :
    ieeeDecode 8 7 (σ * 32768 + m) = some ((if σ = 1 then -1 else 1) * ((m : ℚ) * 2 ^ (-133 : ℤ))) := by
  have h1 : (σ * 32768 + m) / 2 ^ (8 + 7) % 2 = σ := by norm_num; omega
  have h2 : (σ * 32768 + m) / 2 ^ 7 % 2 ^ 8 = 0 := by norm_num; omega
  have h3 : (σ * 32768 + m) % 2 ^ 7 = m := by norm_num; omega
  simp only [ieeeDecode, h1, h2, h3]
  norm_num
  ring_nf

lemma sgn_cases (σ : ℕ) (hσ : σ < 2) (q : ℚ) :
    (σ = 0 → (if σ = 1 then (-1 : ℚ) else 1) * q = q) ∧ (σ = 1 → (if σ = 1 then (-1 : ℚ) else 1) * q = -q) := by
  constructor <;> intro h <;> subst h <;> simp

/-- every ±N × 2^(n-133) with N < 256 below the bf16 maximum is a bf16 value -/
lemma bf16_mem_nat (n : ℕ) : ∀ N : ℕ, N < 256 → (N : ℚ) * 2 ^ ((n : ℤ) - 133) ≤ 255 * 2 ^ (120 : ℤ) →
    (N : ℚ) * 2 ^ ((n : ℤ) - 133) ∈ bf16Values ∧ -((N : ℚ) * 2 ^ ((n : ℤ) - 133)) ∈ bf16Values := by
  induction n with
  | zero =>
    intro N hN hmax
    by_cases h128 : 128 ≤ N
    · have hval : ∀ σ, σ < 2 → ieeeDecode 8 7 (σ * 32768 + 1 * 128 + (N - 128)) =
          some ((if σ = 1 then -1 else 1) * ((N : ℚ) * 2 ^ (((0 : ℕ) : ℤ) - 133))) := by
        intro σ hσ
        rw [bf16_normal σ 1 (N - 128) hσ le_rfl (by norm_num) (by omega)]
        congr 1
        rw [Nat.cast_sub h128]
        norm_num
        ring_nf
      refine ⟨⟨0 * 32768 + 1 * 128 + (N - 128), by omega, ?_⟩, ⟨1 * 32768 + 1 * 128 + (N - 128), by omega, ?_⟩⟩
      · rw [hval 0 (by norm_num)]; simp
      · rw [hval 1 (by norm_num)]; simp
    · refine ⟨⟨0 * 32768 + N, by omega, ?_⟩, ⟨1 * 32768 + N, by omega, ?_⟩⟩
      · rw [bf16_sub 0 N (by norm_num) (by omega)]; simp
      · rw [bf16_sub 1 N (by norm_num) (by omega)]; simp
  | succ n ih =>
    intro N hN hmax
    by_cases h128 : 128 ≤ N
    · have hn : n + 2 ≤ 254 := by
        by_contra hcon
        have h1 : (2 : ℚ) ^ (121 : ℤ) ≤ 2 ^ (((n + 1 : ℕ) : ℤ) - 133) :=
          zpow_le_zpow_right₀ (by norm_num) (by push_cast; omega)
        have h2 : (128 : ℚ) ≤ N := by exact_mod_cast h128
        have h3 : (128 : ℚ) * 2 ^ (121 : ℤ) ≤ N * 2 ^ (((n + 1 : ℕ) : ℤ) - 133) :=
          mul_le_mul h2 h1 (le_of_lt (two_zpow_pos _)) (by positivity)
        have h4 : (255 : ℚ) * 2 ^ (120 : ℤ) < 128 * 2 ^ (121 : ℤ) := by norm_num
        linarith
      have hval : ∀ σ, σ < 2 → ieeeDecode 8 7 (σ * 32768 + (n + 2) * 128 + (N - 128)) =
          some ((if σ = 1 then -1 else 1) * ((N : ℚ) * 2 ^ (((n + 1 : ℕ) : ℤ) - 133))) := by
        intro σ hσ
        rw [bf16_normal σ (n + 2) (N - 128) hσ (by omega) hn (by omega)]
        congr 1
        rw [Nat.cast_sub h128,
          show ((n + 2 : ℕ) : ℤ) - 127 = (((n + 1 : ℕ) : ℤ) - 133) + 7 by push_cast; ring,
          zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
        norm_num
        ring_nf
      refine ⟨⟨0 * 32768 + (n + 2) * 128 + (N - 128), by omega, ?_⟩,
        ⟨1 * 32768 + (n + 2) * 128 + (N - 128), by omega, ?_⟩⟩
      · rw [hval 0 (by norm_num)]; simp
      · rw [hval 1 (by norm_num)]; simp
    · have heq : (N : ℚ) * 2 ^ (((n + 1 : ℕ) : ℤ) - 133) = ((2 * N : ℕ) : ℚ) * 2 ^ ((n : ℤ) - 133) := by
        rw [show (((n + 1 : ℕ) : ℤ) - 133) = ((n : ℤ) - 133) + 1 by push_cast; ring,
          zpow_add_one₀ (by norm_num : (2 : ℚ) ≠ 0)]
        push_cast; ring
      rw [heq] at hmax ⊢
      exact ih (2 * N) (by omega) hmax

lemma bf16_mem (N : ℕ) (F : ℤ) (hN : N < 256) (hF : -133 ≤ F) (hmax : (N : ℚ) * 2 ^ F ≤ 255 * 2 ^ (120 : ℤ)) :
    (N : ℚ) * 2 ^ F ∈ bf16Values ∧ -((N : ℚ) * 2 ^ F) ∈ bf16Values := by
  obtain ⟨n, hn⟩ : ∃ n : ℕ, F = (n : ℤ) - 133 := ⟨(F + 133).toNat, by omega⟩
  subst hn
  exact bf16_mem_nat n N hN hmax

/-- unfolding a decoded bf16 value: sign, exponent field e < 256 (≠ 255), mantissa m < 128 -/
lemma bf16_cases (q : ℚ) (hq : q ∈ bf16Values) : ∃ (sg : ℚ) (e m : ℕ), |sg| = 1 ∧ e < 255 ∧ m < 128 ∧
    ((e = 0 ∧ q = sg * ((m : ℚ) / 128) * 2 ^ (-126 : ℤ)) ∨
     (1 ≤ e ∧ q = sg * (1 + (m : ℚ) / 128) * 2 ^ ((e : ℤ) - 127))) := by
  obtain ⟨c, -, h⟩ := hq
  have hm : c % 2 ^ 7 < 128 := Nat.mod_lt _ (by norm_num)
  have he : c / 2 ^ 7 % 2 ^ 8 < 256 := Nat.mod_lt _ (by norm_num)
  simp only [ieeeDecode] at h
  generalize c / 2 ^ 7 % 2 ^ 8 = e at h he
  generalize c % 2 ^ 7 = m at h hm
  generalize hs : (if c / 2 ^ (8 + 7) % 2 = 1 then (-1 : ℚ) else 1) = sg at h
  have hsg : |sg| = 1 := by rw [← hs]; split_ifs <;> simp
  split_ifs at h with h1 h2
  · have := Option.some.inj h
    refine ⟨sg, e, m, hsg, by omega, hm, Or.inl ⟨h2, ?_⟩⟩
    rw [← this]; norm_num
  · have := Option.some.inj h
    refine ⟨sg, e, m, hsg, by omega, hm, Or.inr ⟨by omega, ?_⟩⟩
    rw [← this]; norm_num

lemma bf16_abs_le (q : ℚ) (hq : q ∈ bf16Values) : |q| ≤ 255 * 2 ^ (120 : ℤ) := by
  obtain ⟨sg, e, m, hsg, he, hm, h | h⟩ := bf16_cases q hq
  · rw [h.2, abs_mul, abs_mul, hsg, abs_of_nonneg (by positivity), abs_of_nonneg (le_of_lt (two_zpow_pos _))]
    have : (m : ℚ) ≤ 127 := by exact_mod_cast (by omega : m ≤ 127)
    have h2 : (2 : ℚ) ^ (-126 : ℤ) ≤ 1 := by norm_num
    have h3 : (1 : ℚ) ≤ 255 * 2 ^ (120 : ℤ) := by norm_num
    have h4 : (m : ℚ) / 128 ≤ 1 := by rw [div_le_one (by norm_num)]; linarith
    calc 1 * ((m : ℚ) / 128) * 2 ^ (-126 : ℤ) ≤ 1 * 1 * 1 := by
          apply mul_le_mul _ h2 (le_of_lt (two_zpow_pos _)) (by norm_num)
          linarith
      _ ≤ _ := by linarith
  · rw [h.2, abs_mul, abs_mul, hsg, abs_of_nonneg (by positivity), abs_of_nonneg (le_of_lt (two_zpow_pos _))]
    have : (m : ℚ) ≤ 127 := by exact_mod_cast (by omega : m ≤ 127)
    have h2 : (2 : ℚ) ^ ((e : ℤ) - 127) ≤ 2 ^ (127 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h3 : 1 + (m : ℚ) / 128 ≤ 255 / 128 := by linarith
    calc 1 * (1 + (m : ℚ) / 128) * 2 ^ ((e : ℤ) - 127) ≤ 1 * (255 / 128) * 2 ^ (127 : ℤ) := by
          apply mul_le_mul (by linarith) h2 (le_of_lt (two_zpow_pos _)) (by norm_num)
      _ = 255 * 2 ^ (120 : ℤ) := by norm_num

lemma bf16_abs_ge (q : ℚ) (hq : q ∈ bf16Values) (hq0 : q ≠ 0) : 2 ^ (-133 : ℤ) ≤ |q| := by
  obtain ⟨sg, e, m, hsg, he, hm, h | h⟩ := bf16_cases q hq
  · rw [h.2, abs_mul, abs_mul, hsg, abs_of_nonneg (by positivity), abs_of_nonneg (le_of_lt (two_zpow_pos _))]
    have hm0 : m ≠ 0 := by
      rintro rfl; apply hq0; rw [h.2]; simp
    have : (1 : ℚ) ≤ m := by exact_mod_cast (by omega : 1 ≤ m)
    have hz : (2 : ℚ) ^ (-133 : ℤ) = 1 / 128 * 2 ^ (-126 : ℤ) := by norm_num
    rw [hz]
    apply mul_le_mul _ le_rfl (le_of_lt (two_zpow_pos _)) (by positivity)
    rw [one_mul]
    exact div_le_div_of_nonneg_right this (by norm_num)
  · rw [h.2, abs_mul, abs_mul, hsg, abs_of_nonneg (by positivity), abs_of_nonneg (le_of_lt (two_zpow_pos _))]
    have h2 : (2 : ℚ) ^ (-133 : ℤ) ≤ 2 ^ ((e : ℤ) - 127) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h3 : (1 : ℚ) ≤ 1 + (m : ℚ) / 128 := le_add_of_nonneg_right (by positivity)
    calc (2 : ℚ) ^ (-133 : ℤ) = 1 * 1 * 2 ^ (-133 : ℤ) := by ring
      _ ≤ 1 * (1 + (m : ℚ) / 128) * 2 ^ ((e : ℤ) - 127) := by
          apply mul_le_mul (by linarith) h2 (le_of_lt (two_zpow_pos _)) (by positivity)

lemma fp32_normal (σ e m : ℕ) (hσ : σ < 2) (he1 : 1 ≤ e) (he : e ≤ 254) (hm : m < 8388608) :
    ieeeDecode 8 23 (σ * 2147483648 + e * 8388608 + m) =
      some ((if σ = 1 then -1 else 1) * (1 + (m : ℚ) / 8388608) * (2 : ℚ) ^ ((e : ℤ) - 127)) := by
  have h1 : (σ * 2147483648 + e * 8388608 + m) / 2 ^ (8 + 23) % 2 = σ := by norm_num; omega
  have h2 : (σ * 2147483648 + e * 8388608 + m) / 2 ^ 23 % 2 ^ 8 = e := by norm_num; omega
  have h3 : (σ * 2147483648 + e * 8388608 + m) % 2 ^ 23 = m := by norm_num; omega
  have h4 : e ≠ 2 ^ 8 - 1 := by norm_num; omega
  have h5 : e ≠ 0 := by omega
  simp only [ieeeDecode, h1, h2, h3, h4, h5, if_false]
  norm_num

lemma fp32_sub (σ m : ℕ) (hσ : σ < 2) (hm : m < 8388608) :
    ieeeDecode 8 23 (σ * 2147483648 + m) = some ((if σ = 1 then -1 else 1) * ((m : ℚ) * 2 ^ (-149 : ℤ))) := by
  have h1 : (σ * 2147483648 + m) / 2 ^ (8 + 23) % 2 = σ := by norm_num; omega
  have h2 : (σ * 2147483648 + m) / 2 ^ 23 % 2 ^ 8 = 0 := by norm_num; omega
  have h3 : (σ * 2147483648 + m) % 2 ^ 23 = m := by norm_num; omega
  simp only [ieeeDecode, h1, h2, h3]
  norm_num
  ring_nf

lemma fp32_mem_nat (n : ℕ) : ∀ N : ℕ, N < 16777216 →
    (N : ℚ) * 2 ^ ((n : ℤ) - 149) ≤ 16777215 * 2 ^ (104 : ℤ) →
    (N : ℚ) * 2 ^ ((n : ℤ) - 149) ∈ fp32Values ∧ -((N : ℚ) * 2 ^ ((n : ℤ) - 149)) ∈ fp32Values := by
  induction n with
  | zero =>
    intro N hN hmax
    by_cases h128 : 8388608 ≤ N
    · have hval : ∀ σ, σ < 2 → ieeeDecode 8 23 (σ * 2147483648 + 1 * 8388608 + (N - 8388608)) =
          some ((if σ = 1 then -1 else 1) * ((N : ℚ) * 2 ^ (((0 : ℕ) : ℤ) - 149))) := by
        intro σ hσ
        rw [fp32_normal σ 1 (N - 8388608) hσ le_rfl (by norm_num) (by omega)]
        congr 1
        rw [Nat.cast_sub h128]
        norm_num
        ring_nf
      refine ⟨⟨0 * 2147483648 + 1 * 8388608 + (N - 8388608), by omega, ?_⟩,
        ⟨1 * 2147483648 + 1 * 8388608 + (N - 8388608), by omega, ?_⟩⟩
      · rw [hval 0 (by norm_num)]; simp
      · rw [hval 1 (by norm_num)]; simp
    · refine ⟨⟨0 * 2147483648 + N, by omega, ?_⟩, ⟨1 * 2147483648 + N, by omega, ?_⟩⟩
      · rw [fp32_sub 0 N (by norm_num) (by omega)]; simp
      · rw [fp32_sub 1 N (by norm_num) (by omega)]; simp
  | succ n ih =>
    intro N hN hmax
    by_cases h128 : 8388608 ≤ N
    · have hn : n + 2 ≤ 254 := by
        by_contra hcon
        have h1 : (2 : ℚ) ^ (105 : ℤ) ≤ 2 ^ (((n + 1 : ℕ) : ℤ) - 149) :=
          zpow_le_zpow_right₀ (by norm_num) (by push_cast; omega)
        have h2 : (8388608 : ℚ) ≤ N := by exact_mod_cast h128
        have h3 : (8388608 : ℚ) * 2 ^ (105 : ℤ) ≤ N * 2 ^ (((n + 1 : ℕ) : ℤ) - 149) :=
          mul_le_mul h2 h1 (le_of_lt (two_zpow_pos _)) (by positivity)
        have h4 : (16777215 : ℚ) * 2 ^ (104 : ℤ) < 8388608 * 2 ^ (105 : ℤ) := by norm_num
        linarith
      have hval : ∀ σ, σ < 2 → ieeeDecode 8 23 (σ * 2147483648 + (n + 2) * 8388608 + (N - 8388608)) =
          some ((if σ = 1 then -1 else 1) * ((N : ℚ) * 2 ^ (((n + 1 : ℕ) : ℤ) - 149))) := by
        intro σ hσ
        rw [fp32_normal σ (n + 2) (N - 8388608) hσ (by omega) hn (by omega)]
        congr 1
        rw [Nat.cast_sub h128,
          show ((n + 2 : ℕ) : ℤ) - 127 = (((n + 1 : ℕ) : ℤ) - 149) + 23 by push_cast; ring,
          zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
        norm_num
        ring_nf
      refine ⟨⟨0 * 2147483648 + (n + 2) * 8388608 + (N - 8388608), by omega, ?_⟩,
        ⟨1 * 2147483648 + (n + 2) * 8388608 + (N - 8388608), by omega, ?_⟩⟩
      · rw [hval 0 (by norm_num)]; simp
      · rw [hval 1 (by norm_num)]; simp
    · have heq : (N : ℚ) * 2 ^ (((n + 1 : ℕ) : ℤ) - 149) = ((2 * N : ℕ) : ℚ) * 2 ^ ((n : ℤ) - 149) := by
        rw [show (((n + 1 : ℕ) : ℤ) - 149) = ((n : ℤ) - 149) + 1 by push_cast; ring,
          zpow_add_one₀ (by norm_num : (2 : ℚ) ≠ 0)]
        push_cast; ring
      rw [heq] at hmax ⊢
      exact ih (2 * N) (by omega) hmax

lemma fp32_mem (N : ℕ) (F : ℤ) (hN : N < 16777216) (hF : -149 ≤ F)
    (hmax : (N : ℚ) * 2 ^ F ≤ 16777215 * 2 ^ (104 : ℤ)) :
    (N : ℚ) * 2 ^ F ∈ fp32Values ∧ -((N : ℚ) * 2 ^ F) ∈ fp32Values := by
  obtain ⟨n, hn⟩ : ∃ n : ℕ, F = (n : ℤ) - 149 := ⟨(F + 149).toNat, by omega⟩
  subst hn
  exact fp32_mem_nat n N hN hmax

lemma fp32_abs_le (q : ℚ) (hq : q ∈ fp32Values) : |q| ≤ 16777215 * 2 ^ (104 : ℤ) := by
  obtain ⟨c, -, h⟩ := hq
  have hm : c % 2 ^ 23 < 8388608 := Nat.mod_lt _ (by norm_num)
  have he : c / 2 ^ 23 % 2 ^ 8 < 256 := Nat.mod_lt _ (by norm_num)
  simp only [ieeeDecode] at h
  generalize c / 2 ^ 23 % 2 ^ 8 = e at h he
  generalize c % 2 ^ 23 = m at h hm
  generalize hs : (if c / 2 ^ (8 + 23) % 2 = 1 then (-1 : ℚ) else 1) = sg at h
  have hsg : |sg| = 1 := by rw [← hs]; split_ifs <;> simp
  have hmq : (m : ℚ) ≤ 8388607 := by exact_mod_cast (by omega : m ≤ 8388607)
  split_ifs at h with h1 h2
  · rw [← Option.some.inj h, abs_mul, abs_mul, hsg, abs_of_nonneg (by positivity),
      abs_of_nonneg (le_of_lt (two_zpow_pos _))]
    have h4 : (m : ℚ) / 2 ^ 23 ≤ 1 := by rw [div_le_one (by norm_num)]; linarith
    have h5 : (2 : ℚ) ^ (1 - (2 ^ (8 - 1) - 1) : ℤ) ≤ 1 := by norm_num
    calc 1 * ((m : ℚ) / 2 ^ 23) * 2 ^ (1 - (2 ^ (8 - 1) - 1) : ℤ) ≤ 1 * 1 * 1 := by
          apply mul_le_mul _ h5 (le_of_lt (two_zpow_pos _)) (by norm_num)
          linarith
      _ ≤ _ := by norm_num
  · rw [← Option.some.inj h, abs_mul, abs_mul, hsg, abs_of_nonneg (by positivity),
      abs_of_nonneg (le_of_lt (two_zpow_pos _))]
    have h3 : (2 : ℚ) ^ ((e : ℤ) - (2 ^ (8 - 1) - 1)) ≤ 2 ^ (127 : ℤ) :=
      zpow_le_zpow_right₀ (by norm_num) (by norm_num; omega)
    have h4 : 1 + (m : ℚ) / 2 ^ 23 ≤ 16777215 / 8388608 := by
      have : (m : ℚ) / 2 ^ 23 ≤ 8388607 / 8388608 := by
        rw [div_le_div_iff₀ (by norm_num) (by norm_num)]; norm_num; linarith
      linarith
    calc 1 * (1 + (m : ℚ) / 2 ^ 23) * 2 ^ ((e : ℤ) - (2 ^ (8 - 1) - 1)) ≤
          1 * (16777215 / 8388608) * 2 ^ (127 : ℤ) := by
          apply mul_le_mul (by linarith) h3 (le_of_lt (two_zpow_pos _)) (by norm_num)
      _ = 16777215 * 2 ^ (104 : ℤ) := by norm_num

lemma e4m3_struct (x : ℚ) (hx : x ∈ e4m3Values) : ∃ (N : ℕ) (F : ℤ), N < 16 ∧ -9 ≤ F ∧
    (N : ℚ) * 2 ^ F ≤ 448 ∧ (x = N * 2 ^ F ∨ x = -(N * 2 ^ F)) := by
  obtain ⟨c, -, h⟩ := hx
  have hm : c % 8 < 8 := Nat.mod_lt _ (by norm_num)
  have he : c / 8 % 16 < 16 := Nat.mod_lt _ (by norm_num)
  simp only [e4m3Decode] at h
  generalize c / 8 % 16 = e at h he
  generalize c % 8 = m at h hm
  generalize hs : (if c / 128 % 2 = 1 then (-1 : ℚ) else 1) = sg at h
  have hsg : ∀ q : ℚ, sg * q = q ∨ sg * q = -q := by
    intro q; rw [← hs]; split_ifs <;> simp
  split_ifs at h with h1 h2
  · have hx := Option.some.inj h
    refine ⟨m, -9, by omega, le_rfl, ?_, ?_⟩
    · have : (m : ℚ) ≤ 7 := by exact_mod_cast (by omega : m ≤ 7)
      norm_num; linarith
    · rw [← hx, mul_assoc]
      have : (m : ℚ) / 8 * 2 ^ (-6 : ℤ) = m * 2 ^ (-9 : ℤ) := by norm_num; ring
      rw [this]; exact hsg _
  · have hx := Option.some.inj h
    refine ⟨8 + m, (e : ℤ) - 10, by omega, by omega, ?_, ?_⟩
    · have hm7 : (m : ℚ) ≤ 7 := by exact_mod_cast (by omega : m ≤ 7)
      have hm6 : e = 15 → (m : ℚ) ≤ 6 := fun he15 => by exact_mod_cast (by omega : m ≤ 6)
      have he1 : 1 ≤ e := by omega
      push_cast
      interval_cases e <;> norm_num <;> first | linarith | linarith [hm6 rfl]
    · rw [← hx, mul_assoc]
      have : (1 + (m : ℚ) / 8) * 2 ^ ((e : ℤ) - 7) = ((8 + m : ℕ) : ℚ) * 2 ^ ((e : ℤ) - 10) := by
        rw [show (e : ℤ) - 7 = ((e : ℤ) - 10) + 3 by ring, zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
        push_cast; ring
      rw [this]; exact hsg _

lemma e2m1_struct (x : ℚ) (hx : x ∈ e2m1Values) : ∃ (N : ℕ) (F : ℤ), N < 4 ∧ -1 ≤ F ∧
    (N : ℚ) * 2 ^ F ≤ 6 ∧ (x = N * 2 ^ F ∨ x = -(N * 2 ^ F)) := by
  obtain ⟨c, -, h⟩ := hx
  have hm : c % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have he : c / 2 % 4 < 4 := Nat.mod_lt _ (by norm_num)
  simp only [e2m1Decode] at h
  generalize c / 2 % 4 = e at h he
  generalize c % 2 = m at h hm
  generalize hs : (if c / 8 % 2 = 1 then (-1 : ℚ) else 1) = sg at h
  have hsg : ∀ q : ℚ, sg * q = q ∨ sg * q = -q := by
    intro q; rw [← hs]; split_ifs <;> simp
  split_ifs at h with h1
  · refine ⟨m, -1, by omega, le_rfl, ?_, ?_⟩
    · have : (m : ℚ) ≤ 1 := by exact_mod_cast (by omega : m ≤ 1)
      norm_num; linarith
    · rw [← h]
      have : (m : ℚ) / 2 = m * 2 ^ (-1 : ℤ) := by norm_num; ring
      rw [this]; exact hsg _
  · refine ⟨2 + m, (e : ℤ) - 2, by omega, by omega, ?_, ?_⟩
    · have : (m : ℚ) ≤ 1 := by exact_mod_cast (by omega : m ≤ 1)
      push_cast
      rcases (by omega : e = 1 ∨ e = 2 ∨ e = 3) with rfl | rfl | rfl <;> norm_num <;> linarith
    · rw [← h, mul_assoc]
      have : (1 + (m : ℚ) / 2) * 2 ^ ((e : ℤ) - 1) = ((2 + m : ℕ) : ℚ) * 2 ^ ((e : ℤ) - 2) := by
        rw [show (e : ℤ) - 1 = ((e : ℤ) - 2) + 1 by ring, zpow_add_one₀ (by norm_num : (2 : ℚ) ≠ 0)]
        push_cast; ring
      rw [this]; exact hsg _

lemma e4m3_448 : (448 : ℚ) ∈ e4m3Values := ⟨126, by norm_num, by norm_num [e4m3Decode]⟩

lemma e4m3_min : (2 : ℚ) ^ (-9 : ℤ) ∈ e4m3Values := ⟨1, by norm_num, by norm_num [e4m3Decode]⟩

lemma e2m1_6 : (6 : ℚ) ∈ e2m1Values := ⟨7, by norm_num, by norm_num [e2m1Decode]⟩

theorem e4m3_ue8m0_bf16_iff (c : ℕ) (hc : c < 255) :
    (∀ x ∈ e4m3Values, x * (2 : ℚ) ^ ((c : ℤ) - 127) ∈ bf16Values) ↔ 3 ≤ c ∧ c ≤ 246 := by
  constructor
  · intro h
    by_contra hcon
    rcases Nat.lt_or_ge c 3 with hlo | hhi
    · have hm := h _ e4m3_min
      have hne : (2 : ℚ) ^ (-9 : ℤ) * 2 ^ ((c : ℤ) - 127) ≠ 0 := by positivity
      have h1 := bf16_abs_ge _ hm hne
      rw [abs_of_pos (by positivity), ← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)] at h1
      have h2 : (2 : ℚ) ^ (-9 + ((c : ℤ) - 127)) < 2 ^ (-133 : ℤ) :=
        zpow_lt_zpow_right₀ (by norm_num) (by omega)
      linarith
    · have hc247 : 247 ≤ c := by omega
      have hm := h _ e4m3_448
      have h1 := bf16_abs_le _ hm
      rw [abs_of_pos (by positivity)] at h1
      have h2 : (2 : ℚ) ^ (120 : ℤ) ≤ 2 ^ ((c : ℤ) - 127) := zpow_le_zpow_right₀ (by norm_num) (by omega)
      have h3 : (255 : ℚ) * 2 ^ (120 : ℤ) < 448 * 2 ^ (120 : ℤ) := by norm_num
      nlinarith
  · rintro ⟨hlo, hhi⟩ x hx
    obtain ⟨N, F, hN, hF, hNF, hxe⟩ := e4m3_struct x hx
    have hj : (2 : ℚ) ^ ((c : ℤ) - 127) ≤ 2 ^ (119 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have hprod : (N : ℚ) * 2 ^ (F + ((c : ℤ) - 127)) = N * 2 ^ F * 2 ^ ((c : ℤ) - 127) := by
      rw [zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]; ring
    have hmax : (N : ℚ) * 2 ^ (F + ((c : ℤ) - 127)) ≤ 255 * 2 ^ (120 : ℤ) := by
      rw [hprod]
      calc (N : ℚ) * 2 ^ F * 2 ^ ((c : ℤ) - 127) ≤ 448 * 2 ^ (119 : ℤ) :=
            mul_le_mul hNF hj (le_of_lt (two_zpow_pos _)) (by norm_num)
        _ ≤ 255 * 2 ^ (120 : ℤ) := by norm_num
    have hmem := bf16_mem N (F + ((c : ℤ) - 127)) (by omega) (by omega) hmax
    rw [hprod] at hmem
    rcases hxe with rfl | rfl
    · exact hmem.1
    · rw [neg_mul]; exact hmem.2

theorem e4m3_ue8m0_bf16_literal_false :
    ¬ ∀ x ∈ e4m3Values, ∀ s ∈ ue8m0Values, x * s ∈ bf16Values := by
  intro h
  have := (e4m3_ue8m0_bf16_iff 254 (by norm_num)).mp (fun x hx => h x hx _ ⟨254, by norm_num, rfl⟩)
  omega

theorem e4m3_ue8m0_bf16_edges :
    (448 : ℚ) ∈ e4m3Values ∧ (2 : ℚ) ^ (120 : ℤ) ∈ ue8m0Values ∧ (448 : ℚ) * 2 ^ (120 : ℤ) ∉ bf16Values ∧
    (2 : ℚ) ^ (-9 : ℤ) ∈ e4m3Values ∧ (2 : ℚ) ^ (-125 : ℤ) ∈ ue8m0Values ∧
      (2 : ℚ) ^ (-9 : ℤ) * 2 ^ (-125 : ℤ) ∉ bf16Values := by
  refine ⟨e4m3_448, ⟨247, by norm_num, by norm_num⟩, ?_, e4m3_min, ⟨2, by norm_num, by norm_num⟩, ?_⟩
  · intro hm
    have h1 := bf16_abs_le _ hm
    rw [abs_of_pos (by positivity)] at h1
    norm_num at h1
  · intro hm
    have h1 := bf16_abs_ge _ hm (by positivity)
    rw [abs_of_pos (by positivity)] at h1
    norm_num at h1

theorem e2m1_ue8m0_fp32_iff (c : ℕ) (hc : c < 255) :
    (∀ x ∈ e2m1Values, x * (2 : ℚ) ^ ((c : ℤ) - 127) ∈ fp32Values) ↔ c ≤ 252 := by
  constructor
  · intro h
    by_contra hcon
    have hm := h _ e2m1_6
    have h1 := fp32_abs_le _ hm
    rw [abs_of_pos (by positivity)] at h1
    have h2 : (2 : ℚ) ^ (126 : ℤ) ≤ 2 ^ ((c : ℤ) - 127) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have h3 : (16777215 : ℚ) * 2 ^ (104 : ℤ) < 6 * 2 ^ (126 : ℤ) := by norm_num
    nlinarith
  · intro hhi x hx
    obtain ⟨N, F, hN, hF, hNF, hxe⟩ := e2m1_struct x hx
    have hj : (2 : ℚ) ^ ((c : ℤ) - 127) ≤ 2 ^ (125 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    have hprod : (N : ℚ) * 2 ^ (F + ((c : ℤ) - 127)) = N * 2 ^ F * 2 ^ ((c : ℤ) - 127) := by
      rw [zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]; ring
    have hmax : (N : ℚ) * 2 ^ (F + ((c : ℤ) - 127)) ≤ 16777215 * 2 ^ (104 : ℤ) := by
      rw [hprod]
      calc (N : ℚ) * 2 ^ F * 2 ^ ((c : ℤ) - 127) ≤ 6 * 2 ^ (125 : ℤ) :=
            mul_le_mul hNF hj (le_of_lt (two_zpow_pos _)) (by norm_num)
        _ ≤ 16777215 * 2 ^ (104 : ℤ) := by norm_num
    have hmem := fp32_mem N (F + ((c : ℤ) - 127)) (by omega) (by omega) hmax
    rw [hprod] at hmem
    rcases hxe with rfl | rfl
    · exact hmem.1
    · rw [neg_mul]; exact hmem.2

theorem e2m1_ue8m0_fp32_literal_false :
    ¬ ∀ x ∈ e2m1Values, ∀ s ∈ ue8m0Values, x * s ∈ fp32Values := by
  intro h
  have := (e2m1_ue8m0_fp32_iff 254 (by norm_num)).mp (fun x hx => h x hx _ ⟨254, by norm_num, rfl⟩)
  omega

/-! 3b: the scale over the reals -/

lemma ue8m0Scale_pos (qmax amax : ℝ) : 0 < ue8m0Scale qmax amax := zpow_pos (by norm_num) _

lemma ue8m0Scale_ge (qmax amax : ℝ) (hq : 0 < qmax) (ha : 0 < amax) : amax / qmax ≤ ue8m0Scale qmax amax := by
  unfold ue8m0Scale
  have hpos : 0 < amax / qmax := div_pos ha hq
  calc amax / qmax = (2 : ℝ) ^ (Real.logb 2 (amax / qmax)) :=
        (Real.rpow_logb (by norm_num) (by norm_num) hpos).symm
    _ ≤ (2 : ℝ) ^ ((⌈Real.logb 2 (amax / qmax)⌉ : ℤ) : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (Int.le_ceil _)
    _ = (2 : ℝ) ^ ⌈Real.logb 2 (amax / qmax)⌉ := Real.rpow_intCast _ _

theorem ue8m0Scale_bound (qmax amax x : ℝ) (hq : 0 < qmax) (ha : 0 < amax) (hx : |x| ≤ amax) :
    |x / ue8m0Scale qmax amax| ≤ qmax := by
  have hs := ue8m0Scale_pos qmax amax
  have hge := ue8m0Scale_ge qmax amax hq ha
  rw [abs_div, abs_of_pos hs, div_le_iff₀ hs]
  rw [div_le_iff₀ hq] at hge
  linarith

theorem ue8m0Scale_tight (qmax amax : ℝ) (hq : 0 < qmax) (ha : 0 < amax) :
    qmax / 2 < amax / ue8m0Scale qmax amax := by
  have hs := ue8m0Scale_pos qmax amax
  have hpos : 0 < amax / qmax := div_pos ha hq
  have hlt : ue8m0Scale qmax amax < 2 * (amax / qmax) := by
    unfold ue8m0Scale
    calc (2 : ℝ) ^ ⌈Real.logb 2 (amax / qmax)⌉ = (2 : ℝ) ^ ((⌈Real.logb 2 (amax / qmax)⌉ : ℤ) : ℝ) :=
          (Real.rpow_intCast _ _).symm
      _ < (2 : ℝ) ^ (Real.logb 2 (amax / qmax) + 1) :=
          Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (Int.ceil_lt_add_one _)
      _ = 2 * (amax / qmax) := by
          rw [Real.rpow_add (by norm_num), Real.rpow_logb (by norm_num) (by norm_num) hpos, Real.rpow_one]
          ring
  rw [lt_div_iff₀ hs]
  have : qmax / 2 * ue8m0Scale qmax amax < qmax / 2 * (2 * (amax / qmax)) :=
    mul_lt_mul_of_pos_left hlt (by positivity)
  calc qmax / 2 * ue8m0Scale qmax amax < qmax / 2 * (2 * (amax / qmax)) := this
    _ = amax := by field_simp

theorem ue8m0Scale_e4m3_exponent (amax : ℝ) (h1 : 1 / 10000 ≤ amax) (h2 : amax ≤ 2 ^ 128) :
    -22 ≤ ⌈Real.logb 2 (amax / 448)⌉ ∧ ⌈Real.logb 2 (amax / 448)⌉ ≤ 120 := by
  have hpos : 0 < amax / 448 := by positivity
  constructor
  · have : (-23 : ℤ) < ⌈Real.logb 2 (amax / 448)⌉ := by
      rw [Int.lt_ceil, Real.lt_logb_iff_rpow_lt (by norm_num) hpos]
      have hz : (2 : ℝ) ^ (((-23 : ℤ) : ℝ)) = 2 ^ (-23 : ℤ) := Real.rpow_intCast _ _
      rw [hz]
      have : (2 : ℝ) ^ (-23 : ℤ) < 1 / 10000 / 448 := by norm_num
      have : 1 / 10000 / 448 ≤ amax / 448 := by linarith
      linarith
    omega
  · rw [Int.ceil_le, Real.logb_le_iff_le_rpow (by norm_num) hpos]
    have hz : (2 : ℝ) ^ (((120 : ℤ) : ℝ)) = 2 ^ (120 : ℤ) := Real.rpow_intCast _ _
    rw [hz]
    have : (2 : ℝ) ^ (128 : ℕ) / 448 ≤ 2 ^ (120 : ℤ) := by norm_num
    have : amax / 448 ≤ 2 ^ (128 : ℕ) / 448 := by linarith
    linarith

end NumericsProof
