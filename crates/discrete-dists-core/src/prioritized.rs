use crate::{Combine, Distribution, Max, Min, Proportional, Rng, SegmentTree, Tree, TreeError};
use serde::{Deserialize, Serialize};
use std::iter;

// The sum tree holds priority^alpha for sampling. The min and max trees hold what importance weights and new-element priorities need.
#[derive(Serialize, Deserialize, Clone)]
pub struct Prioritized {
    alpha: f64,
    sums: Tree,
    mins: SegmentTree<Min>,
    maxs: SegmentTree<Max>,
}

fn validate_nonnegative(name: &str, value: f64) -> Result<(), TreeError> {
    if !value.is_finite() || value < 0.0 {
        return Err(TreeError::InvalidPriority {
            reason: format!("{name} must be finite and nonnegative, got {value}"),
        });
    }

    Ok(())
}

impl Prioritized {
    pub fn new(size: u32, alpha: f64) -> Result<Self, TreeError> {
        validate_nonnegative("alpha", alpha)?;

        Ok(Prioritized {
            alpha,
            sums: Tree::new(size),
            mins: SegmentTree::new(size),
            maxs: SegmentTree::new(size),
        })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn size(&self) -> u32 {
        self.sums.size()
    }

    pub fn max_priority(&self) -> Option<f64> {
        let max = self.maxs.root();
        (max != Max::IDENTITY).then_some(max)
    }

    fn view(&self) -> Proportional<&Tree> {
        Proportional::from_tree(0, &self.sums)
    }

    /// A priority of 0 removes the element from sampling.
    pub fn update(&mut self, elements: &[i64], priorities: &[f64]) -> Result<(), TreeError> {
        if elements.len() != priorities.len() {
            return Err(TreeError::LengthMismatch {
                idxs: elements.len(),
                values: priorities.len(),
            });
        }

        let mut stored = Vec::with_capacity(priorities.len());
        for (&element, &priority) in iter::zip(elements, priorities) {
            Tree::validate_weight(element, priority)?;
            // 0^0 is 1, so a zero priority must be special-cased rather than raised to alpha.
            let value = if priority == 0.0 {
                0.0
            } else {
                priority.powf(self.alpha)
            };
            Tree::validate_weight(element, value)?;
            stored.push(value);
        }

        // The sum tree checks every index before writing, so the other trees cannot fail below.
        self.sums.update(elements, &stored)?;
        for ((&element, &priority), &value) in iter::zip(iter::zip(elements, priorities), &stored) {
            // A priority whose power underflows to 0 can never be drawn, so it is absent too.
            if value == 0.0 {
                self.mins.reset(element)?;
                self.maxs.reset(element)?;
            } else {
                self.mins.set(element, value)?;
                self.maxs.set(element, priority)?;
            }
        }

        Ok(())
    }

    pub fn sample_without_replacement(
        &mut self,
        rng: &mut Rng,
        n: usize,
        out: &mut Vec<i64>,
    ) -> Result<(), TreeError> {
        Proportional::from_tree(0, &mut self.sums).sample_without_replacement(rng, n, out)
    }

    // With N present elements and total mass T, (N * P_i)^-beta / max_j (N * P_j)^-beta reduces to (stored_i / min stored)^-beta because N and T cancel.
    pub fn importance_weights(
        &self,
        elements: &[i64],
        beta: f64,
        out: &mut [f64],
    ) -> Result<(), TreeError> {
        validate_nonnegative("beta", beta)?;
        assert_eq!(elements.len(), out.len());

        let stored = self.sums.get_values(elements)?;
        if stored.is_empty() {
            return Ok(());
        }

        let p_min = self.mins.root();
        if p_min == Min::IDENTITY {
            return Err(TreeError::EmptyDistribution);
        }

        for (&s, w) in iter::zip(&stored, out) {
            *w = if s == 0.0 {
                0.0
            } else {
                (s / p_min).powf(-beta)
            };
        }

        Ok(())
    }

    pub fn sample_with_weights(
        &self,
        rng: &mut Rng,
        beta: f64,
        stratified: bool,
        out: &mut [i64],
        weights: &mut [f64],
    ) -> Result<(), TreeError> {
        assert_eq!(out.len(), weights.len());

        if stratified {
            self.stratified_sample(rng, out)?;
        } else {
            self.sample(rng, out)?;
        }

        self.importance_weights(out, beta, weights)
    }
}

impl Distribution for Prioritized {
    fn is_defunct(&self) -> bool {
        self.view().is_defunct()
    }

    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        self.view().sample(rng, out)
    }

    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        self.view().stratified_sample(rng, out)
    }

    fn probs(&self, elements: &[i64], out: &mut [f64]) {
        self.view().probs(elements, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use proptest::collection::vec;
    use proptest::prelude::*;

    // Follows the formula directly, without the N and T cancellation the implementation relies on.
    fn brute_weights(priorities: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
        let stored: Vec<f64> = priorities
            .iter()
            .map(|&p| if p == 0.0 { 0.0 } else { p.powf(alpha) })
            .collect();
        let total: f64 = stored.iter().sum();
        let n = priorities.iter().filter(|&&p| p > 0.0).count() as f64;
        let raw: Vec<f64> = iter::zip(priorities, &stored)
            .map(|(&p, &s)| {
                if p > 0.0 {
                    (n * s / total).powf(-beta)
                } else {
                    0.0
                }
            })
            .collect();
        let max = raw.iter().copied().fold(0.0, f64::max);

        raw.iter().map(|r| r / max).collect()
    }

    #[test]
    fn new_rejects_invalid_alpha() {
        for alpha in [-0.5, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    Prioritized::new(4, alpha),
                    Err(TreeError::InvalidPriority { .. })
                ),
                "alpha {alpha}"
            );
        }
    }

    #[test]
    fn update_rejects_bad_batch_without_changing_state() {
        let mut p = Prioritized::new(4, 1.0).unwrap();
        p.update(&[0], &[2.0]).unwrap();

        assert!(matches!(
            p.update(&[1, 2], &[1.0, f64::NAN]),
            Err(TreeError::NonFiniteWeight { idx: 2, .. })
        ));
        assert!(matches!(
            p.update(&[1, 2], &[1.0, -1.0]),
            Err(TreeError::NegativeWeight { idx: 2, .. })
        ));
        assert_eq!(
            p.update(&[1, 9], &[1.0, 1.0]),
            Err(TreeError::IndexOutOfBounds { idx: 9, size: 4 })
        );
        assert_eq!(
            p.update(&[1], &[1.0, 2.0]),
            Err(TreeError::LengthMismatch { idxs: 1, values: 2 })
        );

        assert_eq!(p.max_priority(), Some(2.0));
        let mut probs = [0.0; 2];
        p.probs(&[0, 1], &mut probs);
        assert_eq!(probs, [1.0, 0.0]);
    }

    #[test]
    fn priority_whose_power_leaves_f64_range_is_absent_or_rejected() {
        let mut p = Prioritized::new(3, 2.0).unwrap();
        p.update(&[0, 1], &[1e-300, 1.0]).unwrap();

        let mut w = [0.0; 2];
        p.importance_weights(&[0, 1], 0.5, &mut w).unwrap();
        assert_eq!(w, [0.0, 1.0]);
        assert_eq!(p.max_priority(), Some(1.0));

        assert!(matches!(
            p.update(&[2, 1], &[3.0, 1e200]),
            Err(TreeError::NonFiniteWeight { idx: 1, .. })
        ));
        let mut probs = [0.0; 3];
        p.probs(&[0, 1, 2], &mut probs);
        assert_eq!(probs, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn importance_weights_reject_invalid_beta_empty_distribution_and_out_of_range() {
        let mut p = Prioritized::new(3, 1.0).unwrap();
        let mut w = [0.0; 1];
        assert_eq!(
            p.importance_weights(&[0], 1.0, &mut w),
            Err(TreeError::EmptyDistribution)
        );

        p.update(&[0], &[1.0]).unwrap();
        for beta in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    p.importance_weights(&[0], beta, &mut w),
                    Err(TreeError::InvalidPriority { .. })
                ),
                "beta {beta}"
            );
        }
        assert_eq!(
            p.importance_weights(&[3], 1.0, &mut w),
            Err(TreeError::IndexOutOfBounds { idx: 3, size: 3 })
        );
    }

    #[test]
    fn min_priority_element_has_weight_one() {
        let mut p = Prioritized::new(4, 0.6).unwrap();
        p.update(&[0, 1, 2], &[2.0, 0.5, 3.0]).unwrap();
        let mut w = [0.0; 3];

        p.importance_weights(&[0, 1, 2], 0.5, &mut w).unwrap();

        assert_eq!(w[1], 1.0);
        assert!(w[0] < 1.0 && w[2] < 1.0);
    }

    #[test]
    fn zero_alpha_is_uniform_with_unit_weights() {
        let mut p = Prioritized::new(5, 0.0).unwrap();
        p.update(&[0, 2, 4], &[0.5, 3.0, 7.0]).unwrap();
        let mut probs = [0.0; 3];
        let mut w = [0.0; 3];

        p.probs(&[0, 2, 4], &mut probs);
        p.importance_weights(&[0, 2, 4], 1.0, &mut w).unwrap();

        for x in probs {
            assert!((x - 1.0 / 3.0).abs() < 1e-12);
        }
        assert_eq!(w, [1.0; 3]);
    }

    #[test]
    fn zero_priority_removes_element_from_probs_weights_and_extremes() {
        let mut p = Prioritized::new(3, 1.0).unwrap();
        p.update(&[0, 1, 2], &[1.0, 4.0, 2.0]).unwrap();
        let mut probs = [0.0; 3];
        let mut w = [0.0; 3];

        p.update(&[1], &[0.0]).unwrap();
        assert_eq!(p.max_priority(), Some(2.0));
        p.probs(&[0, 1, 2], &mut probs);
        assert_eq!(probs[1], 0.0);

        p.update(&[0], &[0.0]).unwrap();
        p.importance_weights(&[0, 1, 2], 1.0, &mut w).unwrap();
        assert_eq!(w, [0.0, 0.0, 1.0]);

        p.update(&[2], &[0.0]).unwrap();
        assert_eq!(p.max_priority(), None);
    }

    #[test]
    fn zero_draws_on_empty_distribution_succeed() {
        let p = Prioritized::new(3, 1.0).unwrap();
        let mut rng = Rng::new(0);

        assert_eq!(
            p.sample_with_weights(&mut rng, 1.0, false, &mut [], &mut []),
            Ok(())
        );
        assert_eq!(p.importance_weights(&[], 1.0, &mut []), Ok(()));
    }

    #[test]
    fn sample_with_weights_returns_importance_weights_of_its_samples() {
        let mut p = Prioritized::new(10, 0.6).unwrap();
        p.update(&[1, 4, 7], &[1.0, 3.0, 0.2]).unwrap();
        let mut out = [0i64; 16];
        let mut weights = [0.0; 16];

        p.sample_with_weights(&mut Rng::new(3), 0.4, true, &mut out, &mut weights)
            .unwrap();

        let mut expected = [0.0; 16];
        p.importance_weights(&out, 0.4, &mut expected).unwrap();
        assert_eq!(weights, expected);
        assert!(out.iter().all(|x| [1, 4, 7].contains(x)));
    }

    #[test]
    fn sample_without_replacement_returns_distinct_present_elements() {
        let mut p = Prioritized::new(6, 1.0).unwrap();
        p.update(&[0, 2, 5], &[1.0, 2.0, 3.0]).unwrap();
        let mut out = Vec::new();

        p.sample_without_replacement(&mut Rng::new(1), 3, &mut out)
            .unwrap();

        out.sort_unstable();
        assert_eq!(out, vec![0, 2, 5]);
        let mut probs = [0.0; 3];
        p.probs(&[0, 2, 5], &mut probs);
        assert!((probs[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bincode_round_trip_preserves_extremes_and_weights() {
        let config = bincode::config::standard();
        let mut p = Prioritized::new(5, 0.6).unwrap();
        p.update(&[0, 3], &[2.0, 0.5]).unwrap();

        let bytes = bincode::serde::encode_to_vec(&p, config).unwrap();
        let (restored, _): (Prioritized, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();

        assert_eq!(restored.size(), 5);
        assert_eq!(restored.alpha(), 0.6);
        assert_eq!(restored.max_priority(), Some(2.0));
        let mut expected = [0.0; 2];
        let mut actual = [0.0; 2];
        p.importance_weights(&[0, 3], 0.5, &mut expected).unwrap();
        restored
            .importance_weights(&[0, 3], 0.5, &mut actual)
            .unwrap();
        assert_eq!(actual, expected);
    }

    proptest! {
        #[test]
        fn weights_match_paper_formula(
            priorities in vec(prop_oneof![1 => Just(0.0), 4 => 0.01..100.0f64], 1..=64),
            alpha in 0.0..=2.0f64,
            beta in 0.0..=2.0f64,
        ) {
            prop_assume!(priorities.iter().any(|&p| p > 0.0));
            let elements: Vec<i64> = (0..priorities.len() as i64).collect();
            let mut p = Prioritized::new(priorities.len() as u32, alpha).unwrap();
            p.update(&elements, &priorities).unwrap();
            let mut weights = vec![0.0; elements.len()];

            p.importance_weights(&elements, beta, &mut weights).unwrap();

            let expected = brute_weights(&priorities, alpha, beta);
            for (w, e) in iter::zip(weights, expected) {
                prop_assert!((w - e).abs() <= 1e-12, "got {w}, expected {e}");
            }
        }

        #[test]
        fn extremes_track_random_update_sequences(
            size in 1u32..=32,
            ops in vec((0..32i64, prop_oneof![1 => Just(0.0), 4 => 0.01..100.0f64]), 0..=100),
        ) {
            let alpha = 0.7;
            let mut p = Prioritized::new(size, alpha).unwrap();
            let mut model = vec![0.0f64; size as usize];

            for (element, priority) in ops {
                let element = element % size as i64;
                p.update(&[element], &[priority]).unwrap();
                model[element as usize] = priority;

                let present: Vec<f64> = model.iter().copied().filter(|&x| x > 0.0).collect();
                let stored: Vec<f64> = present.iter().map(|x| x.powf(alpha)).collect();
                prop_assert_eq!(p.max_priority(), present.iter().copied().reduce(f64::max));
                prop_assert_eq!(p.mins.root(), stored.iter().copied().reduce(f64::min).unwrap_or(f64::INFINITY));
                prop_assert_eq!(p.maxs.root(), present.iter().copied().reduce(f64::max).unwrap_or(f64::NEG_INFINITY));
            }
        }
    }
}
