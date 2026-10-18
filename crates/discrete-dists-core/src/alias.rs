use crate::{Distribution, Rng, Tree, TreeError};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::iter;

// Walker's alias method, built with Vose's algorithm. Column `c` accepts its own element with
// probability `cutoffs[c]` and otherwise yields `aliases[c]`. An empty `cutoffs` marks a defunct
// table, while `probs` keeps its zeros so `probs()` still answers.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Alias {
    lo: i64,
    probs: Vec<f64>,
    cutoffs: Vec<f64>,
    aliases: Vec<u32>,
}

impl Alias {
    pub fn new(lo: i64, weights: &[f64]) -> Result<Self, TreeError> {
        for (idx, &value) in weights.iter().enumerate() {
            Tree::validate_weight(idx as i64, value)?;
        }
        if u32::try_from(weights.len()).is_err() {
            return Err(TreeError::InvalidSupport {
                reason: format!("alias table has more than {} elements", u32::MAX),
            });
        }

        let total: f64 = weights.iter().sum();
        if total == 0.0 {
            return Ok(Alias {
                lo,
                probs: vec![0.0; weights.len()],
                cutoffs: Vec::new(),
                aliases: Vec::new(),
            });
        }
        if !total.is_finite() {
            return Err(TreeError::InvalidSupport {
                reason: format!("total weight must be finite, got {total}"),
            });
        }

        let n = weights.len();
        let mut scaled: Vec<f64> = weights.iter().map(|&w| w * n as f64 / total).collect();
        let (mut small, mut large): (Vec<usize>, Vec<usize>) =
            (0..n).partition(|&i| scaled[i] < 1.0);
        // Columns left unpaired by rounding keep the defaults: they accept themselves with certainty.
        let mut cutoffs = vec![1.0; n];
        let mut aliases: Vec<u32> = (0..n as u32).collect();
        while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
            small.pop();
            cutoffs[s] = scaled[s];
            aliases[s] = l as u32;
            scaled[l] -= 1.0 - cutoffs[s];
            if scaled[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }

        Ok(Alias {
            lo,
            probs: weights.iter().map(|&w| w / total).collect(),
            cutoffs,
            aliases,
        })
    }

    pub fn support(&self) -> (i64, i64) {
        (self.lo, self.lo + self.probs.len() as i64)
    }

    pub fn len(&self) -> usize {
        self.probs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    fn position(&self, element: i64) -> Option<usize> {
        usize::try_from(element.checked_sub(self.lo)?).ok()
    }

    fn draw(&self, column: usize, u: f64) -> i64 {
        let offset = if u < self.cutoffs[column] {
            column
        } else {
            self.aliases[column] as usize
        };

        self.lo + offset as i64
    }

    /// Draws `n` distinct elements by rejection, keeping first appearances in draw order.
    pub fn sample_without_replacement(
        &self,
        rng: &mut Rng,
        n: usize,
    ) -> Result<Vec<i64>, TreeError> {
        let found = self.probs.iter().filter(|&&p| p > 0.0).count();
        if found < n {
            return Err(TreeError::NotEnoughElements {
                requested: n,
                found,
            });
        }

        let budget = 50 * n + 1000;
        let mut seen = HashSet::with_capacity(n);
        let mut unique = Vec::with_capacity(n);
        let mut drawn = 0;

        while unique.len() < n {
            if drawn >= budget {
                return Err(TreeError::NotEnoughElements {
                    requested: n,
                    found: unique.len(),
                });
            }

            let size = (2 * (n - unique.len())).clamp(1, budget - drawn);
            let mut draws = vec![0i64; size];
            self.sample(rng, &mut draws)?;
            drawn += size;

            for x in draws {
                if seen.insert(x) {
                    unique.push(x);
                    if unique.len() == n {
                        break;
                    }
                }
            }
        }

        Ok(unique)
    }
}

impl Distribution for Alias {
    fn is_defunct(&self) -> bool {
        self.cutoffs.is_empty()
    }

    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        if out.is_empty() {
            return Ok(());
        }
        if self.is_defunct() {
            return Err(TreeError::EmptyDistribution);
        }

        let n = self.cutoffs.len() as u64;
        for x in out {
            let column = rng.below(n) as usize;
            let u = rng.next_f64();
            *x = self.draw(column, u);
        }

        Ok(())
    }

    // Slot `j` is confined to the `j`-th equal slice of the columns, so every column carries mass 1/n and the draws cover them evenly.
    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        if out.is_empty() {
            return Ok(());
        }
        if self.is_defunct() {
            return Err(TreeError::EmptyDistribution);
        }

        let n = self.cutoffs.len();
        let m = out.len() as f64;
        for (j, x) in out.iter_mut().enumerate() {
            let column = ((((j as f64 + rng.next_f64()) / m) * n as f64) as usize).min(n - 1);
            let u = rng.next_f64();
            *x = self.draw(column, u);
        }

        Ok(())
    }

    fn probs(&self, elements: &[i64], out: &mut [f64]) {
        assert_eq!(elements.len(), out.len());
        for (&element, p) in iter::zip(elements, out) {
            *p = self
                .position(element)
                .and_then(|i| self.probs.get(i))
                .copied()
                .unwrap_or(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use proptest::collection::vec;
    use proptest::prelude::*;

    fn weight() -> impl Strategy<Value = f64> {
        prop_oneof![
            2 => Just(0.0),
            1 => Just(1e-9),
            3 => 0.0..=1e6f64,
        ]
    }

    #[test]
    fn probs_are_normalized_weights() {
        let table = Alias::new(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut out = [0.0; 4];

        table.probs(&[0, 1, 2, 3], &mut out);

        for (p, expected) in out.iter().zip([0.1, 0.2, 0.3, 0.4]) {
            assert!((p - expected).abs() < 1e-12, "got {p}, expected {expected}");
        }
    }

    proptest! {
        #[test]
        fn columns_reconstruct_the_normalized_weights(
            weights in vec(weight(), 1..=64),
        ) {
            prop_assume!(weights.iter().any(|&w| w > 0.0));
            let table = Alias::new(0, &weights).unwrap();
            let n = weights.len() as f64;

            prop_assert!(table.cutoffs.iter().all(|c| (0.0..=1.0).contains(c)));
            for (e, &p) in table.probs.iter().enumerate() {
                let aliased: f64 = (0..weights.len())
                    .filter(|&c| table.aliases[c] as usize == e)
                    .map(|c| 1.0 - table.cutoffs[c])
                    .sum();
                let mass = (table.cutoffs[e] + aliased) / n;
                prop_assert!((mass - p).abs() < 1e-12, "element {e}: got {mass}, expected {p}");
            }
        }
    }

    #[test]
    fn sample_frequencies_match_weights() {
        let weights = [1.0, 2.0, 4.0, 8.0, 16.0];
        let total: f64 = weights.iter().sum();
        let table = Alias::new(0, &weights).unwrap();
        let mut out = vec![0i64; 200_000];

        table.sample(&mut Rng::new(7), &mut out).unwrap();

        let mut counts = [0usize; 5];
        for &x in &out {
            counts[x as usize] += 1;
        }
        for (k, (&count, w)) in counts.iter().zip(weights).enumerate() {
            let freq = count as f64 / out.len() as f64;
            assert!((freq - w / total).abs() < 0.01, "element {k}: got {freq}");
        }
    }

    #[test]
    fn zero_weight_elements_are_never_drawn() {
        let table = Alias::new(0, &[0.0, 3.0, 0.0, 1.0, 0.0]).unwrap();
        let mut rng = Rng::new(2);
        let mut out = vec![0i64; 10_000];

        table.sample(&mut rng, &mut out).unwrap();
        assert!(out.iter().all(|x| [1, 3].contains(x)));

        table.stratified_sample(&mut rng, &mut out).unwrap();
        assert!(out.iter().all(|x| [1, 3].contains(x)));
    }

    #[test]
    fn single_positive_weight_is_always_drawn() {
        let table = Alias::new(0, &[0.0, 0.0, 5.0, 0.0]).unwrap();
        let mut out = [0i64; 100];

        table.sample(&mut Rng::new(3), &mut out).unwrap();

        assert_eq!(out, [2; 100]);
    }

    #[test]
    fn stratified_sample_hits_each_column_once_when_n_equals_len() {
        let table = Alias::new(0, &[1.0; 8]).unwrap();
        let mut out = [0i64; 8];

        table.stratified_sample(&mut Rng::new(5), &mut out).unwrap();

        out.sort_unstable();
        assert_eq!(out, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn zero_total_table_is_defunct_and_errors_on_draws() {
        let table = Alias::new(4, &[0.0, 0.0]).unwrap();
        let mut out = [-7i64; 3];
        let mut probs = [1.0; 2];

        assert!(table.is_defunct());
        assert!(Alias::new(0, &[]).unwrap().is_defunct());
        assert_eq!(
            table.sample(&mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(
            table.stratified_sample(&mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(out, [-7; 3]);
        assert_eq!(table.sample(&mut Rng::new(1), &mut []), Ok(()));
        assert_eq!(table.stratified_sample(&mut Rng::new(1), &mut []), Ok(()));
        table.probs(&[4, 5], &mut probs);
        assert_eq!(probs, [0.0, 0.0]);
    }

    #[test]
    fn support_starts_at_lo() {
        let table = Alias::new(10, &[1.0, 3.0]).unwrap();
        let mut probs = [0.0; 4];
        let mut out = vec![0i64; 1000];

        table.probs(&[9, 10, 11, 12], &mut probs);
        table.sample(&mut Rng::new(4), &mut out).unwrap();

        assert_eq!(table.support(), (10, 12));
        assert_eq!(probs, [0.0, 0.25, 0.75, 0.0]);
        assert!(out.iter().all(|x| [10, 11].contains(x)));
    }

    #[test]
    fn invalid_weights_are_rejected_with_their_index() {
        assert!(matches!(
            Alias::new(0, &[1.0, f64::NAN]),
            Err(TreeError::NonFiniteWeight { idx: 1, .. })
        ));
        assert!(matches!(
            Alias::new(0, &[1.0, 2.0, -1.0]),
            Err(TreeError::NegativeWeight { idx: 2, .. })
        ));
        assert!(matches!(
            Alias::new(0, &[f64::MAX, f64::MAX]),
            Err(TreeError::InvalidSupport { .. })
        ));
    }

    #[test]
    fn without_replacement_returns_distinct_positive_elements() {
        let table = Alias::new(0, &[1.0, 0.0, 2.0, 3.0, 0.0, 4.0]).unwrap();
        let positive = [0, 2, 3, 5];

        for seed in 0..100 {
            let drawn = table
                .sample_without_replacement(&mut Rng::new(seed), 3)
                .unwrap();
            let distinct: HashSet<i64> = drawn.iter().copied().collect();

            assert_eq!(drawn.len(), 3, "seed {seed}");
            assert_eq!(distinct.len(), 3, "seed {seed}");
            assert!(drawn.iter().all(|x| positive.contains(x)), "seed {seed}");
        }

        let mut all = table
            .sample_without_replacement(&mut Rng::new(0), 4)
            .unwrap();
        all.sort_unstable();
        assert_eq!(all, positive);
    }

    #[test]
    fn without_replacement_rejects_more_draws_than_positive_elements() {
        let table = Alias::new(0, &[1.0, 0.0, 2.0, 0.0]).unwrap();

        assert_eq!(
            table.sample_without_replacement(&mut Rng::new(1), 3),
            Err(TreeError::NotEnoughElements {
                requested: 3,
                found: 2,
            })
        );
    }

    #[test]
    fn without_replacement_fails_when_a_rare_element_exhausts_the_budget() {
        let table = Alias::new(0, &[1.0, 1e-12]).unwrap();

        assert_eq!(
            table.sample_without_replacement(&mut Rng::new(1), 2),
            Err(TreeError::NotEnoughElements {
                requested: 2,
                found: 1,
            })
        );
    }

    #[test]
    fn serde_round_trip_preserves_table_and_draws() {
        let config = bincode::config::standard();
        let table = Alias::new(-3, &[0.5, 0.0, 2.5]).unwrap();

        let bytes = bincode::serde::encode_to_vec(&table, config).unwrap();
        let (restored, _): (Alias, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();

        assert_eq!(restored, table);
        let mut a = [0i64; 20];
        let mut b = [0i64; 20];
        restored.sample(&mut Rng::new(9), &mut a).unwrap();
        table.sample(&mut Rng::new(9), &mut b).unwrap();
        assert_eq!(a, b);
    }
}
