use crate::{Rng, Tree, TreeError};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter;

// Keys are stored as logs so that `ln(u) / w` does not underflow where `u^(1/w)` would.
/// Elements are opaque labels: offering one that is already stored keeps both entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedReservoir {
    capacity: u32,
    heap: BinaryHeap<Entry>,
    skip: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offer {
    pub admitted: bool,
    pub evicted: Option<i64>,
}

// Min-heap on `log_key`: BinaryHeap is a max-heap, so the comparison is reversed.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Entry {
    log_key: f64,
    element: i64,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .log_key
            .total_cmp(&self.log_key)
            .then_with(|| other.element.cmp(&self.element))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Entry {}

// Uniform in (0, 1], so its log is always finite.
fn ln_open_unit(rng: &mut Rng) -> f64 {
    (1.0 - rng.next_f64()).ln()
}

// Weight to skip before the next replacement. Its survival function is T^x for threshold T, so the jump is ln(u) / ln(T).
fn next_skip(rng: &mut Rng, ln_threshold: f64) -> f64 {
    if ln_threshold == 0.0 {
        return f64::INFINITY;
    }

    ln_open_unit(rng) / ln_threshold
}

impl WeightedReservoir {
    pub fn new(capacity: u32) -> Self {
        WeightedReservoir {
            capacity,
            heap: BinaryHeap::new(),
            skip: 0.0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn elements(&self) -> Vec<i64> {
        self.heap.iter().map(|entry| entry.element).collect()
    }

    pub fn clear(&mut self) {
        self.heap.clear();
        self.skip = 0.0;
    }

    pub fn offer(&mut self, rng: &mut Rng, element: i64, weight: f64) -> Result<Offer, TreeError> {
        Tree::validate_weight(element, weight)?;

        Ok(self.admit(rng, element, weight))
    }

    /// Offers every element in order. `admitted` is overwritten and `evicted` is cleared first.
    pub fn offer_batch(
        &mut self,
        rng: &mut Rng,
        elements: &[i64],
        weights: &[f64],
        admitted: &mut [bool],
        evicted: &mut Vec<i64>,
    ) -> Result<(), TreeError> {
        for len in [weights.len(), admitted.len()] {
            if len != elements.len() {
                return Err(TreeError::LengthMismatch {
                    idxs: elements.len(),
                    values: len,
                });
            }
        }
        for (&element, &weight) in iter::zip(elements, weights) {
            Tree::validate_weight(element, weight)?;
        }

        evicted.clear();
        for ((&element, &weight), slot) in iter::zip(iter::zip(elements, weights), admitted) {
            let offer = self.admit(rng, element, weight);
            *slot = offer.admitted;
            evicted.extend(offer.evicted);
        }

        Ok(())
    }

    // Callers must have validated `weight`.
    fn admit(&mut self, rng: &mut Rng, element: i64, weight: f64) -> Offer {
        let not_admitted = Offer {
            admitted: false,
            evicted: None,
        };
        if weight == 0.0 || self.capacity == 0 {
            return not_admitted;
        }

        if self.heap.len() < self.capacity as usize {
            let log_key = ln_open_unit(rng) / weight;
            self.heap.push(Entry { log_key, element });
            if self.heap.len() == self.capacity as usize {
                self.skip = next_skip(rng, self.threshold());
            }
            return Offer {
                admitted: true,
                evicted: None,
            };
        }

        self.skip -= weight;
        if self.skip > 0.0 {
            return not_admitted;
        }

        // A-ExpJ (Efraimidis & Spirakis, 2006). The replacement key must beat the threshold T, so its
        // weighted key is drawn uniformly from (T^w, 1] rather than from (0, 1]; drawing it from
        // the full range would waste the skip that was just consumed.
        // With r = t + (1 - t) u, 1 - r = (1 - t)(1 - u); expm1 and ln_1p keep small weights from cancelling to 0.
        let one_minus_t = -(weight * self.threshold()).exp_m1();
        let log_key = (-(one_minus_t * (1.0 - rng.next_f64()))).ln_1p() / weight;
        let evicted = self.heap.pop().map(|entry| entry.element);
        self.heap.push(Entry { log_key, element });
        self.skip = next_skip(rng, self.threshold());

        Offer {
            admitted: true,
            evicted,
        }
    }

    // The smallest key in the reservoir. Only called when the reservoir is non-empty.
    fn threshold(&self) -> f64 {
        self.heap
            .peek()
            .expect("threshold requires a non-empty reservoir")
            .log_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    fn frequencies(counts: &[u64], trials: u64) -> Vec<f64> {
        counts.iter().map(|&c| c as f64 / trials as f64).collect()
    }

    // Final members of a reservoir fed `weights` in order, over independent trials.
    fn inclusion_frequencies(capacity: u32, weights: &[f64], trials: u64) -> Vec<f64> {
        let mut rng = Rng::new(0);
        let mut counts = vec![0u64; weights.len()];
        for _ in 0..trials {
            let mut reservoir = WeightedReservoir::new(capacity);
            for (i, &w) in weights.iter().enumerate() {
                reservoir.offer(&mut rng, i as i64, w).unwrap();
            }
            for e in reservoir.elements() {
                counts[e as usize] += 1;
            }
        }

        frequencies(&counts, trials)
    }

    // Naive A-Res: key u^(1/w) for every item, keeping the `k` largest.
    fn naive_inclusion_frequencies(k: usize, weights: &[f64], trials: u64) -> Vec<f64> {
        let mut rng = Rng::new(1);
        let mut counts = vec![0u64; weights.len()];
        for _ in 0..trials {
            let mut keyed: Vec<(f64, usize)> = weights
                .iter()
                .enumerate()
                .map(|(i, &w)| ((1.0 - rng.next_f64()).powf(1.0 / w), i))
                .collect();
            keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, i) in &keyed[..k] {
                counts[i] += 1;
            }
        }

        frequencies(&counts, trials)
    }

    fn state_bytes(reservoir: &WeightedReservoir) -> Vec<u8> {
        bincode::serde::encode_to_vec(reservoir, bincode::config::standard()).unwrap()
    }

    #[test]
    fn capacity_bounds_the_reservoir_and_positive_offers_fill_it() {
        let mut rng = Rng::new(0);
        let mut reservoir = WeightedReservoir::new(3);
        assert!(reservoir.is_empty());
        assert_eq!(
            reservoir.offer(&mut rng, 0, 0.0),
            Ok(Offer {
                admitted: false,
                evicted: None
            })
        );
        for i in 1..=3 {
            assert_eq!(
                reservoir.offer(&mut rng, i, 0.5),
                Ok(Offer {
                    admitted: true,
                    evicted: None
                })
            );
        }
        assert_eq!(reservoir.len(), 3);

        for i in 4..1000 {
            reservoir.offer(&mut rng, i, 1.0 + (i % 5) as f64).unwrap();
            assert!(reservoir.len() <= 3);
        }
        assert_eq!(reservoir.len(), 3);
    }

    #[test]
    fn zero_capacity_admits_nothing() {
        let mut rng = Rng::new(0);
        let mut reservoir = WeightedReservoir::new(0);

        for i in 0..10 {
            assert!(!reservoir.offer(&mut rng, i, 1.0).unwrap().admitted);
        }
        assert!(reservoir.is_empty());
    }

    #[test]
    fn zero_weights_are_never_admitted_and_draw_no_randomness() {
        let mut rng = Rng::new(5);
        let mut reservoir = WeightedReservoir::new(2);
        reservoir.offer(&mut rng, 0, 1.0).unwrap();
        reservoir.offer(&mut rng, 1, 1.0).unwrap();
        let skip = reservoir.skip;
        let mut expected = rng.clone();

        for i in 2..100 {
            assert!(!reservoir.offer(&mut rng, i, 0.0).unwrap().admitted);
        }

        assert_eq!(reservoir.skip, skip);
        assert_eq!(reservoir.elements().len(), 2);
        assert_eq!(rng.next_f64(), expected.next_f64());
    }

    #[test]
    fn evicted_elements_were_members_and_are_no_longer() {
        let mut rng = Rng::new(2);
        let mut reservoir = WeightedReservoir::new(4);

        for i in 0..500 {
            let before = reservoir.elements();
            let offer = reservoir.offer(&mut rng, i, 1.0 + (i % 7) as f64).unwrap();
            if let Some(evicted) = offer.evicted {
                assert!(before.contains(&evicted));
                assert!(!reservoir.elements().contains(&evicted));
            }
        }
    }

    #[test]
    fn single_slot_inclusion_is_proportional_to_weight() {
        let weights = [1.0, 2.0, 3.0, 4.0];
        let total: f64 = weights.iter().sum();

        let freq = inclusion_frequencies(1, &weights, 100_000);

        for (f, w) in freq.iter().zip(weights) {
            assert!(
                (f - w / total).abs() < 0.01,
                "got {f}, expected {}",
                w / total
            );
        }
    }

    #[test]
    fn inclusion_matches_naive_a_res() {
        let weights: Vec<f64> = (1..=10).map(f64::from).collect();

        let actual = inclusion_frequencies(3, &weights, 50_000);
        let expected = naive_inclusion_frequencies(3, &weights, 50_000);

        for (k, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - e).abs() < 0.02, "element {k}: got {a}, naive {e}");
        }
    }

    #[test]
    fn equal_weights_include_each_element_equally() {
        let freq = inclusion_frequencies(3, &[1.0; 10], 50_000);

        for (k, f) in freq.iter().enumerate() {
            assert!((f - 0.3).abs() < 0.02, "element {k}: got {f}");
        }
    }

    #[test]
    fn invalid_offers_leave_state_unchanged() {
        let mut rng = Rng::new(0);
        let mut reservoir = WeightedReservoir::new(2);
        reservoir.offer(&mut rng, 0, 1.0).unwrap();
        reservoir.offer(&mut rng, 1, 1.0).unwrap();
        let before = state_bytes(&reservoir);
        let mut admitted = [false; 2];
        let mut evicted = Vec::new();

        assert!(matches!(
            reservoir.offer(&mut rng, 2, f64::NAN),
            Err(TreeError::NonFiniteWeight { idx: 2, .. })
        ));
        assert!(matches!(
            reservoir.offer(&mut rng, 2, f64::INFINITY),
            Err(TreeError::NonFiniteWeight { idx: 2, .. })
        ));
        assert!(matches!(
            reservoir.offer(&mut rng, 2, -1.0),
            Err(TreeError::NegativeWeight { idx: 2, .. })
        ));
        assert!(matches!(
            reservoir.offer_batch(&mut rng, &[5, 6], &[1.0, -1.0], &mut admitted, &mut evicted),
            Err(TreeError::NegativeWeight { idx: 6, .. })
        ));
        assert_eq!(
            reservoir.offer_batch(&mut rng, &[5, 6], &[1.0], &mut admitted, &mut evicted),
            Err(TreeError::LengthMismatch { idxs: 2, values: 1 })
        );
        assert_eq!(
            reservoir.offer_batch(
                &mut rng,
                &[5, 6],
                &[1.0, 1.0],
                &mut admitted[..1],
                &mut evicted
            ),
            Err(TreeError::LengthMismatch { idxs: 2, values: 1 })
        );

        assert_eq!(state_bytes(&reservoir), before);
    }

    #[test]
    fn offer_batch_matches_sequential_offers() {
        let elements: Vec<i64> = (0..50).collect();
        let weights: Vec<f64> = (0..50).map(|i| f64::from(i % 4)).collect();
        let mut batch = WeightedReservoir::new(5);
        let mut admitted = vec![false; 50];
        let mut evicted = Vec::new();
        batch
            .offer_batch(
                &mut Rng::new(3),
                &elements,
                &weights,
                &mut admitted,
                &mut evicted,
            )
            .unwrap();

        let mut sequential = WeightedReservoir::new(5);
        let mut rng = Rng::new(3);
        let mut expected_evicted = Vec::new();
        for (i, (&e, &w)) in iter::zip(&elements, &weights).enumerate() {
            let offer = sequential.offer(&mut rng, e, w).unwrap();
            assert_eq!(admitted[i], offer.admitted, "offer {i}");
            expected_evicted.extend(offer.evicted);
        }

        assert_eq!(evicted, expected_evicted);
        assert_eq!(state_bytes(&batch), state_bytes(&sequential));
    }

    #[test]
    fn serde_round_trip_preserves_future_offers() {
        let config = bincode::config::standard();
        let mut rng = Rng::new(8);
        let mut reservoir = WeightedReservoir::new(4);
        for i in 0..100 {
            reservoir.offer(&mut rng, i, 1.0 + (i % 3) as f64).unwrap();
        }

        let bytes = bincode::serde::encode_to_vec(&reservoir, config).unwrap();
        let (mut restored, _): (WeightedReservoir, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        let mut a = rng.clone();
        let mut b = rng;
        for i in 100..300 {
            let w = 1.0 + (i % 3) as f64;
            assert_eq!(
                restored.offer(&mut a, i, w),
                reservoir.offer(&mut b, i, w),
                "offer {i}"
            );
        }

        let mut restored_elements = restored.elements();
        let mut elements = reservoir.elements();
        restored_elements.sort_unstable();
        elements.sort_unstable();
        assert_eq!(restored_elements, elements);
    }
}
