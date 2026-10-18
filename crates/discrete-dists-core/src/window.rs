use crate::{mixture, Distribution, Proportional, Rng, Tree, TreeError};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter;

const MAX_WINDOWS: usize = 64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Window {
    pub min_age: f64,
    pub max_age: f64,
    pub p: f64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WindowMixture {
    windows: Vec<Window>,
    trees: Vec<Tree>,
    boundaries: Vec<f64>,
    stamps: Vec<f64>,
    weights: Vec<f64>,
    members: Vec<u64>,
    counts: Vec<u64>,
    generations: Vec<u64>,
    crossings: BinaryHeap<Crossing>,
    cursor: f64,
}

// Min-heap on `at`: BinaryHeap is a max-heap, so the comparison is reversed.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Crossing {
    at: f64,
    element: u32,
    generation: u64,
    boundary: u32,
}

impl Ord for Crossing {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at
            .total_cmp(&self.at)
            .then_with(|| other.element.cmp(&self.element))
            .then_with(|| other.generation.cmp(&self.generation))
    }
}

impl PartialOrd for Crossing {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Crossing {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Crossing {}

fn invalid(reason: String) -> TreeError {
    TreeError::InvalidWindow { reason }
}

fn validate_windows(windows: &[Window]) -> Result<(), TreeError> {
    if !(1..=MAX_WINDOWS).contains(&windows.len()) {
        return Err(invalid(format!(
            "expected 1 to {MAX_WINDOWS} windows, got {}",
            windows.len()
        )));
    }

    for (k, w) in windows.iter().enumerate() {
        if !w.min_age.is_finite() {
            return Err(invalid(format!(
                "window {k} min_age {} is not finite",
                w.min_age
            )));
        }
        if w.max_age.is_nan() || w.max_age <= w.min_age {
            return Err(invalid(format!(
                "window {k} max_age {} must exceed min_age {}",
                w.max_age, w.min_age
            )));
        }
        if !w.p.is_finite() || w.p < 0.0 {
            return Err(invalid(format!(
                "window {k} mass {} must be finite and non-negative",
                w.p
            )));
        }
    }

    if windows.iter().map(|w| w.p).sum::<f64>() <= 0.0 {
        return Err(invalid(
            "window masses must sum to a positive value".to_string(),
        ));
    }

    Ok(())
}

fn parts<'a>(
    windows: &[Window],
    views: &'a [Proportional<&'a Tree>],
) -> Vec<(&'a dyn Distribution, f64)> {
    iter::zip(views, windows)
        .map(|(view, w)| (view as &dyn Distribution, w.p))
        .collect()
}

impl WindowMixture {
    pub fn new(size: u32, windows: Vec<Window>, cursor: f64) -> Result<Self, TreeError> {
        validate_windows(&windows)?;
        if !cursor.is_finite() {
            return Err(invalid(format!("cursor {cursor} is not finite")));
        }

        let mut boundaries: Vec<f64> = windows
            .iter()
            .flat_map(|w| [w.min_age, w.max_age])
            .filter(|b| b.is_finite())
            .collect();
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup();

        let n = size as usize;
        let k = windows.len();
        Ok(WindowMixture {
            trees: iter::repeat_with(|| Tree::new(size)).take(k).collect(),
            windows,
            boundaries,
            stamps: vec![0.0; n],
            weights: vec![0.0; n],
            members: vec![0; n],
            counts: vec![0; k],
            generations: vec![0; n],
            crossings: BinaryHeap::new(),
            cursor,
        })
    }

    pub fn cursor(&self) -> f64 {
        self.cursor
    }

    pub fn size(&self) -> u32 {
        self.trees[0].size()
    }

    pub fn windows(&self) -> &[Window] {
        &self.windows
    }

    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    fn checked_element(&self, element: i64) -> Result<usize, TreeError> {
        let size = self.size();
        if element < 0 || element >= i64::from(size) {
            return Err(TreeError::IndexOutOfBounds { idx: element, size });
        }

        Ok(element as usize)
    }

    fn views(&self) -> Vec<Proportional<&Tree>> {
        self.trees
            .iter()
            .map(|tree| Proportional::from_tree(0, tree))
            .collect()
    }

    fn mask_at(&self, age: f64) -> u64 {
        self.windows
            .iter()
            .enumerate()
            .filter(|(_, w)| w.min_age <= age && age < w.max_age)
            .fold(0, |mask, (k, _)| mask | (1u64 << k))
    }

    fn set_membership(&mut self, e: usize, mask: u64) -> Result<(), TreeError> {
        let mut changed = self.members[e] ^ mask;
        while changed != 0 {
            let k = changed.trailing_zeros() as usize;
            changed &= changed - 1;

            if mask & (1u64 << k) != 0 {
                self.trees[k].update_single(e as i64, self.weights[e])?;
                self.counts[k] += 1;
            } else {
                self.trees[k].update_single(e as i64, 0.0)?;
                self.counts[k] -= 1;
            }
        }
        self.members[e] = mask;

        Ok(())
    }

    // The generation lets stale entries be recognized on pop.
    fn schedule(&mut self, e: usize, age: f64) {
        let j = self.boundaries.partition_point(|&b| b <= age);
        if let Some(&boundary) = self.boundaries.get(j) {
            self.crossings.push(Crossing {
                at: self.stamps[e] + boundary,
                element: e as u32,
                generation: self.generations[e],
                boundary: j as u32,
            });
        }
    }

    pub fn insert(
        &mut self,
        elements: &[i64],
        stamps: &[f64],
        weights: &[f64],
    ) -> Result<(), TreeError> {
        for len in [stamps.len(), weights.len()] {
            if len != elements.len() {
                return Err(TreeError::LengthMismatch {
                    idxs: elements.len(),
                    values: len,
                });
            }
        }
        for (&element, (&stamp, &weight)) in iter::zip(elements, iter::zip(stamps, weights)) {
            self.checked_element(element)?;
            if !stamp.is_finite() {
                return Err(invalid(format!(
                    "stamp {stamp} of element {element} is not finite"
                )));
            }
            Tree::validate_weight(element, weight)?;
        }

        for (&element, (&stamp, &weight)) in iter::zip(elements, iter::zip(stamps, weights)) {
            let e = element as usize;
            self.generations[e] += 1;
            self.stamps[e] = stamp;
            self.weights[e] = weight;
            self.set_membership(e, 0)?;

            let age = self.cursor - stamp;
            let mask = self.mask_at(age);
            self.set_membership(e, mask)?;
            self.schedule(e, age);
        }

        // Reinserts leave their old entries behind; compacting keeps the heap proportional to the live elements.
        if self.crossings.len() > 2 * self.size() as usize + 64 {
            let generations = &self.generations;
            self.crossings
                .retain(|c| generations[c.element as usize] == c.generation);
        }

        Ok(())
    }

    pub fn remove(&mut self, elements: &[i64]) -> Result<(), TreeError> {
        for &element in elements {
            self.checked_element(element)?;
        }

        for &element in elements {
            let e = element as usize;
            self.generations[e] += 1;
            self.set_membership(e, 0)?;
        }

        Ok(())
    }

    pub fn advance(&mut self, to: f64) -> Result<(), TreeError> {
        if !to.is_finite() || to < self.cursor {
            return Err(invalid(format!(
                "cannot advance cursor from {} to {to}",
                self.cursor
            )));
        }
        self.cursor = to;

        while let Some(&crossing) = self.crossings.peek() {
            if crossing.at > to {
                break;
            }
            self.crossings.pop();

            let e = crossing.element as usize;
            if self.generations[e] != crossing.generation {
                continue;
            }

            // Rounding in `stamp + boundary - stamp` can leave the age just below the boundary; clamping keeps the element from re-crossing the same boundary forever.
            let age =
                (self.cursor - self.stamps[e]).max(self.boundaries[crossing.boundary as usize]);
            let mask = self.mask_at(age);
            self.set_membership(e, mask)?;
            self.schedule(e, age);
        }

        Ok(())
    }
}

impl Distribution for WindowMixture {
    fn is_defunct(&self) -> bool {
        let views = self.views();
        mixture::is_defunct(&parts(&self.windows, &views))
    }

    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        let views = self.views();
        mixture::sample(&parts(&self.windows, &views), rng, out)
    }

    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        let views = self.views();
        mixture::stratified_sample(&parts(&self.windows, &views), rng, out)
    }

    fn probs(&self, elements: &[i64], out: &mut [f64]) {
        let views = self.views();
        mixture::probs(&parts(&self.windows, &views), elements, out)
    }
}

impl WindowMixture {
    pub fn sample_without_replacement(
        &self,
        rng: &mut Rng,
        n: usize,
    ) -> Result<Vec<i64>, TreeError> {
        let views = self.views();
        mixture::sample_without_replacement(&parts(&self.windows, &views), rng, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use proptest::collection::vec;
    use proptest::prelude::*;

    fn window(min_age: f64, max_age: f64, p: f64) -> Window {
        Window {
            min_age,
            max_age,
            p,
        }
    }

    fn assert_invalid(result: Result<WindowMixture, TreeError>) {
        assert!(
            matches!(result, Err(TreeError::InvalidWindow { .. })),
            "expected InvalidWindow"
        );
    }

    fn probs_of(mixture: &WindowMixture, size: i64) -> Vec<f64> {
        let elements: Vec<i64> = (0..size).collect();
        let mut out = vec![0.0; elements.len()];
        mixture.probs(&elements, &mut out);
        out
    }

    #[test]
    fn new_rejects_window_count_outside_one_to_sixty_four() {
        let windows = |n| (0..n).map(|_| window(0.0, 1.0, 1.0)).collect::<Vec<_>>();

        assert_invalid(WindowMixture::new(4, vec![], 0.0));
        assert_invalid(WindowMixture::new(4, windows(65), 0.0));
        assert!(WindowMixture::new(4, windows(64), 0.0).is_ok());
    }

    #[test]
    fn new_rejects_invalid_age_bounds() {
        for (min_age, max_age) in [
            (f64::NAN, 1.0),
            (f64::NEG_INFINITY, 1.0),
            (0.0, f64::NAN),
            (5.0, 5.0),
            (5.0, 4.0),
        ] {
            assert_invalid(WindowMixture::new(
                4,
                vec![window(min_age, max_age, 1.0)],
                0.0,
            ));
        }
        assert!(WindowMixture::new(4, vec![window(0.0, f64::INFINITY, 1.0)], 0.0).is_ok());
    }

    #[test]
    fn new_rejects_invalid_masses() {
        assert_invalid(WindowMixture::new(4, vec![window(0.0, 1.0, f64::NAN)], 0.0));
        assert_invalid(WindowMixture::new(4, vec![window(0.0, 1.0, -1.0)], 0.0));
        assert_invalid(WindowMixture::new(
            4,
            vec![window(0.0, 1.0, f64::INFINITY)],
            0.0,
        ));
        assert_invalid(WindowMixture::new(4, vec![window(0.0, 1.0, 0.0)], 0.0));
    }

    #[test]
    fn new_rejects_non_finite_cursor() {
        assert_invalid(WindowMixture::new(4, vec![window(0.0, 1.0, 1.0)], f64::NAN));
        assert_invalid(WindowMixture::new(
            4,
            vec![window(0.0, 1.0, 1.0)],
            f64::INFINITY,
        ));
    }

    #[test]
    fn insert_rejects_bad_batch_without_changing_state() {
        let mut mixture =
            WindowMixture::new(4, vec![window(0.0, f64::INFINITY, 1.0)], 0.0).unwrap();

        assert_eq!(
            mixture.insert(&[0, 1], &[0.0], &[1.0, 1.0]),
            Err(TreeError::LengthMismatch { idxs: 2, values: 1 })
        );
        assert_eq!(
            mixture.insert(&[0, 4], &[0.0, 0.0], &[1.0, 1.0]),
            Err(TreeError::IndexOutOfBounds { idx: 4, size: 4 })
        );
        assert!(matches!(
            mixture.insert(&[0, 1], &[0.0, f64::NAN], &[1.0, 1.0]),
            Err(TreeError::InvalidWindow { .. })
        ));
        assert_eq!(
            mixture.insert(&[0, 1], &[0.0, 0.0], &[1.0, -1.0]),
            Err(TreeError::NegativeWeight {
                idx: 1,
                value: -1.0
            })
        );
        assert_eq!(mixture.counts(), &[0]);
        assert_eq!(probs_of(&mixture, 4), [0.0; 4]);
    }

    #[test]
    fn advance_rejects_backwards_and_non_finite_targets() {
        let mut mixture =
            WindowMixture::new(4, vec![window(0.0, f64::INFINITY, 1.0)], 5.0).unwrap();

        assert!(matches!(
            mixture.advance(4.0),
            Err(TreeError::InvalidWindow { .. })
        ));
        assert!(matches!(
            mixture.advance(f64::NAN),
            Err(TreeError::InvalidWindow { .. })
        ));
        assert!(matches!(
            mixture.advance(f64::INFINITY),
            Err(TreeError::InvalidWindow { .. })
        ));
        assert_eq!(mixture.advance(5.0), Ok(()));
        assert_eq!(mixture.cursor(), 5.0);
    }

    #[test]
    fn element_moves_to_next_window_at_half_open_boundary() {
        let mut mixture = WindowMixture::new(
            2,
            vec![window(0.0, 10.0, 0.5), window(10.0, f64::INFINITY, 0.5)],
            0.0,
        )
        .unwrap();
        mixture.insert(&[0], &[0.0], &[1.0]).unwrap();

        mixture.advance(9.5).unwrap();
        assert_eq!(mixture.counts(), &[1, 0]);

        mixture.advance(10.0).unwrap();
        assert_eq!(mixture.counts(), &[0, 1]);
    }

    #[test]
    fn stamp_ahead_of_cursor_joins_window_when_age_reaches_min() {
        let mut mixture = WindowMixture::new(2, vec![window(0.0, 10.0, 1.0)], 0.0).unwrap();
        mixture.insert(&[0], &[5.0], &[1.0]).unwrap();
        assert_eq!(mixture.counts(), &[0]);

        mixture.advance(4.0).unwrap();
        assert_eq!(mixture.counts(), &[0]);

        mixture.advance(5.0).unwrap();
        assert_eq!(mixture.counts(), &[1]);

        mixture.advance(15.0).unwrap();
        assert_eq!(mixture.counts(), &[0]);
    }

    #[test]
    fn reinsert_invalidates_old_crossing() {
        let mut mixture = WindowMixture::new(1, vec![window(0.0, 10.0, 1.0)], 0.0).unwrap();
        mixture.insert(&[0], &[0.0], &[1.0]).unwrap();

        mixture.advance(6.0).unwrap();
        mixture.insert(&[0], &[5.0], &[1.0]).unwrap();
        mixture.advance(12.0).unwrap();

        assert_eq!(mixture.counts(), &[1]);
    }

    #[test]
    fn remove_drops_element_from_every_window_and_stops_its_crossings() {
        let mut mixture = WindowMixture::new(
            2,
            vec![window(0.0, 10.0, 0.5), window(0.0, f64::INFINITY, 0.5)],
            0.0,
        )
        .unwrap();
        mixture.insert(&[0, 1], &[0.0, 0.0], &[1.0, 1.0]).unwrap();

        mixture.remove(&[0]).unwrap();
        assert_eq!(mixture.counts(), &[1, 1]);

        mixture.advance(20.0).unwrap();
        assert_eq!(mixture.counts(), &[0, 1]);
        assert_eq!(probs_of(&mixture, 2), [0.0, 1.0]);
    }

    #[test]
    fn probs_follow_window_shares_and_empty_windows_hand_their_share_over() {
        let mut mixture = WindowMixture::new(
            4,
            vec![window(0.0, 10.0, 0.75), window(10.0, f64::INFINITY, 0.25)],
            0.0,
        )
        .unwrap();
        mixture.insert(&[0], &[0.0], &[1.0]).unwrap();
        mixture.insert(&[1, 2, 3], &[-10.0; 3], &[1.0; 3]).unwrap();

        let probs = probs_of(&mixture, 4);
        assert!((probs[0] - 0.75).abs() < 1e-12);
        for p in &probs[1..] {
            assert!((p - 0.25 / 3.0).abs() < 1e-12);
        }

        mixture.remove(&[0]).unwrap();
        let probs = probs_of(&mixture, 4);
        assert_eq!(probs[0], 0.0);
        for p in &probs[1..] {
            assert!((p - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn sampling_an_all_empty_mixture_fails_but_zero_draws_succeed() {
        let mixture = WindowMixture::new(4, vec![window(0.0, 10.0, 1.0)], 0.0).unwrap();
        let mut out = [0i64; 3];

        assert!(mixture.is_defunct());
        assert_eq!(
            mixture.sample(&mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(
            mixture.sample_without_replacement(&mut Rng::new(1), 2),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(mixture.sample(&mut Rng::new(1), &mut []), Ok(()));
        assert_eq!(
            mixture.sample_without_replacement(&mut Rng::new(1), 0),
            Ok(vec![])
        );
    }

    #[test]
    fn serde_round_trip_preserves_probs_and_future_advances() {
        let config = bincode::config::standard();
        let mut mixture = WindowMixture::new(
            4,
            vec![window(0.0, 10.0, 0.5), window(10.0, f64::INFINITY, 0.5)],
            0.0,
        )
        .unwrap();
        mixture
            .insert(&[0, 1, 2], &[0.0, 3.0, -2.0], &[1.0, 2.0, 0.5])
            .unwrap();

        let bytes = bincode::serde::encode_to_vec(&mixture, config).unwrap();
        let (mut restored, _): (WindowMixture, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();
        assert_eq!(probs_of(&restored, 4), probs_of(&mixture, 4));

        mixture.advance(12.0).unwrap();
        restored.advance(12.0).unwrap();
        assert_eq!(restored.counts(), mixture.counts());
        assert_eq!(probs_of(&restored, 4), probs_of(&mixture, 4));
    }

    #[test]
    fn reinserts_keep_crossing_heap_bounded() {
        let size = 4;
        let mut mixture = WindowMixture::new(
            size,
            vec![window(0.0, 10.0, 0.5), window(10.0, f64::INFINITY, 0.5)],
            0.0,
        )
        .unwrap();

        for i in 0..1000i64 {
            mixture.insert(&[i % 4], &[0.0], &[1.0]).unwrap();
            assert!(mixture.crossings.len() <= 2 * size as usize + 64 + 1);
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(i64, f64, f64),
        Remove(i64),
        Advance(f64),
    }

    fn quarters(lo: i32, hi: i32) -> impl Strategy<Value = f64> {
        (lo..=hi).prop_map(|q| f64::from(q) * 0.25)
    }

    fn window_spec() -> impl Strategy<Value = Window> {
        (
            quarters(-8, 32),
            prop_oneof![3 => quarters(1, 32), 1 => Just(f64::INFINITY)],
            quarters(0, 4),
        )
            .prop_map(|(min_age, width, p)| window(min_age, min_age + width, p))
    }

    fn case() -> impl Strategy<Value = (u32, Vec<Window>, Vec<Op>)> {
        (1u32..=8).prop_flat_map(|size| {
            let op = prop_oneof![
                (0..i64::from(size), quarters(-40, 40), quarters(0, 8))
                    .prop_map(|(e, s, w)| Op::Insert(e, s, w)),
                (0..i64::from(size)).prop_map(Op::Remove),
                quarters(0, 40).prop_map(Op::Advance),
            ];
            let windows = vec(window_spec(), 1..=4)
                .prop_filter("needs positive mass", |ws| ws.iter().any(|w| w.p > 0.0));
            (Just(size), windows, vec(op, 0..=60))
        })
    }

    // Brute-force reference: recomputes every window's members and mass from scratch.
    struct Model {
        windows: Vec<Window>,
        cursor: f64,
        live: Vec<Option<(f64, f64)>>,
    }

    impl Model {
        fn in_window(&self, k: usize, stamp: f64) -> bool {
            let age = self.cursor - stamp;
            self.windows[k].min_age <= age && age < self.windows[k].max_age
        }

        fn members(&self, k: usize) -> impl Iterator<Item = (f64, f64)> + '_ {
            self.live
                .iter()
                .flatten()
                .copied()
                .filter(move |&(stamp, _)| self.in_window(k, stamp))
        }

        fn counts(&self) -> Vec<u64> {
            (0..self.windows.len())
                .map(|k| self.members(k).count() as u64)
                .collect()
        }

        fn probs(&self) -> Vec<f64> {
            let k_total = self.windows.len();
            let mass: Vec<f64> = (0..k_total)
                .map(|k| self.members(k).map(|(_, w)| w).sum())
                .collect();
            let live_p: f64 = (0..k_total)
                .filter(|&k| mass[k] > 0.0)
                .map(|k| self.windows[k].p)
                .sum();

            self.live
                .iter()
                .map(|slot| match *slot {
                    Some((stamp, w)) if live_p > 0.0 => (0..k_total)
                        .filter(|&k| mass[k] > 0.0 && self.in_window(k, stamp))
                        .map(|k| self.windows[k].p / live_p * w / mass[k])
                        .sum(),
                    _ => 0.0,
                })
                .collect()
        }
    }

    proptest! {
        #[test]
        fn probs_and_counts_match_naive_model((size, windows, ops) in case()) {
            let mut mixture = WindowMixture::new(size, windows.clone(), 0.0).unwrap();
            let mut model = Model {
                windows,
                cursor: 0.0,
                live: vec![None; size as usize],
            };
            let elements: Vec<i64> = (0..i64::from(size)).collect();

            for op in ops {
                match op {
                    Op::Insert(e, stamp, w) => {
                        mixture.insert(&[e], &[stamp], &[w]).unwrap();
                        model.live[e as usize] = Some((stamp, w));
                    }
                    Op::Remove(e) => {
                        mixture.remove(&[e]).unwrap();
                        model.live[e as usize] = None;
                    }
                    Op::Advance(dt) => {
                        model.cursor += dt;
                        mixture.advance(model.cursor).unwrap();
                    }
                }

                let mut actual = vec![0.0; elements.len()];
                mixture.probs(&elements, &mut actual);
                let expected = model.probs();
                for (k, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    prop_assert!((a - e).abs() <= 1e-12, "element {k}: got {a}, expected {e}");
                }
                let expected_counts = model.counts();
                prop_assert_eq!(mixture.counts(), expected_counts.as_slice());
            }
        }
    }
}
