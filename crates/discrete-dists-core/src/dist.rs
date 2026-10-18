use crate::{Rng, Tree, TreeError};
use std::borrow::{Borrow, BorrowMut};
use std::collections::HashSet;
use std::iter;

pub trait Distribution {
    fn is_defunct(&self) -> bool;
    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError>;
    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError>;
    fn probs(&self, elements: &[i64], out: &mut [f64]);
}

fn support_width(lo: i64, hi: i64) -> Result<u32, TreeError> {
    u32::try_from(hi - lo).map_err(|_| TreeError::InvalidSupport {
        reason: format!("support [{lo}, {hi}) must have a width in 0..={}", u32::MAX),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct Uniform {
    lo: i64,
    hi: i64,
}

impl Uniform {
    pub fn new(lo: i64, hi: i64) -> Self {
        assert!(lo <= hi, "support [{lo}, {hi}) is reversed");

        Uniform { lo, hi }
    }

    pub fn support(&self) -> (i64, i64) {
        (self.lo, self.hi)
    }

    /// Widens the support to cover `elements`; an empty slice changes nothing.
    pub fn update(&mut self, elements: &[i64]) {
        if let (Some(&min), Some(&max)) = (elements.iter().min(), elements.iter().max()) {
            self.lo = self.lo.min(min);
            self.hi = self.hi.max(max + 1);
        }
    }

    pub fn update_single(&mut self, element: i64) {
        self.lo = self.lo.min(element);
        self.hi = self.hi.max(element + 1);
    }

    pub fn update_support(&mut self, lo: i64, hi: i64) {
        *self = Uniform::new(lo, hi);
    }

    /// Floyd's algorithm: exactly `n` distinct elements, each subset equally likely.
    pub fn sample_without_replacement(
        &self,
        rng: &mut Rng,
        n: usize,
        out: &mut Vec<i64>,
    ) -> Result<(), TreeError> {
        out.clear();
        let width = (self.hi - self.lo) as u64;
        if n as u64 > width {
            return Err(TreeError::NotEnoughElements {
                requested: n,
                found: width as usize,
            });
        }

        let mut chosen = HashSet::with_capacity(n);
        for j in (width - n as u64)..width {
            let t = rng.below(j + 1);
            let pick = if chosen.contains(&t) { j } else { t };
            chosen.insert(pick);
            out.push(self.lo + pick as i64);
        }

        Ok(())
    }

    fn sampling_width(&self) -> Result<u64, TreeError> {
        if self.is_defunct() {
            return Err(TreeError::EmptyDistribution);
        }

        Ok((self.hi - self.lo) as u64)
    }
}

impl Distribution for Uniform {
    fn is_defunct(&self) -> bool {
        self.lo == self.hi
    }

    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        if out.is_empty() {
            return Ok(());
        }
        let width = self.sampling_width()?;
        for x in out {
            *x = self.lo + rng.below(width) as i64;
        }

        Ok(())
    }

    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        if out.is_empty() {
            return Ok(());
        }
        let width = self.sampling_width()?;
        let n = out.len() as f64;
        for (j, x) in out.iter_mut().enumerate() {
            let position = ((j as f64 + rng.next_f64()) / n) * width as f64;
            *x = self.lo + (position as i64).min(width as i64 - 1);
        }

        Ok(())
    }

    fn probs(&self, elements: &[i64], out: &mut [f64]) {
        assert_eq!(elements.len(), out.len());
        let mass = 1.0 / (self.hi - self.lo) as f64;
        for (&element, p) in iter::zip(elements, out) {
            *p = if (self.lo..self.hi).contains(&element) {
                mass
            } else {
                0.0
            };
        }
    }
}

#[derive(Clone)]
pub struct Proportional<T: Borrow<Tree> = Tree> {
    lo: i64,
    tree: T,
}

impl Proportional {
    pub fn new(lo: i64, hi: i64) -> Self {
        let width = support_width(lo, hi).expect("invalid proportional support");

        Proportional {
            lo,
            tree: Tree::new(width),
        }
    }
}

impl<T: Borrow<Tree>> Proportional<T> {
    pub fn from_tree(lo: i64, tree: T) -> Self {
        Proportional { lo, tree }
    }

    pub fn support(&self) -> (i64, i64) {
        (self.lo, self.lo + i64::from(self.tree().size()))
    }

    pub fn tree(&self) -> &Tree {
        self.tree.borrow()
    }

    fn checked_offset(&self, element: i64) -> Result<i64, TreeError> {
        let (lo, hi) = self.support();
        if (lo..hi).contains(&element) {
            Ok(element - lo)
        } else {
            Err(TreeError::OutOfSupport { element, lo, hi })
        }
    }

    fn checked_offsets(&self, elements: &[i64]) -> Result<Vec<i64>, TreeError> {
        elements.iter().map(|&e| self.checked_offset(e)).collect()
    }

    pub fn get_values(&self, elements: &[i64]) -> Result<Vec<f64>, TreeError> {
        self.tree().get_values(&self.checked_offsets(elements)?)
    }

    /// Groups are aligned to tree positions relative to `lo`, not to the element values themselves.
    pub fn sample_grouped(
        &self,
        rng: &mut Rng,
        groups: usize,
        group_size: u32,
        out: &mut [i64],
    ) -> Result<(), TreeError> {
        self.tree().sample_grouped(rng, groups, group_size, out)?;
        for x in out {
            *x += self.lo;
        }

        Ok(())
    }
}

impl<T: BorrowMut<Tree>> Proportional<T> {
    pub fn tree_mut(&mut self) -> &mut Tree {
        self.tree.borrow_mut()
    }

    pub fn sample_without_replacement(
        &mut self,
        rng: &mut Rng,
        n: usize,
        out: &mut Vec<i64>,
    ) -> Result<(), TreeError> {
        self.tree_mut().sample_without_replacement(rng, n, out)?;
        for x in out.iter_mut() {
            *x += self.lo;
        }

        Ok(())
    }

    pub fn update(&mut self, elements: &[i64], values: &[f64]) -> Result<(), TreeError> {
        if elements.len() != values.len() {
            return Err(TreeError::LengthMismatch {
                idxs: elements.len(),
                values: values.len(),
            });
        }

        // Every pair is checked before any is written, so a bad batch leaves the tree unchanged.
        for (&element, &value) in iter::zip(elements, values) {
            Tree::validate_weight(self.checked_offset(element)?, value)?;
        }
        let lo = self.lo;
        let tree = self.tree_mut();
        for (&element, &value) in iter::zip(elements, values) {
            tree.set_validated_value((element - lo) as usize, value);
        }

        Ok(())
    }

    pub fn update_single(&mut self, element: i64, value: f64) -> Result<(), TreeError> {
        let offset = self.checked_offset(element)?;
        self.tree_mut().update_single(offset, value)
    }

    pub fn scale(&mut self, factor: f64) -> Result<(), TreeError> {
        self.tree_mut().scale(factor)
    }

    /// Same width shifts the existing weights; a wider support must contain the old one so absolute weights are kept.
    pub fn update_support(&mut self, lo: i64, hi: i64) -> Result<(), TreeError> {
        let new_width = support_width(lo, hi)?;
        let old_width = self.tree().size();

        if new_width < old_width {
            return Err(TreeError::InvalidSupport {
                reason: format!(
                    "cannot shrink proportional support from width {old_width} to {new_width}"
                ),
            });
        }

        if new_width == old_width {
            self.lo = lo;
            return Ok(());
        }

        let old_hi = self.lo + i64::from(old_width);
        if lo > self.lo || hi < old_hi {
            return Err(TreeError::InvalidSupport {
                reason:
                    "widened support must contain the existing support in order to preserve weights"
                        .to_string(),
            });
        }

        let old_positions: Vec<i64> = (0..i64::from(old_width)).collect();
        let values = self.tree().get_values(&old_positions)?;
        let shift = self.lo - lo;
        let new_positions: Vec<i64> = old_positions.iter().map(|p| p + shift).collect();

        let mut tree = Tree::new(new_width);
        tree.update(&new_positions, &values)?;
        *self.tree_mut() = tree;
        self.lo = lo;

        Ok(())
    }
}

impl<T: Borrow<Tree>> Distribution for Proportional<T> {
    fn is_defunct(&self) -> bool {
        self.tree().total() == 0.0
    }

    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        self.tree().sample(rng, out)?;
        for x in out {
            *x += self.lo;
        }

        Ok(())
    }

    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        self.tree().stratified_sample(rng, out)?;
        for x in out {
            *x += self.lo;
        }

        Ok(())
    }

    fn probs(&self, elements: &[i64], out: &mut [f64]) {
        assert_eq!(elements.len(), out.len());
        let tree = self.tree();
        let total = tree.total();
        for (&element, p) in iter::zip(elements, out) {
            *p = if total > 0.0 {
                tree.get_value(element - self.lo).map_or(0.0, |w| w / total)
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(out: &[i64], lo: i64, hi: i64) -> Vec<usize> {
        let mut counts = vec![0; (hi - lo) as usize];
        for &x in out {
            counts[(x - lo) as usize] += 1;
        }
        counts
    }

    #[test]
    fn uniform_update_widens_to_cover_elements() {
        let mut u = Uniform::new(2, 5);
        u.update(&[3]);
        assert_eq!(u.support(), (2, 5));

        u.update(&[0, 9]);
        assert_eq!(u.support(), (0, 10));

        u.update_single(-3);
        assert_eq!(u.support(), (-3, 10));

        u.update(&[]);
        assert_eq!(u.support(), (-3, 10));
    }

    #[test]
    fn uniform_is_defunct_only_when_empty() {
        assert!(!Uniform::new(3, 7).is_defunct());
        assert!(Uniform::new(4, 4).is_defunct());
    }

    #[test]
    fn uniform_sample_stays_in_half_open_support() {
        let u = Uniform::new(-5, 5);
        let mut out = vec![0i64; 10_000];

        u.sample(&mut Rng::new(1), &mut out).unwrap();

        assert!(out.iter().all(|x| (-5..5).contains(x)));
        assert!(counts(&out, -5, 5).iter().all(|&c| c > 800 && c < 1200));
    }

    #[test]
    fn uniform_sampling_when_defunct_errors_and_leaves_out_untouched() {
        let u = Uniform::new(4, 4);
        let mut out = [-7i64; 3];

        assert_eq!(
            u.sample(&mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(
            u.stratified_sample(&mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(out, [-7; 3]);
    }

    #[test]
    fn uniform_stratified_sample_hits_each_element_once_when_n_equals_width() {
        let u = Uniform::new(10, 14);
        let mut out = [0i64; 4];

        u.stratified_sample(&mut Rng::new(5), &mut out).unwrap();

        assert_eq!(out, [10, 11, 12, 13]);
    }

    #[test]
    fn uniform_stratified_sample_spreads_evenly_when_n_exceeds_width() {
        let u = Uniform::new(0, 5);
        let mut out = [0i64; 10];

        u.stratified_sample(&mut Rng::new(5), &mut out).unwrap();

        assert_eq!(counts(&out, 0, 5), vec![2; 5]);
    }

    #[test]
    fn uniform_probs_are_zero_outside_support_and_uniform_inside() {
        let u = Uniform::new(2, 6);
        let mut out = [0.0; 4];

        u.probs(&[1, 2, 5, 6], &mut out);

        assert_eq!(out, [0.0, 0.25, 0.25, 0.0]);
    }

    #[test]
    fn zero_draws_from_a_defunct_uniform_succeed() {
        let u = Uniform::new(3, 3);
        let mut rng = Rng::new(0);

        assert_eq!(u.sample(&mut rng, &mut []), Ok(()));
        assert_eq!(u.stratified_sample(&mut rng, &mut []), Ok(()));
    }

    #[test]
    fn uniform_probs_are_zero_when_defunct() {
        let mut out = [1.0; 2];

        Uniform::new(3, 3).probs(&[2, 3], &mut out);

        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn proportional_is_defunct_until_it_has_mass() {
        let mut p = Proportional::new(10, 15);
        assert!(p.is_defunct());
        assert_eq!(
            p.sample(&mut Rng::new(1), &mut [0; 2]),
            Err(TreeError::EmptyDistribution)
        );

        p.update_single(12, 2.0).unwrap();
        assert!(!p.is_defunct());
    }

    #[test]
    fn proportional_update_uses_absolute_elements() {
        let mut p = Proportional::new(10, 15);
        p.update(&[10, 12], &[1.0, 3.0]).unwrap();

        assert_eq!(p.get_values(&[10, 11, 12]).unwrap(), vec![1.0, 0.0, 3.0]);
        assert_eq!(p.support(), (10, 15));
    }

    #[test]
    fn proportional_rejects_elements_outside_shifted_support_by_absolute_element() {
        let mut p = Proportional::new(10, 15);
        let out_of_support = |element| TreeError::OutOfSupport {
            element,
            lo: 10,
            hi: 15,
        };

        assert_eq!(p.update(&[9], &[1.0]), Err(out_of_support(9)));
        assert_eq!(p.update(&[15], &[1.0]), Err(out_of_support(15)));
        assert_eq!(p.update_single(15, 1.0), Err(out_of_support(15)));
        assert_eq!(p.get_values(&[12, 15]), Err(out_of_support(15)));
        assert_eq!(
            out_of_support(15).to_string(),
            "element 15 is outside support [10, 15)"
        );
        assert!(p.is_defunct());
    }

    #[test]
    fn proportional_sample_draws_positive_weight_elements_in_support() {
        let mut p = Proportional::new(10, 15);
        p.update(&[11, 14], &[1.0, 1.0]).unwrap();
        let mut out = vec![0i64; 2_000];

        p.sample(&mut Rng::new(2), &mut out).unwrap();

        assert!(out.iter().all(|&x| x == 11 || x == 14));
    }

    #[test]
    fn proportional_probs_are_weight_fractions_and_zero_outside_support() {
        let mut p = Proportional::new(10, 15);
        p.update(&[10, 12], &[1.0, 3.0]).unwrap();
        let mut out = [0.0; 5];

        p.probs(&[9, 10, 12, 14, 15], &mut out);

        assert_eq!(out, [0.0, 0.25, 0.75, 0.0, 0.0]);
    }

    #[test]
    fn proportional_probs_are_zero_when_defunct() {
        let p = Proportional::new(0, 3);
        let mut out = [1.0; 2];

        p.probs(&[0, 2], &mut out);

        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn proportional_scale_leaves_probs_unchanged() {
        let mut p = Proportional::new(0, 3);
        p.update(&[0, 2], &[1.0, 3.0]).unwrap();
        let mut before = [0.0; 3];
        let mut after = [0.0; 3];
        p.probs(&[0, 1, 2], &mut before);

        p.scale(0.5).unwrap();
        p.probs(&[0, 1, 2], &mut after);

        assert_eq!(before, [0.25, 0.0, 0.75]);
        for (a, b) in after.iter().zip(before) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn update_support_with_same_width_shifts_weights() {
        let mut p = Proportional::new(0, 5);
        p.update(&[1, 3], &[1.0, 3.0]).unwrap();

        p.update_support(10, 15).unwrap();

        assert_eq!(p.support(), (10, 15));
        let mut out = [0.0; 4];
        p.probs(&[11, 13, 1, 3], &mut out);
        assert_eq!(out, [0.25, 0.75, 0.0, 0.0]);
    }

    #[test]
    fn update_support_wider_preserves_absolute_weights() {
        let mut p = Proportional::new(2, 5);
        p.update(&[2, 3, 4], &[1.0, 2.0, 3.0]).unwrap();

        p.update_support(0, 8).unwrap();

        assert_eq!(p.support(), (0, 8));
        assert_eq!(
            p.get_values(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        );
        let mut out = [0.0; 3];
        p.probs(&[2, 4, 7], &mut out);
        assert_eq!(out, [1.0 / 6.0, 0.5, 0.0]);
    }

    #[test]
    fn uniform_sample_without_replacement_of_whole_support_is_a_permutation() {
        let u = Uniform::new(5, 15);
        let mut out = Vec::new();

        for seed in 0..100 {
            u.sample_without_replacement(&mut Rng::new(seed), 10, &mut out)
                .unwrap();

            out.sort_unstable();
            assert_eq!(out, (5..15).collect::<Vec<i64>>(), "seed {seed}");
        }
    }

    #[test]
    fn uniform_sample_without_replacement_rejects_n_above_width() {
        let u = Uniform::new(0, 3);
        let mut out = Vec::new();

        assert_eq!(
            u.sample_without_replacement(&mut Rng::new(1), 4, &mut out),
            Err(TreeError::NotEnoughElements {
                requested: 4,
                found: 3
            })
        );
    }

    #[test]
    fn uniform_sample_without_replacement_includes_each_element_with_probability_n_over_width() {
        let u = Uniform::new(0, 20);
        let trials: u64 = 20_000;
        let mut counts = [0usize; 20];
        let mut out = Vec::new();

        for seed in 0..trials {
            u.sample_without_replacement(&mut Rng::new(seed), 5, &mut out)
                .unwrap();
            for &x in &out {
                counts[x as usize] += 1;
            }
        }

        for (k, count) in counts.iter().enumerate() {
            let freq = *count as f64 / trials as f64;
            assert!((freq - 0.25).abs() < 0.03, "element {k}: got {freq}");
        }
    }

    #[test]
    fn proportional_sample_without_replacement_is_offset_and_leaves_weights_unchanged() {
        let mut p = Proportional::new(10, 20);
        p.update(&[11, 15, 19], &[1.0, 2.0, 3.0]).unwrap();
        let mut out = Vec::new();

        p.sample_without_replacement(&mut Rng::new(4), 3, &mut out)
            .unwrap();

        out.sort_unstable();
        assert_eq!(out, vec![11, 15, 19]);
        assert_eq!(p.get_values(&[11, 15, 19]).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn proportional_sample_without_replacement_rejects_n_above_positive_mass() {
        let mut p = Proportional::new(0, 5);
        p.update(&[1, 2], &[1.0, 1.0]).unwrap();
        let mut out = Vec::new();

        assert_eq!(
            p.sample_without_replacement(&mut Rng::new(1), 3, &mut out),
            Err(TreeError::NotEnoughElements {
                requested: 3,
                found: 2
            })
        );
        assert_eq!(p.get_values(&[1, 2]).unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn proportional_view_samples_through_borrowed_tree() {
        let mut tree = Tree::new(5);
        tree.update(&[1, 3], &[1.0, 3.0]).unwrap();
        let view = Proportional::from_tree(10, &tree);
        let mut out = vec![0i64; 2_000];

        view.sample(&mut Rng::new(9), &mut out).unwrap();

        assert_eq!(view.support(), (10, 15));
        assert!(out.iter().all(|&x| x == 11 || x == 13));
    }

    #[test]
    fn proportional_sample_grouped_aligns_blocks_relative_to_lo() {
        let mut p = Proportional::new(5, 21);
        p.update(&(5..21).collect::<Vec<i64>>(), &[1.0; 16])
            .unwrap();
        let mut out = [0i64; 64];

        for seed in 0..50 {
            p.sample_grouped(&mut Rng::new(seed), 1, 4, &mut out)
                .unwrap();

            assert!(out.iter().all(|e| (5..21).contains(e)), "seed {seed}");
            let block = (out[0] - 5) / 4;
            assert!(
                out.iter().all(|e| (e - 5) / 4 == block),
                "seed {seed}: {out:?}"
            );
        }
    }

    #[test]
    fn update_support_rejects_shrinking() {
        let mut p = Proportional::new(10, 15);

        let err = p.update_support(10, 14).unwrap_err();

        assert!(
            matches!(&err, TreeError::InvalidSupport { reason } if reason.contains("cannot shrink"))
        );
        assert_eq!(p.support(), (10, 15));
    }

    #[test]
    fn update_support_rejects_widening_that_does_not_contain_old_support() {
        let mut p = Proportional::new(0, 4);
        p.update(&[0], &[1.0]).unwrap();

        for (lo, hi) in [(1, 7), (-2, 3)] {
            let err = p.update_support(lo, hi).unwrap_err();
            assert!(
                matches!(&err, TreeError::InvalidSupport { reason } if reason.contains("must contain")),
                "({lo}, {hi}): got {err:?}"
            );
        }
        assert_eq!(p.support(), (0, 4));
        assert_eq!(p.get_values(&[0]).unwrap(), vec![1.0]);
    }
}
