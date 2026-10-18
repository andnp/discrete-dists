use crate::{Distribution, Rng, TreeError};
use std::collections::HashSet;
use std::iter;

// Defunct parts are dropped and the rest renormalized; empty when no live mass remains.
fn renormalized<'a>(parts: &[(&'a dyn Distribution, f64)]) -> Vec<(&'a dyn Distribution, f64)> {
    let live: Vec<_> = parts
        .iter()
        .filter(|(d, _)| !d.is_defunct())
        .copied()
        .collect();
    let total: f64 = live.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 {
        return Vec::new();
    }

    live.into_iter().map(|(d, w)| (d, w / total)).collect()
}

// Assigns each of `n` slots to a part by its (normalized) weight.
fn draw_counts(parts: &[(&dyn Distribution, f64)], rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut counts = vec![0; parts.len()];
    for _ in 0..n {
        let u = rng.next_f64();
        let mut acc = 0.0;
        let mut k = parts.len() - 1;
        for (i, (_, w)) in parts.iter().enumerate() {
            acc += w;
            if u < acc {
                k = i;
                break;
            }
        }
        counts[k] += 1;
    }

    counts
}

fn fill_by_part(
    parts: &[(&dyn Distribution, f64)],
    counts: &[usize],
    rng: &mut Rng,
    mut out: &mut [i64],
    draw: fn(&dyn Distribution, &mut Rng, &mut [i64]) -> Result<(), TreeError>,
) -> Result<(), TreeError> {
    for ((d, _), &count) in iter::zip(parts, counts) {
        let (segment, rest) = out.split_at_mut(count);
        draw(*d, rng, segment)?;
        out = rest;
    }

    Ok(())
}

fn shuffle(rng: &mut Rng, out: &mut [i64]) {
    for i in (1..out.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        out.swap(i, j);
    }
}

pub fn sample(
    parts: &[(&dyn Distribution, f64)],
    rng: &mut Rng,
    out: &mut [i64],
) -> Result<(), TreeError> {
    if out.is_empty() {
        return Ok(());
    }

    let parts = renormalized(parts);
    if parts.is_empty() {
        return Err(TreeError::EmptyDistribution);
    }

    let counts = draw_counts(&parts, rng, out.len());
    fill_by_part(&parts, &counts, rng, &mut *out, |d, rng, seg| {
        d.sample(rng, seg)
    })?;
    shuffle(rng, out);

    Ok(())
}

pub fn stratified_sample(
    parts: &[(&dyn Distribution, f64)],
    rng: &mut Rng,
    out: &mut [i64],
) -> Result<(), TreeError> {
    if out.is_empty() {
        return Ok(());
    }

    let parts = renormalized(parts);
    if parts.is_empty() {
        return Err(TreeError::EmptyDistribution);
    }

    let counts = draw_counts(&parts, rng, out.len());
    fill_by_part(&parts, &counts, rng, &mut *out, |d, rng, seg| {
        d.stratified_sample(rng, seg)
    })?;
    shuffle(rng, out);

    Ok(())
}

pub fn probs(parts: &[(&dyn Distribution, f64)], elements: &[i64], out: &mut [f64]) {
    assert_eq!(elements.len(), out.len());
    out.fill(0.0);
    let mut part_probs = vec![0.0; elements.len()];
    for (d, w) in renormalized(parts) {
        d.probs(elements, &mut part_probs);
        for (o, p) in iter::zip(out.iter_mut(), &part_probs) {
            *o += w * p;
        }
    }
}

pub fn is_defunct(parts: &[(&dyn Distribution, f64)]) -> bool {
    parts.iter().all(|(d, _)| d.is_defunct())
}

/// Rejection sampling: keeps first appearances in draw order and gives up after a fixed draw budget.
pub fn sample_without_replacement(
    parts: &[(&dyn Distribution, f64)],
    rng: &mut Rng,
    n: usize,
) -> Result<Vec<i64>, TreeError> {
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
        sample(parts, rng, &mut draws)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Proportional, Uniform};

    fn frequencies(out: &[i64], size: usize) -> Vec<f64> {
        let mut counts = vec![0usize; size];
        for &x in out {
            counts[x as usize] += 1;
        }
        counts
            .into_iter()
            .map(|c| c as f64 / out.len() as f64)
            .collect()
    }

    #[test]
    fn probs_sum_to_one_over_live_support() {
        let uniform = Uniform::new(0, 10);
        let mut proportional = Proportional::new(5, 15);
        proportional.update(&[5, 7, 14], &[1.0, 2.0, 3.0]).unwrap();
        let parts: [(&dyn Distribution, f64); 2] = [(&uniform, 0.4), (&proportional, 0.6)];
        let elements: Vec<i64> = (-2..18).collect();
        let mut out = vec![0.0; elements.len()];

        probs(&parts, &elements, &mut out);

        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn defunct_parts_are_excluded_and_renormalized() {
        let live = Uniform::new(0, 4);
        let defunct = Uniform::new(10, 10);
        let parts: [(&dyn Distribution, f64); 2] = [(&live, 0.2), (&defunct, 0.8)];
        let mut out = [0.0; 5];

        probs(&parts, &[0, 1, 2, 3, 10], &mut out);

        assert_eq!(out, [0.25, 0.25, 0.25, 0.25, 0.0]);
        let mut samples = [0i64; 200];
        sample(&parts, &mut Rng::new(4), &mut samples).unwrap();
        assert!(samples.iter().all(|x| (0..4).contains(x)));
    }

    #[test]
    fn all_defunct_parts_error_on_sampling_and_have_zero_probs() {
        let a = Uniform::new(3, 3);
        let b = Proportional::new(0, 5);
        let parts: [(&dyn Distribution, f64); 2] = [(&a, 0.5), (&b, 0.5)];
        let mut out = [-7i64; 3];
        let mut probabilities = [1.0; 2];

        assert!(is_defunct(&parts));
        assert_eq!(
            sample(&parts, &mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(
            stratified_sample(&parts, &mut Rng::new(1), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(out, [-7; 3]);
        probs(&parts, &[3, 4], &mut probabilities);
        assert_eq!(probabilities, [0.0, 0.0]);
    }

    #[test]
    fn empty_mixture_is_defunct() {
        assert!(is_defunct(&[]));
    }

    #[test]
    fn zero_sample_request_succeeds_even_when_all_parts_are_defunct() {
        let a = Uniform::new(3, 3);
        let b = Proportional::new(0, 5);
        let parts: [(&dyn Distribution, f64); 2] = [(&a, 0.5), (&b, 0.5)];
        let mut out: [i64; 0] = [];

        assert_eq!(sample(&parts, &mut Rng::new(1), &mut out), Ok(()));
        assert_eq!(
            stratified_sample(&parts, &mut Rng::new(1), &mut out),
            Ok(())
        );
    }

    #[test]
    fn sample_frequencies_match_probs() {
        let uniform = Uniform::new(0, 10);
        let mut proportional = Proportional::new(0, 10);
        proportional
            .update(
                &(0..10).collect::<Vec<_>>(),
                &(1..=10).map(f64::from).collect::<Vec<_>>(),
            )
            .unwrap();
        let parts: [(&dyn Distribution, f64); 2] = [(&uniform, 0.5), (&proportional, 0.5)];
        let elements: Vec<i64> = (0..10).collect();
        let mut expected = [0.0; 10];
        probs(&parts, &elements, &mut expected);
        let n = 200_000;
        let mut out = vec![0i64; n];

        sample(&parts, &mut Rng::new(8), &mut out).unwrap();

        for (k, (freq, p)) in frequencies(&out, 10).iter().zip(expected).enumerate() {
            assert!(
                (freq - p).abs() < 0.02,
                "element {k}: got {freq}, expected {p}"
            );
        }
    }

    #[test]
    fn stratified_sample_of_one_uniform_covers_each_element_once() {
        let uniform = Uniform::new(0, 4);
        let parts: [(&dyn Distribution, f64); 1] = [(&uniform, 1.0)];
        let mut out = [0i64; 4];

        stratified_sample(&parts, &mut Rng::new(6), &mut out).unwrap();

        out.sort_unstable();
        assert_eq!(out, [0, 1, 2, 3]);
    }

    #[test]
    fn sample_without_replacement_returns_unique_elements_in_support() {
        let uniform = Uniform::new(0, 50);
        let mut proportional = Proportional::new(0, 50);
        proportional
            .update(&(0..25).collect::<Vec<_>>(), &[1.0; 25])
            .unwrap();
        let parts: [(&dyn Distribution, f64); 2] = [(&uniform, 0.5), (&proportional, 0.5)];

        let out = sample_without_replacement(&parts, &mut Rng::new(12), 30).unwrap();

        assert_eq!(out.len(), 30);
        assert_eq!(out.iter().collect::<HashSet<_>>().len(), 30);
        assert!(out.iter().all(|x| (0..50).contains(x)));
    }

    #[test]
    fn sample_without_replacement_fails_when_support_is_too_small() {
        let uniform = Uniform::new(0, 10);
        let parts: [(&dyn Distribution, f64); 1] = [(&uniform, 1.0)];

        assert_eq!(
            sample_without_replacement(&parts, &mut Rng::new(3), 11),
            Err(TreeError::NotEnoughElements {
                requested: 11,
                found: 10
            })
        );
    }

    #[test]
    fn sample_without_replacement_draws_most_of_a_uniform_reliably() {
        let a = Uniform::new(0, 1000);
        let b = Uniform::new(0, 1000);
        let parts: [(&dyn Distribution, f64); 2] = [(&a, 0.5), (&b, 0.5)];

        for seed in 0..50 {
            let out = sample_without_replacement(&parts, &mut Rng::new(seed), 900).unwrap();
            assert_eq!(out.len(), 900);
            assert_eq!(out.iter().collect::<HashSet<_>>().len(), 900);
        }
    }
}
