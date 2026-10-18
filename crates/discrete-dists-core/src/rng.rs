use rand_core::{Rng as _, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rng(Xoshiro256PlusPlus);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(Xoshiro256PlusPlus::seed_from_u64(seed))
    }

    /// Uniform in [0, 1) from the top 53 bits, so every value is exactly representable.
    pub fn next_f64(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0, n) via Lemire's multiply-shift; rejection removes the modulo bias.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below requires n > 0");

        let mut m = u128::from(self.0.next_u64()) * u128::from(n);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u128::from(self.0.next_u64()) * u128::from(n);
                low = m as u64;
            }
        }

        (m >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(rng: &mut Rng, n: usize) -> Vec<u64> {
        (0..n).map(|_| rng.below(u64::MAX)).collect()
    }

    #[test]
    fn same_seed_gives_same_sequence() {
        let a = draws(&mut Rng::new(42), 1000);
        let b = draws(&mut Rng::new(42), 1000);

        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        let a = draws(&mut Rng::new(1), 16);
        let b = draws(&mut Rng::new(2), 16);

        assert_ne!(a, b);
    }

    #[test]
    fn next_f64_stays_in_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..100_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "got {x}");
        }
    }

    #[test]
    fn below_stays_under_bound_and_is_roughly_uniform() {
        let mut rng = Rng::new(11);
        let mut counts = [0u32; 10];
        for _ in 0..100_000 {
            let k = rng.below(10);
            assert!(k < 10);
            counts[k as usize] += 1;
        }

        for (bucket, count) in counts.iter().enumerate() {
            assert!(
                count.abs_diff(10_000) <= 500,
                "bucket {bucket} has {count} draws"
            );
        }
    }

    #[test]
    fn serde_round_trip_continues_the_same_sequence() {
        let config = bincode::config::standard();
        let mut rng = Rng::new(99);
        draws(&mut rng, 37);

        let bytes = bincode::serde::encode_to_vec(&rng, config).unwrap();
        let (mut restored, _): (Rng, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();

        assert_eq!(draws(&mut restored, 100), draws(&mut rng, 100));
    }
}
