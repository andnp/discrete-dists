use criterion::{criterion_group, criterion_main, Criterion};
use discrete_dists_core::{Distribution, Rng, Window, WindowMixture};
use std::hint::black_box;

const SIZE: u32 = 1_000_000;
const BATCH: usize = 32;
const PRECOMPUTED: usize = 4096;

struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn index(&mut self, size: u32) -> i64 {
        (self.next_u64() % size as u64) as i64
    }

    fn positive_weight(&mut self) -> f64 {
        (self.next_u64() % 1_000_000 + 1) as f64 / 1000.0
    }
}

fn flat_windows() -> Vec<Window> {
    vec![
        Window {
            min_age: 0.0,
            max_age: 1000.0,
            half_life: None,
            p: 0.5,
        },
        Window {
            min_age: 1000.0,
            max_age: f64::INFINITY,
            half_life: None,
            p: 0.5,
        },
    ]
}

fn flat_and_decaying() -> Vec<Window> {
    vec![
        Window {
            min_age: 0.0,
            max_age: f64::INFINITY,
            half_life: None,
            p: 0.5,
        },
        Window {
            min_age: 0.0,
            max_age: 1500.0,
            half_life: Some(100.0),
            p: 0.5,
        },
    ]
}

// Ages are spread over [0, 2000) so the first 1000 steps of advancing fire window crossings.
fn prefilled(windows: Vec<Window>, rng: &mut XorShift) -> WindowMixture {
    let mut mixture = WindowMixture::new(SIZE, windows, 0.0).unwrap();
    let elements: Vec<i64> = (0..SIZE as i64).collect();
    let stamps: Vec<f64> = (0..SIZE).map(|i| -((i % 2000) as f64)).collect();
    let weights: Vec<f64> = (0..SIZE).map(|_| rng.positive_weight()).collect();
    mixture.insert(&elements, &stamps, &weights).unwrap();

    mixture
}

fn bench_insert_advance_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_insert_advance_step");
    let shapes = [
        ("flat_windows", flat_windows()),
        ("flat_and_decaying", flat_and_decaying()),
    ];
    for (name, windows) in shapes {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut mixture = prefilled(windows, &mut rng);
        let idxs: Vec<i64> = (0..PRECOMPUTED).map(|_| rng.index(SIZE)).collect();
        let weights: Vec<f64> = (0..PRECOMPUTED).map(|_| rng.positive_weight()).collect();
        let mut step = 0;

        group.bench_function(name, |b| {
            b.iter(|| {
                let start = (step * BATCH) % (PRECOMPUTED - BATCH);
                step += 1;
                let stamps = [mixture.cursor(); BATCH];
                black_box(mixture.insert(
                    &idxs[start..start + BATCH],
                    &stamps,
                    &weights[start..start + BATCH],
                ))
                .unwrap();
                black_box(mixture.advance(mixture.cursor() + 1.0)).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("window_sample");
    let shapes = [
        ("flat_windows", flat_windows()),
        ("flat_and_decaying", flat_and_decaying()),
    ];
    for (name, windows) in shapes {
        let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
        let mixture = prefilled(windows, &mut rng);
        let mut sampler = Rng::new(0);
        let mut out = [0i64; BATCH];

        group.bench_function(name, |b| {
            b.iter(|| black_box(mixture.sample(&mut sampler, &mut out)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert_advance_step, bench_sample);
criterion_main!(benches);
