use criterion::{criterion_group, criterion_main, Criterion};
use discrete_dists_core::{Prioritized, Rng};
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
}

fn filled() -> Prioritized {
    let mut prioritized = Prioritized::new(SIZE, 0.6).unwrap();
    let elements: Vec<i64> = (0..i64::from(SIZE)).collect();
    let priorities: Vec<f64> = (0..SIZE).map(|i| f64::from(i % 97 + 1)).collect();
    prioritized.update(&elements, &priorities).unwrap();

    prioritized
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_batch");
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let idxs: Vec<i64> = (0..PRECOMPUTED).map(|_| rng.index(SIZE)).collect();
    let priorities = [2.0; BATCH];

    let mut prioritized = filled();
    let mut step = 0;
    group.bench_function("prioritized", |b| {
        b.iter(|| {
            let start = (step * BATCH) % (PRECOMPUTED - BATCH);
            step += 1;
            let batch = &idxs[start..start + BATCH];
            black_box(prioritized.update(batch, &priorities)).unwrap();
        });
    });
    group.finish();
}

fn bench_sample_with_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_with_weights_batch");
    let prioritized = filled();
    let mut sampler = Rng::new(0);
    let mut out = [0i64; BATCH];
    let mut weights = [0.0; BATCH];
    group.bench_function("prioritized", |b| {
        b.iter(|| {
            black_box(prioritized.sample_with_weights(
                &mut sampler,
                0.5,
                false,
                &mut out,
                &mut weights,
            ))
            .unwrap()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_update, bench_sample_with_weights);
criterion_main!(benches);
