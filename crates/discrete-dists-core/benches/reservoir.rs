use criterion::{criterion_group, criterion_main, Criterion};
use discrete_dists_core::{Rng, WeightedReservoir};
use std::hint::black_box;

const CAPACITY: u32 = 10_000;
const BATCH: usize = 1024;

fn bench_offer(c: &mut Criterion) {
    let mut group = c.benchmark_group("offer");
    let mut rng = Rng::new(0);
    let mut reservoir = WeightedReservoir::new(CAPACITY);
    for i in 0..CAPACITY {
        reservoir.offer(&mut rng, i64::from(i), 1.0).unwrap();
    }

    let elements: Vec<i64> = (0..BATCH as i64).collect();
    let weights: Vec<f64> = (0..BATCH).map(|i| f64::from(1 + i as u32 % 7)).collect();
    let mut admitted = vec![false; BATCH];
    let mut evicted = Vec::new();
    group.bench_function(format!("batch/{BATCH}"), |b| {
        b.iter(|| {
            black_box(reservoir.offer_batch(
                &mut rng,
                &elements,
                &weights,
                &mut admitted,
                &mut evicted,
            ))
            .unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_offer);
criterion_main!(benches);
