use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use discrete_dists_core::Tree;
use std::hint::black_box;

const SIZES: [u32; 2] = [10_000, 1_000_000];
const BATCHES: [usize; 2] = [32, 1024];
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

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn filled_tree(size: u32, rng: &mut XorShift) -> Tree {
    let mut tree = Tree::new(size);
    let idxs: Vec<i64> = (0..size as i64).collect();
    let values: Vec<f64> = (0..size).map(|_| rng.positive_weight()).collect();
    tree.update(&idxs, &values).unwrap();
    tree
}

fn bench_update_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_single");
    for size in SIZES {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut tree = filled_tree(size, &mut rng);
        let idxs: Vec<i64> = (0..PRECOMPUTED).map(|_| rng.index(size)).collect();
        let values: Vec<f64> = (0..PRECOMPUTED).map(|_| rng.positive_weight()).collect();
        let mut cursor = 0;

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let i = cursor % PRECOMPUTED;
                cursor += 1;
                black_box(tree.update_single(idxs[i], values[i]))
            });
        });
    }
    group.finish();
}

fn bench_update_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_batch");
    for size in SIZES {
        for batch in BATCHES {
            let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
            let mut tree = filled_tree(size, &mut rng);
            let idxs: Vec<i64> = (0..batch).map(|_| rng.index(size)).collect();
            let values: Vec<f64> = (0..batch).map(|_| rng.positive_weight()).collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{size}/{batch}")),
                &batch,
                |b, _| {
                    b.iter(|| black_box(tree.update(&idxs, &values)));
                },
            );
        }
    }
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    for size in SIZES {
        for batch in BATCHES {
            let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
            let tree = filled_tree(size, &mut rng);
            let total = tree.total();
            let values: Vec<f64> = (0..batch).map(|_| rng.unit() * total).collect();
            let mut out = vec![0i64; batch];

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{size}/{batch}")),
                &batch,
                |b, _| {
                    b.iter(|| {
                        tree.query(&values, &mut out);
                        black_box(&out);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_update_single,
    bench_update_batch,
    bench_query
);
criterion_main!(benches);
