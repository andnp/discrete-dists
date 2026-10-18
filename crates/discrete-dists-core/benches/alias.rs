use criterion::{criterion_group, criterion_main, Criterion};
use discrete_dists_core::{Alias, Distribution, Rng, Tree};
use std::hint::black_box;

const SIZE: u32 = 1_000_000;
const SMALL: usize = 32;
const LARGE: usize = 1024;

fn weights() -> Vec<f64> {
    (0..SIZE).map(|i| f64::from(1 + i % 7)).collect()
}

fn filled_tree(weights: &[f64]) -> Tree {
    let idxs: Vec<i64> = (0..i64::from(SIZE)).collect();
    let mut tree = Tree::new(SIZE);
    tree.update(&idxs, weights).unwrap();

    tree
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    let weights = weights();
    group.bench_function("alias", |b| {
        b.iter(|| black_box(Alias::new(0, &weights)).unwrap());
    });
    group.finish();
}

fn bench_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_batch");
    let weights = weights();
    let alias = Alias::new(0, &weights).unwrap();
    let tree = filled_tree(&weights);
    let mut sampler = Rng::new(0);
    for n in [SMALL, LARGE] {
        let mut out = vec![0i64; n];
        group.bench_function(format!("alias/{n}"), |b| {
            b.iter(|| black_box(alias.sample(&mut sampler, &mut out)).unwrap());
        });

        let mut out = vec![0i64; n];
        group.bench_function(format!("tree/{n}"), |b| {
            b.iter(|| black_box(tree.sample(&mut sampler, &mut out)).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build, bench_sample);
criterion_main!(benches);
