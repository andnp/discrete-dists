use criterion::{criterion_group, criterion_main, Criterion};
use discrete_dists_core::{Distribution, Rng, Tree, UniformSet};
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

// Even indices are members. Odd indices are not, so inserting then removing an odd batch restores the set.
fn half_members() -> Vec<i64> {
    (0..SIZE as i64).step_by(2).collect()
}

fn half_full_set() -> UniformSet {
    let mut set = UniformSet::new(SIZE);
    set.insert(&half_members()).unwrap();

    set
}

fn weight_one_tree() -> Tree {
    let members = half_members();
    let mut tree = Tree::new(SIZE);
    tree.update(&members, &vec![1.0; members.len()]).unwrap();

    tree
}

fn bench_insert_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_remove_batch");
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let idxs: Vec<i64> = (0..PRECOMPUTED)
        .map(|_| rng.index(SIZE / 2) * 2 + 1)
        .collect();
    let ones = [1.0; BATCH];
    let zeros = [0.0; BATCH];

    let mut set = half_full_set();
    let mut step = 0;
    group.bench_function("uniform_set", |b| {
        b.iter(|| {
            let start = (step * BATCH) % (PRECOMPUTED - BATCH);
            step += 1;
            let batch = &idxs[start..start + BATCH];
            black_box(set.insert(batch)).unwrap();
            black_box(set.remove(batch)).unwrap();
        });
    });

    let mut tree = weight_one_tree();
    let mut step = 0;
    group.bench_function("tree", |b| {
        b.iter(|| {
            let start = (step * BATCH) % (PRECOMPUTED - BATCH);
            step += 1;
            let batch = &idxs[start..start + BATCH];
            black_box(tree.update(batch, &ones)).unwrap();
            black_box(tree.update(batch, &zeros)).unwrap();
        });
    });
    group.finish();
}

fn bench_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_batch");
    let set = half_full_set();
    let mut sampler = Rng::new(0);
    let mut out = [0i64; BATCH];
    group.bench_function("uniform_set", |b| {
        b.iter(|| black_box(set.sample(&mut sampler, &mut out)).unwrap());
    });

    let tree = weight_one_tree();
    group.bench_function("tree", |b| {
        b.iter(|| black_box(tree.sample(&mut sampler, &mut out)).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_insert_remove, bench_sample);
criterion_main!(benches);
