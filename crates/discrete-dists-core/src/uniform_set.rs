use crate::{Distribution, Rng, TreeError, Uniform};
use serde::{Deserialize, Serialize};
use std::iter;

const ABSENT: u32 = u32::MAX;

// `positions[e]` is the index of `e` in `members`, or `ABSENT` when `e` is not a member.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct UniformSet {
    members: Vec<u32>,
    positions: Vec<u32>,
}

impl UniformSet {
    pub fn new(size: u32) -> Self {
        UniformSet {
            members: Vec::new(),
            positions: vec![ABSENT; size as usize],
        }
    }

    pub fn size(&self) -> u32 {
        self.positions.len() as u32
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn contains(&self, element: i64) -> bool {
        self.index(element)
            .is_some_and(|e| self.positions[e] != ABSENT)
    }

    pub fn insert(&mut self, elements: &[i64]) -> Result<(), TreeError> {
        let elements = self.checked(elements)?;
        for e in elements {
            if self.positions[e] == ABSENT {
                self.positions[e] = self.members.len() as u32;
                self.members.push(e as u32);
            }
        }

        Ok(())
    }

    pub fn remove(&mut self, elements: &[i64]) -> Result<(), TreeError> {
        let elements = self.checked(elements)?;
        for e in elements {
            let pos = self.positions[e];
            if pos == ABSENT {
                continue;
            }

            self.members.swap_remove(pos as usize);
            if let Some(&moved) = self.members.get(pos as usize) {
                self.positions[moved as usize] = pos;
            }
            self.positions[e] = ABSENT;
        }

        Ok(())
    }

    /// Draws `n` distinct members, each subset equally likely.
    pub fn sample_without_replacement(
        &self,
        rng: &mut Rng,
        n: usize,
    ) -> Result<Vec<i64>, TreeError> {
        let mut positions = Vec::new();
        Uniform::new(0, self.len() as i64).sample_without_replacement(rng, n, &mut positions)?;

        Ok(positions
            .into_iter()
            .map(|p| i64::from(self.members[p as usize]))
            .collect())
    }

    fn index(&self, element: i64) -> Option<usize> {
        usize::try_from(element)
            .ok()
            .filter(|&e| e < self.positions.len())
    }

    // Validates every element before any is applied, so a bad batch leaves the set unchanged.
    fn checked(&self, elements: &[i64]) -> Result<Vec<usize>, TreeError> {
        elements
            .iter()
            .map(|&element| {
                self.index(element).ok_or(TreeError::IndexOutOfBounds {
                    idx: element,
                    size: self.size(),
                })
            })
            .collect()
    }

    fn sampling_members(&self) -> Result<&[u32], TreeError> {
        if self.is_empty() {
            return Err(TreeError::EmptyDistribution);
        }

        Ok(&self.members)
    }
}

impl Distribution for UniformSet {
    fn is_defunct(&self) -> bool {
        self.is_empty()
    }

    fn sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        if out.is_empty() {
            return Ok(());
        }
        let members = self.sampling_members()?;
        for x in out {
            *x = i64::from(members[rng.below(members.len() as u64) as usize]);
        }

        Ok(())
    }

    fn stratified_sample(&self, rng: &mut Rng, out: &mut [i64]) -> Result<(), TreeError> {
        if out.is_empty() {
            return Ok(());
        }
        let members = self.sampling_members()?;
        let width = members.len();
        let n = out.len() as f64;
        for (j, x) in out.iter_mut().enumerate() {
            let position = ((j as f64 + rng.next_f64()) / n) * width as f64;
            *x = i64::from(members[(position as usize).min(width - 1)]);
        }

        Ok(())
    }

    fn probs(&self, elements: &[i64], out: &mut [f64]) {
        assert_eq!(elements.len(), out.len());
        for (&element, p) in iter::zip(elements, out) {
            *p = if self.contains(element) {
                1.0 / self.len() as f64
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use std::collections::{HashMap, HashSet};

    const SIZE: u32 = 16;

    fn set_of(elements: &[i64]) -> UniformSet {
        let mut set = UniformSet::new(SIZE);
        set.insert(elements).unwrap();
        set
    }

    fn sorted_members(set: &UniformSet) -> Vec<i64> {
        let mut members: Vec<i64> = set.members.iter().map(|&m| i64::from(m)).collect();
        members.sort_unstable();
        members
    }

    #[test]
    fn insert_remove_contains_and_len_track_membership() {
        let mut set = UniformSet::new(10);
        set.insert(&[1, 3, 7]).unwrap();

        assert_eq!(set.len(), 3);
        assert!(set.contains(3));
        assert!(!set.contains(4));

        set.remove(&[3]).unwrap();

        assert_eq!(set.len(), 2);
        assert!(!set.contains(3));
        assert_eq!(set.size(), 10);
    }

    #[test]
    fn insert_of_member_and_remove_of_non_member_are_no_ops() {
        let mut set = set_of(&[1, 2]);

        set.insert(&[1, 1]).unwrap();
        set.remove(&[5]).unwrap();

        assert_eq!(sorted_members(&set), vec![1, 2]);
        set.remove(&[1, 1]).unwrap();
        assert_eq!(sorted_members(&set), vec![2]);
    }

    #[test]
    fn out_of_range_elements_are_never_members_and_reject_the_whole_batch() {
        let mut set = set_of(&[2]);

        assert!(!set.contains(-1));
        assert!(!set.contains(i64::from(SIZE)));
        assert_eq!(
            set.insert(&[4, i64::from(SIZE)]),
            Err(TreeError::IndexOutOfBounds {
                idx: i64::from(SIZE),
                size: SIZE,
            })
        );
        assert_eq!(
            set.remove(&[2, -1]),
            Err(TreeError::IndexOutOfBounds {
                idx: -1,
                size: SIZE
            })
        );
        assert_eq!(sorted_members(&set), vec![2]);
    }

    proptest! {
        #[test]
        fn swap_remove_keeps_positions_consistent_with_model(
            ops in vec((any::<bool>(), vec(0i64..SIZE as i64, 0..6)), 0..60),
        ) {
            let mut set = UniformSet::new(SIZE);
            let mut model = HashSet::new();
            for (insert, elements) in ops {
                if insert {
                    set.insert(&elements).unwrap();
                    model.extend(elements.iter().copied());
                } else {
                    set.remove(&elements).unwrap();
                    for e in &elements {
                        model.remove(e);
                    }
                }

                prop_assert_eq!(set.len(), model.len());
                for e in 0..SIZE as i64 {
                    prop_assert_eq!(set.contains(e), model.contains(&e));
                }
                for (i, &m) in set.members.iter().enumerate() {
                    prop_assert_eq!(set.positions[m as usize] as usize, i);
                }
            }
        }
    }

    #[test]
    fn sampling_returns_only_members_roughly_uniformly() {
        let set = set_of(&[2, 5, 9, 11]);
        let mut out = vec![0i64; 20_000];
        set.sample(&mut Rng::new(1), &mut out).unwrap();

        let mut counts: HashMap<i64, usize> = HashMap::new();
        for x in out {
            *counts.entry(x).or_default() += 1;
        }

        assert_eq!(sorted_members(&set), vec![2, 5, 9, 11]);
        assert_eq!(counts.len(), 4);
        for (element, count) in counts {
            assert!(set.contains(element));
            assert!(
                (4500..=5500).contains(&count),
                "count {count} for {element}"
            );
        }
    }

    #[test]
    fn stratified_sample_hits_each_member_once_when_n_equals_len() {
        let set = set_of(&[2, 5, 9, 11]);
        let mut out = vec![0i64; 4];
        set.stratified_sample(&mut Rng::new(3), &mut out).unwrap();

        out.sort_unstable();
        assert_eq!(out, vec![2, 5, 9, 11]);
    }

    #[test]
    fn without_replacement_returns_distinct_members() {
        let set = set_of(&[2, 5, 9, 11]);
        for seed in 0..50 {
            let drawn = set
                .sample_without_replacement(&mut Rng::new(seed), 3)
                .unwrap();
            let distinct: HashSet<i64> = drawn.iter().copied().collect();

            assert_eq!(drawn.len(), 3);
            assert_eq!(distinct.len(), 3);
            assert!(drawn.iter().all(|&x| set.contains(x)));
        }

        let mut all = set.sample_without_replacement(&mut Rng::new(0), 4).unwrap();
        all.sort_unstable();
        assert_eq!(all, vec![2, 5, 9, 11]);
    }

    #[test]
    fn without_replacement_rejects_more_draws_than_members() {
        let set = set_of(&[2, 5]);

        assert_eq!(
            set.sample_without_replacement(&mut Rng::new(0), 3),
            Err(TreeError::NotEnoughElements {
                requested: 3,
                found: 2,
            })
        );
    }

    #[test]
    fn empty_set_is_defunct_and_errors_on_draws() {
        let set = UniformSet::new(SIZE);
        let mut out = [0i64; 3];

        assert!(set.is_defunct());
        assert_eq!(
            set.sample(&mut Rng::new(0), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(
            set.stratified_sample(&mut Rng::new(0), &mut out),
            Err(TreeError::EmptyDistribution)
        );
        assert_eq!(
            set.sample_without_replacement(&mut Rng::new(0), 1),
            Err(TreeError::NotEnoughElements {
                requested: 1,
                found: 0,
            })
        );
    }

    #[test]
    fn zero_draws_succeed_even_when_empty() {
        let set = UniformSet::new(SIZE);

        assert_eq!(set.sample(&mut Rng::new(0), &mut []), Ok(()));
        assert_eq!(set.stratified_sample(&mut Rng::new(0), &mut []), Ok(()));
        assert_eq!(
            set.sample_without_replacement(&mut Rng::new(0), 0),
            Ok(vec![])
        );
    }

    #[test]
    fn probs_are_uniform_over_members_and_zero_elsewhere() {
        let set = set_of(&[1, 3]);
        let mut out = [0.0; 5];
        set.probs(&[1, 2, 3, -1, 16], &mut out);

        assert_eq!(out, [0.5, 0.0, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn serde_round_trip_preserves_membership_and_draws() {
        let config = bincode::config::standard();
        let set = set_of(&[2, 5, 9]);

        let bytes = bincode::serde::encode_to_vec(&set, config).unwrap();
        let (restored, _): (UniformSet, usize) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();

        assert_eq!(restored, set);
        assert_eq!(
            restored.sample_without_replacement(&mut Rng::new(4), 3),
            set.sample_without_replacement(&mut Rng::new(4), 3),
        );
    }
}
