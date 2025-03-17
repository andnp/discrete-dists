## 1.1.2 (2025-03-17)

### Fix

- move to new bincode API

## 1.1.1 (2025-01-28)

### Fix

- add special case handling for proportional and uniform dists

## 1.1.0 (2025-01-28)

### Feat

- allow sampling from mixture dist with defunct subdists

### Fix

- heavily simplify required interface for distribution

## 1.0.3 (2025-01-26)

### Fix

- fix uniform idx updating semantics

## 1.0.2 (2025-01-26)

### Fix

- ensure updating uniform support is right inclusive

## 1.0.1 (2025-01-26)

### Fix

- scale stratification buckets to total size of sumtree

## 1.0.0 (2025-01-24)

### BREAKING CHANGE

- change the external API to use the term `elements`
when referring to things in the support of the distribution.
- technically this isn't a strict breaking change (it is
fully backwards compatible); however the intended semantics of the
library's usage are changing in this commit, plus we _want_ to move to
v1, so it seems appropriate.

### Feat

- add public facing methods to change dist support
- allow proportional dist to have shifted support
- define uniform distributions as having lo and hi support bounds

### Fix

- convert set to intermediate list before making array

### Refactor

- change idxs -> elements

## 0.2.3 (2025-01-23)

### Fix

- dont skip release if tag was successfully run

## 0.2.2 (2025-01-23)

### Fix

- use first build for release immediately following tagging

## 0.2.1 (2025-01-23)

### Fix

- include more project metadata in pyproject.toml

## 0.2.0 (2025-01-23)

### Feat

- allow updating uniform distribution support with single arg

## 0.1.2 (2025-01-23)

### Fix

- distribute rust sumtree pyi type stubs

## 0.1.1 (2025-01-23)

### Fix

- convert to vec intermediate rep to resolve to_pyarray not found error
- breaking ordering semantics of sample

## 0.1.0 (2025-01-23)

### Feat

- build base distribution types
- migrate sumtree from replaytables

### Fix

- checkout code in order to tag
- automate tagging and publishing

### Refactor

- migrate from multi-dim sumtree -> single-dim
