# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [1.0.0] candidate

### Changed
- Rename `MPArray.bdims` -> `MPArray.ranks`
- Rename `MPArray.pdims` -> `MPArray.shape`
- Rename `MPArray.plegs` -> `MPArray.ndims`
- Rename `MPArray.normal_form` -> `MPArray.canonical_form`
- Rename `MPArray.normalize` -> `MPArray.canonicalize`
- Rename `MPArray.get_phys` -> `MPArray.get`
- Rename `MPArray.paxis_iter` -> `MPArray.axis_iter`
- Rename `MPArray.pleg2bleg` -> `MPArray.leg2vleg`
- Rename `LocalTensors.pdims` -> `LocalTensors.shape`
- Rename `LocalTensors.normal_form` -> `LocalTensors.canonical_form`
- Rename `mparray.full_bdim` -> `mparray.full_rank`
- Rename `mparray.outer` -> `mparray.chain`
- Rename `mparray.louter` -> `mparray.outer`

### Removed
- Remove `MPArray.bdim`, use `max(mpa.bdims)` instead
- Remove `mpo_to_global`, use `to_array_global` instead
