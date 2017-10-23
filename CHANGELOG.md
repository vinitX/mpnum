# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [1.0.0] candidate

### Changed
- `MPArray.compress`: `method='var'` requires `num_sweeps`
- `mparray.linalg.eig`:
  - Change order of parameters
  - Require `num_sweeps` and one of `startvec` and `startvec_rank`
- Rename `MPArray.bdims` -> `MPArray.ranks`
- Rename `MPArray.pdims` -> `MPArray.shape`
- Rename `MPArray.plegs` -> `MPArray.ndims`
- Rename `MPArray.normal_form` -> `MPArray.canonical_form`
- Rename `MPArray.normalize` -> `MPArray.canonicalize`
- Rename `MPArray.get_phys` -> `MPArray.get`
- Rename `MPArray.paxis_iter` -> `MPArray.axis_iter`
- Rename `MPArray.pleg2bleg` -> `MPArray.leg2vleg`
- Rename `MPArray.dims` -> `MPArray.lt.shape`
- Rename `LocalTensors.normal_form` -> `LocalTensors.canonical_form`
- Rename `mparray.full_bdim` -> `mparray.full_rank`
- Rename `mparray.outer` -> `mparray.chain`
- Rename `mparray.louter` -> `mparray.localouter`
- Rename `mparray.linalg.mineig` -> `mparray.linalg.eig`
- Rename `mparray.linalg.mineig_sum` -> `mparray.linalg.eig_sum`
- Rename `mparray.tools` -> `mparray.utils`
- Rename `mparray.tools.check_nonneg_trunc` -> `mparray.utils.pmf.project_nonneg`
- Rename `mparray.tools.check_pmf` -> `mparray.utils.pmf.project_pmf`
- Rename `mparray.linalg.mineig` -> `mparray.linalg.eig`
- `mparray.linalg.eig`:
  - Rename parameter `max_num_sweeps` -> `num_sweeps` (now required)
  - Rename parameter `minimize_sites` -> `var_sites`

### Removed
- Remove `MPArray.bdim`, use `max(mpa.ranks)` instead
- Remove `mpo_to_global`, use `to_array_global` instead
- Remove `mpnum.tools.verify_real_nonnegative`, use `mpnum.utils.pmf.project_nonneg` instead
