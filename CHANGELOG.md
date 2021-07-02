# Changelog

## [Unreleased]

### Added
- ability to filter examples out during training of `CLMaskPretrainer`
- ability to transform examples during training of `CLMaskPretrainer`

## [0.1.2] - 2021-06-30
### Added
- `Shelf` implementation and tests
- `Embedder` class

### Changed
- Moved CCLMModelBase to Embedder (breaking change)
- `Preprocessor.string_to_array` makes `length` optional
- Updated docker image tf version

### Removed
- `setup.py`
- Removed all uses of `CCLMModelBase`

## [0.1.1] - 2021-05-07
### Changed
- Switched to poetry

### Deprecated
- `setup.py` installation

## [0.1.0] - 2021-05-06
### Added
- `augmentation` module
- make publishable to pypi
- refactor into submodules