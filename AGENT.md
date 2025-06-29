# LightFM Agent Configuration

## Build/Test Commands
- Run all tests: `pytest`
- Run single test file: `pytest tests/test_api.py`
- Run specific test: `pytest tests/test_api.py::test_empty_matrix`
- Build package: `python setup.py build`
- Install in dev mode: `pip install -e .`
- Cythonize extensions: `python setup.py cythonize`
- Clean build files: `python setup.py clean`

## Linting/Formatting
- Use black formatter (configured in .pre-commit-config.yaml)
- Use flake8 linting with max line length 100
- Pre-commit hooks check trailing whitespace, EOF, YAML format

## Architecture
- Main package: `lightfm/` - core recommendation library
- Key modules: `lightfm.py` (main LightFM class), `data.py` (data handling), `evaluation.py` (metrics)
- C extensions: Cython-generated files for performance (`_lightfm_fast_*.c`)
- Compatibility layer: `lightfm/compat/` for cross-platform support
- Datasets: `lightfm/datasets/` contains built-in dataset loaders

## Code Style
- Use black formatting (22.1.0)
- Imports: NumPy as `np`, SciPy sparse as `sp`
- Type hints: Use for public APIs
- Error handling: Raise appropriate exceptions with descriptive messages
- Naming: snake_case for functions/variables, PascalCase for classes
