# Contributing guidelines

## Before contributing

Welcome to sieve! Before contributing to the project, please ensure you understand our contribution process.

## Contributing code

1. Set up a Python development environment
   (advice: use [venv](https://docs.python.org/3/library/venv.html),
   [virtualenv](https://virtualenv.pypa.io/), or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
2. Install tox: `python -m pip install tox`
3. Clone the repository: `git clone https://github.com/YOUR_USERNAME/sieve.git`
4. Start a new branch off main: `git switch -c new-branch main`
5. Make your code changes
6. Check that your code follows the style guidelines of the project: `tox -e check`
7. Build the documentation: `tox -e docs`
8. Run the tests: `tox -e py310`
   (change the version number according to the Python you are using)
9. Commit, push, and open a pull request!

## Development environment

We use [pyproject.toml](pyproject.toml) to manage project dependencies and configuration. You can install the package in development mode with:

```bash
pip install -e .
```

## Tests

Tests are located in the `tests/` directory and can be run with tox:

```bash
tox -e py310
```

## Documentation

Documentation is built using Sphinx and is located in the `docs/` directory. Build the documentation with:

```bash
tox -e docs
```

## Code style

We follow the PEP 8 style guide. You can check your code with:

```bash
tox -e check
```
