# CLAUDE.md — Stable-Baselines3

## Project Overview

Stable-Baselines3 (SB3) is a PyTorch-based library of reliable reinforcement learning algorithm implementations. It provides a sklearn-like API for training RL agents with Gymnasium environments.

- **Version**: 2.8.0a3
- **License**: MIT
- **Python**: >=3.10 (supports 3.10, 3.11, 3.12, 3.13)
- **Core deps**: `gymnasium>=0.29.1,<1.3.0`, `numpy>=1.20,<3.0`, `torch>=2.3,<3.0`

## Quick Reference — Common Commands

```bash
# Install for development
pip install -e '.[docs,tests,extra]'

# Run tests (excludes expensive tests, includes coverage)
make pytest

# Format code (ruff import sorting + black formatting)
make format

# Check code style without fixing
make check-codestyle

# Lint with ruff
make lint

# Type check with mypy
make type

# Run all pre-commit checks (format + type + lint)
make commit-checks

# Build documentation
make doc

# Check documentation spelling
make spelling
```

## Repository Structure

```
stable_baselines3/
├── a2c/                    # Advantage Actor-Critic (on-policy)
├── ppo/                    # Proximal Policy Optimization (on-policy)
├── dqn/                    # Deep Q-Network (off-policy, discrete)
├── sac/                    # Soft Actor-Critic (off-policy, continuous)
├── td3/                    # Twin Delayed DDPG (off-policy, continuous)
├── ddpg/                   # Deep Deterministic PG (off-policy, special case of TD3)
├── her/                    # Hindsight Experience Replay (replay buffer, not standalone algo)
├── common/
│   ├── base_class.py       # BaseAlgorithm — root of all algorithms
│   ├── on_policy_algorithm.py   # Base for A2C, PPO
│   ├── off_policy_algorithm.py  # Base for DQN, SAC, TD3, DDPG
│   ├── policies.py         # ActorCriticPolicy, MlpPolicy, CnnPolicy, MultiInputPolicy
│   ├── buffers.py          # ReplayBuffer, RolloutBuffer, DictReplayBuffer, etc.
│   ├── torch_layers.py     # NatureCNN, MlpExtractor, create_mlp()
│   ├── distributions.py    # DiagGaussian, Categorical, StateDependentNoise, etc.
│   ├── callbacks.py        # BaseCallback, EvalCallback, CheckpointCallback
│   ├── logger.py           # TensorBoard/CSV/JSON logging
│   ├── evaluation.py       # evaluate_policy()
│   ├── utils.py            # set_random_seed(), get_device(), polyak_update()
│   ├── noise.py            # ActionNoise, OrnsteinUhlenbeckActionNoise
│   ├── monitor.py          # Monitor wrapper for environments
│   ├── env_checker.py      # Environment validation utilities
│   ├── preprocessing.py    # Observation preprocessing
│   ├── type_aliases.py     # Common type hints
│   ├── save_util.py        # Model save/load utilities
│   ├── vec_env/            # Vectorized environments (DummyVecEnv, SubprocVecEnv, VecNormalize, etc.)
│   ├── envs/               # Test environments (identity, bit-flipping, multi-input)
│   └── sb2_compat/         # Stable Baselines v2 compatibility (TF-like RMSProp)
tests/                      # ~27 test files covering all components
docs/                       # Sphinx documentation (RST format)
scripts/                    # Build, test, and Docker scripts
```

## Architecture

### Class Hierarchy

```
BaseAlgorithm
├── OnPolicyAlgorithm        # Rollout buffer, GAE
│   ├── A2C
│   └── PPO
└── OffPolicyAlgorithm       # Replay buffer, action noise
    ├── DQN
    ├── SAC
    └── TD3
        └── DDPG             # Special case of TD3
```

### Each Algorithm Module Contains

- `<algo>.py` — The algorithm implementation (inherits from On/OffPolicyAlgorithm)
- `policies.py` — Algorithm-specific policy networks
- `__init__.py` — Exports and policy aliases (`MlpPolicy`, `CnnPolicy`, `MultiInputPolicy`)

### Key Design Patterns

- **Policy aliases**: Each algorithm registers string-to-class mappings (`"MlpPolicy"` -> class)
- **Buffer abstraction**: `RolloutBuffer` for on-policy, `ReplayBuffer` for off-policy
- **Callback system**: Hook into training via `BaseCallback` subclasses
- **Schedule functions**: Learning rate and other hyperparameters can be callables `(progress_remaining) -> float`
- **Device-agnostic**: Automatic CPU/CUDA selection via `get_device()`

## Code Style and Conventions

### Formatting

- **Line length**: 127 characters (code and config), 88 characters (documentation)
- **Formatter**: Black
- **Import sorting**: Ruff (isort rules)
- **Linter**: Ruff with rules: E, F, B, UP, C90, RUF
- **Max cyclomatic complexity**: 15

### Type Hints and Docstrings

All functions must have type hints and Google-style docstrings using Sphinx `:param:` / `:return:` format:

```python
def my_function(arg1: type1, arg2: type2) -> returntype:
    """
    Short description of the function.

    :param arg1: describe what is arg1
    :param arg2: describe what is arg2
    :return: describe what is returned
    """
```

### MyPy Configuration

- `ignore_missing_imports = true`
- `follow_imports = "silent"`
- Excludes: `tests/test_logger.py`, `tests/test_train_eval_mode.py`

## Testing

Tests live in `tests/` and use pytest. The test runner script is `scripts/run_tests.sh`:

```bash
python3 -m pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=. -v --color=yes -m "not expensive"
```

- Tests marked `@pytest.mark.expensive` are excluded from default runs
- `PYTHONHASHSEED=0` is set for deterministic ordering (useful with pytest-xdist)
- Coverage omits: `tests/`, `setup.py`, `results_plotter.py`, `vec_video_recorder.py`

### Running Specific Tests

```bash
# Run a single test file
python -m pytest tests/test_run.py -v

# Run a single test function
python -m pytest tests/test_run.py::test_ppo -v

# Run tests matching a pattern
python -m pytest tests/ -k "test_save" -v

# Include expensive tests
python -m pytest tests/ -v
```

## CI Pipeline

Defined in `.github/workflows/ci.yml`. Runs on push/PR to master.

- **Matrix**: Python 3.10, 3.11, 3.12, 3.13
- **Gymnasium versions**: 1.0.0 (default) and 0.29.1 (legacy, Python 3.10 only)
- **Steps**: lint -> doc build -> codestyle check -> type check -> tests
- **Skip CI**: Include `[ci skip]` in commit message

## Important Conventions for AI Assistants

1. **Always run `make format` after code changes** to ensure correct formatting and import ordering.
2. **Run `make commit-checks`** before finalizing changes — this runs format, type, and lint checks.
3. **New features require tests** in the `tests/` directory.
4. **Update the changelog** at `docs/misc/changelog.rst` for user-facing changes.
5. **HER is a replay buffer**, not a standalone algorithm — use `HerReplayBuffer` class.
6. **DDPG is a special case of TD3** — it inherits TD3's policy and overrides minimal behavior.
7. **Ruff ignored rules**: B028 (explicit stacklevel) and RUF013 (implicit optional) are intentionally suppressed.
8. **Per-file ruff ignores**: B027 allowed in `callbacks.py` and `noise.py` (abstract method stubs); RUF012/RUF013 allowed in tests.
9. **Do not modify test_logger.py or test_train_eval_mode.py** for mypy compliance — they are excluded from type checking.
10. **Version is stored in** `stable_baselines3/version.txt`, not in `setup.py` or `__init__.py`.

## Related Projects

- **SB3-Contrib**: Additional algorithms (TQC, QRDQN, etc.) — `stable-baselines3-contrib`
- **RL Zoo**: Training framework and hyperparameter tuning — `rl-baselines3-zoo`
- **SBX**: JAX-based implementations — `sbx`
