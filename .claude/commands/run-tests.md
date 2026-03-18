# Run Tests

Execute the test suite for pytorch_segmentation_models_trainer using the project venv.

## Usage

```
/run-tests [args]
```

Where `[args]` is an optional filter passed directly to pytest:
- `/run-tests` — runs all tests
- `/run-tests tests/test_model.py` — runs a specific file
- `/run-tests tests/test_loss_builder.py -v` — runs with verbose output
- `/run-tests -k test_compute_loss` — runs tests matching a keyword
- `/run-tests tests/ -x` — stops on first failure

## Instructions

Run the test suite using the project's venv at `.venv/`. Always use the venv Python and pytest.

Execute the following command (replace `$ARGUMENTS` with the args provided):

```bash
cd /Users/philipeborba/github_repos/pytorch_segmentation_models_trainer && .venv/bin/python -m pytest $ARGUMENTS -v --tb=short 2>&1
```

If no arguments are provided, run:
```bash
cd /Users/philipeborba/github_repos/pytorch_segmentation_models_trainer && .venv/bin/python -m pytest tests/ -v --tb=short --ignore=tests/test_inference.py --ignore=tests/test_predict.py --ignore=tests/test_script.py 2>&1
```

After running, report:
1. Total tests: passed / failed / errored / skipped
2. For each failure or error: the test name, the error type, and the relevant traceback lines
3. A brief summary of what needs to be fixed (if anything)
