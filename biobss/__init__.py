from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TESTS_ROOT = PROJECT_ROOT / "tests"
FIXTURES_ROOT = PROJECT_ROOT / "sample_data"

from biobss import (  # isort: skip
    common,
    ecgtools,
    edatools,
    hrvtools,
    imutools,
    pipeline,
    plottools,
    ppgtools,
    preprocess,
    reader,
    resptools,
    sqatools,
    timetools,
    utils,
)

__version__ = "0.1.1"
