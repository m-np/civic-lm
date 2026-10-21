from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "data" / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES
