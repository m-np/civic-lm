from pathlib import Path

import pytest
from civic_bench.generators import definitions, superseded
from civic_bench.generators.common import REPO_ROOT, Tables, definition_text, window

PROCESSED = REPO_ROOT / "data" / "processed"


def test_definition_text():
    assert definition_text("(a) Act means the Federal Food, Drug, and Cosmetic Act.", "Act") == \
        "the Federal Food, Drug, and Cosmetic Act."
    assert definition_text("Design history file (DHF) means a compilation", "Design history file") == "a compilation"


def test_window_includes_center_and_respects_exclude():
    ps = [{"text": f"p{i} " + "w " * 50} for i in range(10)]
    text, idx = window(ps, 5, words=120)
    assert 5 in idx and text.count("p") >= 2
    _, idx2 = window(ps, 5, exclude={5}, words=120)
    assert 5 not in idx2


@pytest.mark.skipif(not (PROCESSED / "diffs.parquet").exists(), reason="needs data/processed (run make data)")
def test_generators_are_deterministic_and_split_safe():
    t = Tables.load(REPO_ROOT / "configs" / "scope.yaml", Path(PROCESSED))
    a = definitions.generate(t, n_base=6, seed=3, dataset_version="t") + \
        superseded.generate(t, n_base=4, seed=3, dataset_version="t")
    b = definitions.generate(t, n_base=6, seed=3, dataset_version="t") + \
        superseded.generate(t, n_base=4, seed=3, dataset_version="t")
    assert [i.model_dump_json() for i in a] == [i.model_dump_json() for i in b]
    # a lineage never appears in more than one split
    splits = {}
    for it in a:
        assert splits.setdefault(it.lineage_id, it.split) == it.split
    # no question names its section number
    for it in a:
        sec = it.lineage_id.split("-CFR-")[1]
        assert sec not in it.question
