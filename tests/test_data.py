import pytest

from qqp.config import Config, load_config
from qqp.data import normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("How can I learn Python?", "how can i learn python"),
        ("  What's   the BEST way?!  ", "whats the best way"),
        ("C++ vs. C#", "c vs c"),
        ("", ""),
        (None, "none"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_default_config_file_matches_dataclass_defaults():
    assert load_config("configs/default.yaml") == Config()


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("rnn:\n  hidden_size: 64\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden_size"):
        load_config(path)
