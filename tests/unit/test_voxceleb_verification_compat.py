from __future__ import annotations

from pathlib import Path

import pytest

from recipes.voxceleb.verification_compat import parse_verification_pair


def test_parse_verification_pair_accepts_variable_whitespace() -> None:
    assert parse_verification_pair("1   id10270/x6uYqmx31kE/00001.wav  id10271/abc/00002.wav") == (
        1,
        "id10270/x6uYqmx31kE/00001",
        "id10271/abc/00002",
    )


def test_parse_verification_pair_ignores_blank_lines() -> None:
    assert parse_verification_pair("  \n", source="veri_test2.txt", line_number=7) is None


def test_parse_verification_pair_reports_file_and_line_for_malformed_rows() -> None:
    with pytest.raises(ValueError, match="veri_test2.txt:3"):
        parse_verification_pair("1 only_one_field", source="veri_test2.txt", line_number=3)


def test_voxceleb_recipes_do_not_use_positional_space_split_for_verification() -> None:
    project_root = Path(__file__).parents[2]
    for relative in (
        "recipes/voxceleb/voxceleb_prepare.py",
        "recipes/voxceleb/speaker_verification_cosine.py",
    ):
        source = (project_root / relative).read_text(encoding="utf-8")
        assert 'line.split(" ")' not in source
        assert "parse_verification_pair" in source