"""Tests for ZoteroExporter.create_item_from_dict."""

import json

import pytest

from app.services.scholar.zotero import ZoteroExporter


@pytest.mark.parametrize("year", [2020, "2020"])
def test_numeric_or_string_year_exports(year):
    exporter = ZoteroExporter()
    item = exporter.create_item_from_dict(
        {"title": "On Radium", "authors": ["Marie Curie"], "year": year}
    )

    assert item.date == "2020"
    assert exporter.export_to_bibtex([item]).splitlines()[0] == "@article{curie2020_0,"
    assert "PY  - 2020" in exporter.export_to_ris([item])
    csl = json.loads(exporter.export_to_csl_json([item]))
    assert csl[0]["issued"] == {"date-parts": [[2020]]}


def test_missing_year_leaves_date_empty():
    exporter = ZoteroExporter()
    item = exporter.create_item_from_dict({"title": "Untitled"})
    assert item.date is None
    assert exporter.export_to_bibtex([item]).splitlines()[0] == "@article{unknownnd_0,"
