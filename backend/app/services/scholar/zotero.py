"""
Zotero Export Service for citation management.
Generates Zotero-compatible RIS, BibTeX, and CSL-JSON exports.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ZoteroItem:
    """Represents a citation item for Zotero export."""

    item_type: str = "journalArticle"  # book, bookSection, conferencePaper, etc.
    title: str = ""
    authors: list[dict] = field(
        default_factory=list
    )  # [{"firstName": "", "lastName": ""}]
    abstract: Optional[str] = None
    date: Optional[str] = None  # YYYY-MM-DD or YYYY
    doi: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    issn: Optional[str] = None
    language: str = "en"
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ZoteroCollection:
    """Represents a collection of citations."""

    name: str
    items: list[ZoteroItem] = field(default_factory=list)
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class ZoteroExporter:
    """
    Exports citations to various formats compatible with Zotero.
    """

    def export_to_ris(self, items: list[ZoteroItem]) -> str:
        """
        Export citations to RIS format.

        Args:
            items: List of ZoteroItem objects

        Returns:
            RIS formatted string
        """
        lines = []

        for item in items:
            # Map item type to RIS type
            type_map = {
                "journalArticle": "JOUR",
                "book": "BOOK",
                "bookSection": "CHAP",
                "conferencePaper": "CONF",
                "thesis": "THES",
                "report": "RPRT",
                "webpage": "ELEC",
                "preprint": "UNPB",
            }
            ris_type = type_map.get(item.item_type, "GEN")

            lines.append(f"TY  - {ris_type}")
            lines.append(f"TI  - {item.title}")

            for author in item.authors:
                first = author.get("firstName", "")
                last = author.get("lastName", "")
                if last:
                    lines.append(f"AU  - {last}, {first}".strip(", "))

            if item.abstract:
                lines.append(f"AB  - {item.abstract}")

            if item.date:
                lines.append(f"PY  - {item.date[:4]}")  # Year only
                lines.append(f"DA  - {item.date}")

            if item.journal:
                lines.append(f"JO  - {item.journal}")
                lines.append(f"T2  - {item.journal}")

            if item.volume:
                lines.append(f"VL  - {item.volume}")

            if item.issue:
                lines.append(f"IS  - {item.issue}")

            if item.pages:
                if "-" in item.pages:
                    start, end = item.pages.split("-", 1)
                    lines.append(f"SP  - {start.strip()}")
                    lines.append(f"EP  - {end.strip()}")
                else:
                    lines.append(f"SP  - {item.pages}")

            if item.doi:
                lines.append(f"DO  - {item.doi}")

            if item.url:
                lines.append(f"UR  - {item.url}")

            if item.publisher:
                lines.append(f"PB  - {item.publisher}")

            if item.isbn:
                lines.append(f"SN  - {item.isbn}")
            elif item.issn:
                lines.append(f"SN  - {item.issn}")

            if item.language:
                lines.append(f"LA  - {item.language}")

            for tag in item.tags:
                lines.append(f"KW  - {tag}")

            for note in item.notes:
                lines.append(f"N1  - {note}")

            lines.append("ER  - ")
            lines.append("")

        return "\n".join(lines)

    def export_to_bibtex(self, items: list[ZoteroItem]) -> str:
        """
        Export citations to BibTeX format.

        Args:
            items: List of ZoteroItem objects

        Returns:
            BibTeX formatted string
        """
        entries = []

        for i, item in enumerate(items):
            # Map item type to BibTeX type
            type_map = {
                "journalArticle": "article",
                "book": "book",
                "bookSection": "incollection",
                "conferencePaper": "inproceedings",
                "thesis": "phdthesis",
                "report": "techreport",
                "webpage": "misc",
                "preprint": "unpublished",
            }
            bib_type = type_map.get(item.item_type, "misc")

            # Generate citation key
            if item.authors:
                first_author = item.authors[0].get("lastName", "Unknown")
            else:
                first_author = "Unknown"
            year = item.date[:4] if item.date else "nd"
            cite_key = f"{first_author.lower()}{year}_{i}"

            lines = [f"@{bib_type}{{{cite_key},"]

            # Title
            lines.append(f"  title = {{{item.title}}},")

            # Authors
            if item.authors:
                author_strs = []
                for author in item.authors:
                    first = author.get("firstName", "")
                    last = author.get("lastName", "")
                    if last:
                        author_strs.append(f"{last}, {first}".strip(", "))
                lines.append(f"  author = {{{' and '.join(author_strs)}}},")

            # Year
            if item.date:
                lines.append(f"  year = {{{item.date[:4]}}},")

            # Journal/booktitle
            if item.journal:
                if bib_type == "article":
                    lines.append(f"  journal = {{{item.journal}}},")
                else:
                    lines.append(f"  booktitle = {{{item.journal}}},")

            # Volume and number
            if item.volume:
                lines.append(f"  volume = {{{item.volume}}},")
            if item.issue:
                lines.append(f"  number = {{{item.issue}}},")

            # Pages
            if item.pages:
                lines.append(f"  pages = {{{item.pages}}},")

            # DOI
            if item.doi:
                lines.append(f"  doi = {{{item.doi}}},")

            # URL
            if item.url:
                lines.append(f"  url = {{{item.url}}},")

            # Publisher
            if item.publisher:
                lines.append(f"  publisher = {{{item.publisher}}},")

            # Abstract
            if item.abstract:
                # Escape special characters
                abstract = item.abstract.replace("{", "\\{").replace("}", "\\}")
                lines.append(f"  abstract = {{{abstract}}},")

            # Keywords
            if item.tags:
                lines.append(f"  keywords = {{{', '.join(item.tags)}}},")

            lines.append("}")
            entries.append("\n".join(lines))

        return "\n\n".join(entries)

    def export_to_csl_json(self, items: list[ZoteroItem]) -> str:
        """
        Export citations to CSL-JSON format (used by Zotero internally).

        Args:
            items: List of ZoteroItem objects

        Returns:
            CSL-JSON formatted string
        """
        csl_items = []

        for i, item in enumerate(items):
            # Map item type to CSL type
            type_map = {
                "journalArticle": "article-journal",
                "book": "book",
                "bookSection": "chapter",
                "conferencePaper": "paper-conference",
                "thesis": "thesis",
                "report": "report",
                "webpage": "webpage",
                "preprint": "article",
            }

            csl_item = {
                "id": f"item_{i}",
                "type": type_map.get(item.item_type, "document"),
                "title": item.title,
            }

            # Authors
            if item.authors:
                csl_item["author"] = [
                    {
                        "family": a.get("lastName", ""),
                        "given": a.get("firstName", ""),
                    }
                    for a in item.authors
                ]

            # Date
            if item.date:
                parts = item.date.split("-")
                date_parts = [[int(p) for p in parts if p.isdigit()]]
                csl_item["issued"] = {"date-parts": date_parts}

            # Container (journal, book title, etc.)
            if item.journal:
                csl_item["container-title"] = item.journal

            # Volume and issue
            if item.volume:
                csl_item["volume"] = item.volume
            if item.issue:
                csl_item["issue"] = item.issue

            # Pages
            if item.pages:
                csl_item["page"] = item.pages

            # DOI
            if item.doi:
                csl_item["DOI"] = item.doi

            # URL
            if item.url:
                csl_item["URL"] = item.url

            # Publisher
            if item.publisher:
                csl_item["publisher"] = item.publisher

            # Abstract
            if item.abstract:
                csl_item["abstract"] = item.abstract

            # Language
            if item.language:
                csl_item["language"] = item.language

            csl_items.append(csl_item)

        return json.dumps(csl_items, indent=2, ensure_ascii=False)

    def export_collection(
        self,
        collection: ZoteroCollection,
        format: str = "ris",
    ) -> str:
        """
        Export a collection to the specified format.

        Args:
            collection: ZoteroCollection to export
            format: Export format (ris, bibtex, csl-json)

        Returns:
            Formatted string
        """
        if format == "ris":
            return self.export_to_ris(collection.items)
        elif format == "bibtex":
            return self.export_to_bibtex(collection.items)
        elif format == "csl-json":
            return self.export_to_csl_json(collection.items)
        else:
            raise ValueError(f"Unknown format: {format}")

    def create_item_from_dict(self, data: dict) -> ZoteroItem:
        """
        Create a ZoteroItem from a dictionary.

        Args:
            data: Dictionary with item data

        Returns:
            ZoteroItem object
        """
        # Parse authors
        authors = []
        for author in data.get("authors", []):
            if isinstance(author, str):
                # Parse "Last, First" or "First Last" format
                if "," in author:
                    parts = author.split(",", 1)
                    authors.append(
                        {
                            "lastName": parts[0].strip(),
                            "firstName": parts[1].strip() if len(parts) > 1 else "",
                        }
                    )
                else:
                    parts = author.rsplit(" ", 1)
                    authors.append(
                        {
                            "firstName": parts[0] if len(parts) > 1 else "",
                            "lastName": parts[-1],
                        }
                    )
            elif isinstance(author, dict):
                authors.append(author)

        return ZoteroItem(
            item_type=data.get("type", "journalArticle"),
            title=data.get("title", ""),
            authors=authors,
            abstract=data.get("abstract"),
            date=data.get("date") or data.get("year"),
            doi=data.get("doi"),
            url=data.get("url"),
            journal=data.get("journal") or data.get("venue"),
            volume=data.get("volume"),
            issue=data.get("issue"),
            pages=data.get("pages"),
            publisher=data.get("publisher"),
            isbn=data.get("isbn"),
            issn=data.get("issn"),
            tags=data.get("tags", []),
            notes=data.get("notes", []),
        )


# Singleton instance
_zotero_exporter: Optional[ZoteroExporter] = None


def get_zotero_exporter() -> ZoteroExporter:
    """Get or create singleton ZoteroExporter instance."""
    global _zotero_exporter
    if _zotero_exporter is None:
        _zotero_exporter = ZoteroExporter()
    return _zotero_exporter
