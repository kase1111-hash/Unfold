"""
Export Service for Anki and Obsidian formats.
Converts flashcards and notes to portable formats.
"""

import json
import csv
import io
import zipfile
import re
from typing import Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class FlashcardData:
    """Flashcard data for export."""

    card_id: str
    question: str
    answer: str
    tags: list[str]
    hint: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None


class ExportService:
    """
    Exports flashcards and learning data to various formats.
    Supports Anki (APKG/CSV) and Obsidian (Markdown) formats.
    """

    def __init__(self):
        self.default_deck_name = "Unfold Flashcards"

    def export_to_anki_csv(
        self,
        flashcards: list[FlashcardData],
        include_tags: bool = True,
    ) -> str:
        """
        Export flashcards to Anki-compatible CSV format.

        The CSV format is:
        front;back;tags

        Args:
            flashcards: List of flashcards to export
            include_tags: Whether to include tags column

        Returns:
            CSV string content
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=";", quoting=csv.QUOTE_ALL)

        for card in flashcards:
            front = card.question
            back = card.answer

            # Add hint to front if available
            if card.hint:
                front += f"\n\n<i>Hint: {card.hint}</i>"

            # Add source to back if available
            if card.source:
                back += f"\n\n<small>Source: {card.source}</small>"

            row = [front, back]

            if include_tags:
                tags = " ".join(card.tags) if card.tags else ""
                row.append(tags)

            writer.writerow(row)

        return output.getvalue()

    def export_to_anki_txt(
        self,
        flashcards: list[FlashcardData],
    ) -> str:
        """
        Export flashcards to Anki-compatible text format.

        Format:
        Q: question
        A: answer
        Tags: tag1, tag2

        Args:
            flashcards: List of flashcards to export

        Returns:
            Text string content
        """
        lines = []

        for card in flashcards:
            lines.append(f"Q: {card.question}")
            lines.append(f"A: {card.answer}")
            if card.tags:
                lines.append(f"Tags: {', '.join(card.tags)}")
            lines.append("")  # Blank line between cards

        return "\n".join(lines)

    def export_to_anki_json(
        self,
        flashcards: list[FlashcardData],
        deck_name: Optional[str] = None,
    ) -> str:
        """
        Export flashcards to JSON format for Anki import tools.

        Args:
            flashcards: List of flashcards to export
            deck_name: Name of the Anki deck

        Returns:
            JSON string content
        """
        deck_name = deck_name or self.default_deck_name

        data = {
            "deck_name": deck_name,
            "cards": [],
            "exported_at": datetime.utcnow().isoformat(),
            "source": "Unfold",
        }

        for card in flashcards:
            card_data = {
                "id": card.card_id,
                "front": card.question,
                "back": card.answer,
                "tags": card.tags or [],
            }

            if card.hint:
                card_data["hint"] = card.hint

            if card.source:
                card_data["source"] = card.source

            data["cards"].append(card_data)

        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_to_obsidian_markdown(
        self,
        flashcards: list[FlashcardData],
        title: str = "Flashcards",
        include_frontmatter: bool = True,
    ) -> str:
        """
        Export flashcards to Obsidian-compatible Markdown.

        Uses the Obsidian Spaced Repetition plugin format:
        #flashcard
        Question
        ?
        Answer

        Args:
            flashcards: List of flashcards to export
            title: Document title
            include_frontmatter: Whether to include YAML frontmatter

        Returns:
            Markdown string content
        """
        lines = []

        # YAML frontmatter
        if include_frontmatter:
            lines.extend(
                [
                    "---",
                    f"title: {title}",
                    f"created: {datetime.utcnow().strftime('%Y-%m-%d')}",
                    "tags:",
                    "  - flashcards",
                    "  - unfold-export",
                    "---",
                    "",
                ]
            )

        # Title
        lines.append(f"# {title}")
        lines.append("")
        lines.append(
            f"*Exported from Unfold on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}*"
        )
        lines.append("")

        # Flashcards
        for card in flashcards:
            # Add tags as Obsidian tags
            if card.tags:
                tag_line = " ".join([f"#{tag}" for tag in card.tags])
                lines.append(tag_line)

            lines.append("#flashcard")
            lines.append(card.question)

            if card.hint:
                lines.append(f"*Hint: {card.hint}*")

            lines.append("?")
            lines.append(card.answer)

            if card.source:
                lines.append(f"*Source: {card.source}*")

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def export_to_obsidian_callout(
        self,
        flashcards: list[FlashcardData],
        title: str = "Flashcards",
    ) -> str:
        """
        Export flashcards using Obsidian callout syntax.

        > [!question]- Question text
        > Answer text

        Args:
            flashcards: List of flashcards to export
            title: Document title

        Returns:
            Markdown string content
        """
        lines = [
            f"# {title}",
            "",
            f"*Exported from Unfold on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        for card in flashcards:
            # Question as collapsible callout
            lines.append(f"> [!question]- {card.question}")

            if card.hint:
                lines.append(f"> *Hint: {card.hint}*")
                lines.append(">")

            # Answer
            for answer_line in card.answer.split("\n"):
                lines.append(f"> {answer_line}")

            if card.source:
                lines.append(">")
                lines.append(f"> *Source: {card.source}*")

            # Tags
            if card.tags:
                lines.append(">")
                lines.append(f"> Tags: {', '.join(card.tags)}")

            lines.append("")

        return "\n".join(lines)

    def export_to_markdown_table(
        self,
        flashcards: list[FlashcardData],
        title: str = "Flashcards",
    ) -> str:
        """
        Export flashcards as a Markdown table.

        Args:
            flashcards: List of flashcards to export
            title: Document title

        Returns:
            Markdown string content
        """
        lines = [
            f"# {title}",
            "",
            "| Question | Answer | Tags |",
            "|----------|--------|------|",
        ]

        for card in flashcards:
            # Escape pipe characters in content
            question = card.question.replace("|", "\\|").replace("\n", " ")
            answer = card.answer.replace("|", "\\|").replace("\n", " ")
            tags = ", ".join(card.tags) if card.tags else ""

            # Truncate long content
            if len(question) > 100:
                question = question[:97] + "..."
            if len(answer) > 100:
                answer = answer[:97] + "..."

            lines.append(f"| {question} | {answer} | {tags} |")

        return "\n".join(lines)

    def export_to_json(
        self,
        flashcards: list[FlashcardData],
        include_metadata: bool = True,
    ) -> str:
        """
        Export flashcards to generic JSON format.

        Args:
            flashcards: List of flashcards to export
            include_metadata: Whether to include export metadata

        Returns:
            JSON string content
        """
        data = {
            "flashcards": [
                {
                    "id": card.card_id,
                    "question": card.question,
                    "answer": card.answer,
                    "tags": card.tags or [],
                    "hint": card.hint,
                    "source": card.source,
                    "created_at": (
                        card.created_at.isoformat() if card.created_at else None
                    ),
                }
                for card in flashcards
            ],
        }

        if include_metadata:
            data["metadata"] = {
                "exported_at": datetime.utcnow().isoformat(),
                "source": "Unfold",
                "version": "1.0",
                "count": len(flashcards),
            }

        return json.dumps(data, indent=2, ensure_ascii=False)

    def create_export_bundle(
        self,
        flashcards: list[FlashcardData],
        title: str = "Unfold Export",
    ) -> bytes:
        """
        Create a ZIP bundle with all export formats.

        Args:
            flashcards: List of flashcards to export
            title: Export title/name

        Returns:
            ZIP file bytes
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Safe filename
            safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")

            # Add all formats
            zf.writestr(
                f"{safe_title}_anki.csv",
                self.export_to_anki_csv(flashcards),
            )
            zf.writestr(
                f"{safe_title}_anki.txt",
                self.export_to_anki_txt(flashcards),
            )
            zf.writestr(
                f"{safe_title}_anki.json",
                self.export_to_anki_json(flashcards, deck_name=title),
            )
            zf.writestr(
                f"{safe_title}_obsidian_sr.md",
                self.export_to_obsidian_markdown(flashcards, title=title),
            )
            zf.writestr(
                f"{safe_title}_obsidian_callout.md",
                self.export_to_obsidian_callout(flashcards, title=title),
            )
            zf.writestr(
                f"{safe_title}_table.md",
                self.export_to_markdown_table(flashcards, title=title),
            )
            zf.writestr(
                f"{safe_title}.json",
                self.export_to_json(flashcards),
            )

            # Add README
            readme = f"""# {title} Export

This bundle contains flashcards exported from Unfold in multiple formats.

## Files

- `{safe_title}_anki.csv` - CSV format for Anki import
- `{safe_title}_anki.txt` - Text format for Anki import
- `{safe_title}_anki.json` - JSON format for Anki tools
- `{safe_title}_obsidian_sr.md` - Obsidian Spaced Repetition plugin format
- `{safe_title}_obsidian_callout.md` - Obsidian callout format
- `{safe_title}_table.md` - Markdown table format
- `{safe_title}.json` - Generic JSON format

## Anki Import Instructions

1. Open Anki
2. Go to File > Import
3. Select the CSV or TXT file
4. Set field separator to semicolon (;) for CSV
5. Map fields: Field 1 = Front, Field 2 = Back, Field 3 = Tags

## Obsidian Import Instructions

1. Copy the desired .md file to your Obsidian vault
2. For spaced repetition, install the "Spaced Repetition" community plugin
3. Use the `_sr.md` file format with that plugin

Exported on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
            zf.writestr("README.md", readme)

        return buffer.getvalue()


# Singleton instance
_export_service: Optional[ExportService] = None


def get_export_service() -> ExportService:
    """Get or create singleton ExportService instance."""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
