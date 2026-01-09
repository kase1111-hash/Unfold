"""
Bias Audit Module for content analysis and fairness assessment.
Implements sentiment analysis and language inclusivity metrics.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

# Optional imports for sentiment analysis
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    pipeline = None


class BiasCategory(str, Enum):
    """Categories of potential bias."""
    GENDER = "gender"
    RACIAL = "racial"
    POLITICAL = "political"
    RELIGIOUS = "religious"
    SOCIOECONOMIC = "socioeconomic"
    AGEISM = "ageism"
    ABLEISM = "ableism"
    LANGUAGE = "language"


class SeverityLevel(str, Enum):
    """Severity levels for bias findings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    label: str  # positive, negative, neutral
    score: float  # Confidence score 0-1
    text_sample: Optional[str] = None


@dataclass
class BiasFinding:
    """A single bias finding from audit."""
    finding_id: str
    category: BiasCategory
    severity: SeverityLevel
    description: str
    text_excerpt: Optional[str] = None
    position: Optional[tuple[int, int]] = None  # Start, end offsets
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "text_excerpt": self.text_excerpt,
            "position": self.position,
            "suggestions": self.suggestions,
        }


@dataclass
class InclusivityMetrics:
    """Language inclusivity metrics."""
    overall_score: float  # 0-100
    gender_neutral_score: float
    accessible_language_score: float
    reading_level: str  # elementary, intermediate, advanced
    complex_term_count: int
    passive_voice_ratio: float
    average_sentence_length: float

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "gender_neutral_score": round(self.gender_neutral_score, 1),
            "accessible_language_score": round(self.accessible_language_score, 1),
            "reading_level": self.reading_level,
            "complex_term_count": self.complex_term_count,
            "passive_voice_ratio": round(self.passive_voice_ratio, 2),
            "average_sentence_length": round(self.average_sentence_length, 1),
        }


@dataclass
class BiasAuditReport:
    """Complete bias audit report for a document."""
    report_id: str
    document_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Sentiment analysis
    overall_sentiment: Optional[SentimentResult] = None
    section_sentiments: list[SentimentResult] = field(default_factory=list)

    # Bias findings
    findings: list[BiasFinding] = field(default_factory=list)

    # Inclusivity metrics
    inclusivity: Optional[InclusivityMetrics] = None

    # Summary scores
    bias_risk_score: float = 0.0  # 0-100, higher = more risk
    transparency_score: float = 100.0  # 0-100, higher = more transparent

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat(),
            "sentiment": {
                "overall": {
                    "label": self.overall_sentiment.label,
                    "score": round(self.overall_sentiment.score, 3),
                } if self.overall_sentiment else None,
                "section_count": len(self.section_sentiments),
            },
            "findings": [f.to_dict() for f in self.findings],
            "findings_by_severity": {
                "critical": len([f for f in self.findings if f.severity == SeverityLevel.CRITICAL]),
                "high": len([f for f in self.findings if f.severity == SeverityLevel.HIGH]),
                "medium": len([f for f in self.findings if f.severity == SeverityLevel.MEDIUM]),
                "low": len([f for f in self.findings if f.severity == SeverityLevel.LOW]),
            },
            "inclusivity": self.inclusivity.to_dict() if self.inclusivity else None,
            "scores": {
                "bias_risk": round(self.bias_risk_score, 1),
                "transparency": round(self.transparency_score, 1),
            },
            "recommendations": self.recommendations,
        }


class BiasAuditor:
    """
    Service for auditing content for bias and inclusivity.
    """

    # Gender-specific terms that could be replaced with neutral alternatives
    GENDERED_TERMS = {
        "he": "they",
        "she": "they",
        "his": "their",
        "her": "their",
        "him": "them",
        "himself": "themselves",
        "herself": "themselves",
        "mankind": "humankind",
        "manmade": "artificial",
        "fireman": "firefighter",
        "policeman": "police officer",
        "chairman": "chairperson",
        "businessman": "businessperson",
        "mankind": "humanity",
        "man-hours": "person-hours",
        "manpower": "workforce",
    }

    # Words/phrases that may indicate bias
    BIAS_INDICATORS = {
        BiasCategory.GENDER: [
            r"\bmen are\b", r"\bwomen are\b", r"\bgirls are\b", r"\bboys are\b",
            r"\btypically male\b", r"\btypically female\b",
        ],
        BiasCategory.RACIAL: [
            r"\bill?egals?\b", r"\balien\b", r"\bexotic\b",
        ],
        BiasCategory.AGEISM: [
            r"\bold people are\b", r"\byoung people are\b", r"\belderly\b",
            r"\bsenile\b", r"\bover the hill\b",
        ],
        BiasCategory.ABLEISM: [
            r"\bcrippled?\b", r"\blame\b", r"\bhandicapped\b",
            r"\bsuffering from\b", r"\bconfined to\b", r"\bwheelchair.?bound\b",
        ],
    }

    # Complex/jargon terms
    COMPLEX_TERMS = [
        "paradigm", "synergy", "leverage", "utilize", "operationalize",
        "methodology", "heretofore", "aforementioned", "notwithstanding",
        "pursuant", "vis-à-vis", "ergo", "ipso facto",
    ]

    def __init__(self):
        self._sentiment_pipeline = None
        self._finding_counter = 0

    def _get_sentiment_pipeline(self):
        """Lazy load sentiment analysis pipeline."""
        if self._sentiment_pipeline is None and HAS_TRANSFORMERS:
            try:
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                )
            except Exception:
                pass
        return self._sentiment_pipeline

    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        self._finding_counter += 1
        return f"finding_{self._finding_counter:04d}"

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with label and confidence
        """
        pipeline_fn = self._get_sentiment_pipeline()

        if pipeline_fn is not None:
            try:
                # Truncate text for model
                truncated = text[:512]
                result = pipeline_fn(truncated)[0]
                return SentimentResult(
                    label=result["label"].lower(),
                    score=result["score"],
                    text_sample=truncated[:100] + "..." if len(truncated) > 100 else truncated,
                )
            except Exception:
                pass

        # Fallback: Simple keyword-based sentiment
        positive_words = ["good", "great", "excellent", "positive", "beneficial", "helpful"]
        negative_words = ["bad", "poor", "negative", "harmful", "problematic", "concerning"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            label = "positive"
            score = 0.6 + (positive_count - negative_count) * 0.05
        elif negative_count > positive_count:
            label = "negative"
            score = 0.6 + (negative_count - positive_count) * 0.05
        else:
            label = "neutral"
            score = 0.5

        return SentimentResult(
            label=label,
            score=min(1.0, score),
            text_sample=text[:100] + "..." if len(text) > 100 else text,
        )

    def detect_gendered_language(self, text: str) -> list[BiasFinding]:
        """
        Detect potentially gendered language.

        Args:
            text: Text to analyze

        Returns:
            List of bias findings
        """
        findings = []

        for term, neutral in self.GENDERED_TERMS.items():
            pattern = rf"\b{re.escape(term)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(BiasFinding(
                    finding_id=self._generate_finding_id(),
                    category=BiasCategory.GENDER,
                    severity=SeverityLevel.LOW,
                    description=f"Gendered term '{match.group()}' could be replaced with gender-neutral alternative",
                    text_excerpt=text[max(0, match.start() - 30):match.end() + 30],
                    position=(match.start(), match.end()),
                    suggestions=[f"Consider using '{neutral}' instead of '{match.group()}'"],
                ))

        return findings

    def detect_bias_patterns(self, text: str) -> list[BiasFinding]:
        """
        Detect bias patterns in text.

        Args:
            text: Text to analyze

        Returns:
            List of bias findings
        """
        findings = []

        for category, patterns in self.BIAS_INDICATORS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    findings.append(BiasFinding(
                        finding_id=self._generate_finding_id(),
                        category=category,
                        severity=SeverityLevel.MEDIUM,
                        description=f"Potentially biased language pattern detected: '{match.group()}'",
                        text_excerpt=text[max(0, match.start() - 50):match.end() + 50],
                        position=(match.start(), match.end()),
                        suggestions=["Review this passage for unintentional bias"],
                    ))

        return findings

    def calculate_inclusivity_metrics(self, text: str) -> InclusivityMetrics:
        """
        Calculate language inclusivity metrics.

        Args:
            text: Text to analyze

        Returns:
            InclusivityMetrics
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        # Gender neutrality score
        gendered_count = 0
        for term in self.GENDERED_TERMS.keys():
            gendered_count += len(re.findall(rf"\b{term}\b", text, re.IGNORECASE))

        total_words = len(words)
        gender_neutral_score = max(0, 100 - (gendered_count / max(1, total_words)) * 1000)

        # Complex term count
        complex_count = 0
        for term in self.COMPLEX_TERMS:
            complex_count += len(re.findall(rf"\b{term}\b", text, re.IGNORECASE))

        # Accessible language score
        avg_word_length = sum(len(w) for w in words) / max(1, len(words))
        accessible_score = max(0, 100 - (avg_word_length - 4) * 10 - complex_count * 2)

        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))

        # Passive voice detection (simple heuristic)
        passive_patterns = re.findall(r"\b(was|were|been|being|is|are)\s+\w+ed\b", text, re.IGNORECASE)
        passive_ratio = len(passive_patterns) / max(1, len(sentences))

        # Reading level estimation
        if avg_sentence_length < 15 and avg_word_length < 5:
            reading_level = "elementary"
        elif avg_sentence_length < 25 and avg_word_length < 6:
            reading_level = "intermediate"
        else:
            reading_level = "advanced"

        # Overall score
        overall_score = (
            gender_neutral_score * 0.3 +
            accessible_score * 0.4 +
            max(0, 100 - passive_ratio * 50) * 0.3
        )

        return InclusivityMetrics(
            overall_score=overall_score,
            gender_neutral_score=gender_neutral_score,
            accessible_language_score=accessible_score,
            reading_level=reading_level,
            complex_term_count=complex_count,
            passive_voice_ratio=passive_ratio,
            average_sentence_length=avg_sentence_length,
        )

    def generate_recommendations(
        self,
        findings: list[BiasFinding],
        inclusivity: InclusivityMetrics,
    ) -> list[str]:
        """
        Generate recommendations based on audit results.

        Args:
            findings: List of bias findings
            inclusivity: Inclusivity metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        # Finding-based recommendations
        finding_categories = set(f.category for f in findings)

        if BiasCategory.GENDER in finding_categories:
            recommendations.append(
                "Consider reviewing the text for gender-neutral language. "
                "Replace gendered terms with inclusive alternatives where appropriate."
            )

        if BiasCategory.ABLEISM in finding_categories:
            recommendations.append(
                "Review language related to disability. "
                "Use person-first language and avoid terms that imply limitation."
            )

        # Severity-based recommendations
        high_severity = [f for f in findings if f.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
        if high_severity:
            recommendations.append(
                f"Address {len(high_severity)} high-severity finding(s) that may significantly impact reader perception."
            )

        # Inclusivity-based recommendations
        if inclusivity.reading_level == "advanced":
            recommendations.append(
                "Consider simplifying language to improve accessibility. "
                f"Average sentence length is {inclusivity.average_sentence_length:.1f} words."
            )

        if inclusivity.complex_term_count > 5:
            recommendations.append(
                f"Found {inclusivity.complex_term_count} complex/jargon terms. "
                "Consider providing definitions or using simpler alternatives."
            )

        if inclusivity.passive_voice_ratio > 0.3:
            recommendations.append(
                "Consider reducing passive voice usage for clearer, more direct writing."
            )

        if not recommendations:
            recommendations.append("No significant issues detected. Content appears well-balanced and inclusive.")

        return recommendations

    def audit_document(
        self,
        document_id: str,
        content: str,
        sections: Optional[list[str]] = None,
    ) -> BiasAuditReport:
        """
        Perform a complete bias audit on a document.

        Args:
            document_id: Document identifier
            content: Full document content
            sections: Optional list of document sections

        Returns:
            Complete BiasAuditReport
        """
        import uuid as uuid_module

        # Analyze overall sentiment
        overall_sentiment = self.analyze_sentiment(content)

        # Analyze section sentiments
        section_sentiments = []
        if sections:
            for section in sections[:10]:  # Limit to 10 sections
                section_sentiments.append(self.analyze_sentiment(section))

        # Detect bias
        findings = []
        findings.extend(self.detect_gendered_language(content))
        findings.extend(self.detect_bias_patterns(content))

        # Calculate inclusivity
        inclusivity = self.calculate_inclusivity_metrics(content)

        # Generate recommendations
        recommendations = self.generate_recommendations(findings, inclusivity)

        # Calculate risk score
        severity_weights = {
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 3,
            SeverityLevel.HIGH: 7,
            SeverityLevel.CRITICAL: 15,
        }
        total_weight = sum(severity_weights.get(f.severity, 1) for f in findings)
        bias_risk_score = min(100, total_weight * 2)

        # Calculate transparency score
        transparency_score = max(0, 100 - bias_risk_score * 0.5)

        return BiasAuditReport(
            report_id=f"audit_{uuid_module.uuid4().hex[:12]}",
            document_id=document_id,
            overall_sentiment=overall_sentiment,
            section_sentiments=section_sentiments,
            findings=findings,
            inclusivity=inclusivity,
            bias_risk_score=bias_risk_score,
            transparency_score=transparency_score,
            recommendations=recommendations,
        )


# Singleton instance
_bias_auditor: Optional[BiasAuditor] = None


def get_bias_auditor() -> BiasAuditor:
    """Get or create singleton BiasAuditor instance."""
    global _bias_auditor
    if _bias_auditor is None:
        _bias_auditor = BiasAuditor()
    return _bias_auditor
