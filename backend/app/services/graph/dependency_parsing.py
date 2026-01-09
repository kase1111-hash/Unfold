"""
Dependency Parsing Module for Relation Extraction

Uses spaCy's dependency parser to extract syntactic relations between entities.
Falls back to pattern-based parsing when spaCy model is unavailable.

Dependency patterns commonly used:
- nsubj (nominal subject): "Google developed GPT" -> (Google, developed, GPT)
- dobj (direct object): "researchers created model" -> (researchers, created, model)
- pobj (prepositional object): "trained on dataset" -> (model, trained_on, dataset)
- compound: "neural network" -> compound relation
- appos (apposition): "BERT, a language model" -> (BERT, is_a, language model)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import re


class DependencyRelation(Enum):
    """Syntactic dependency relations that map to semantic relations."""
    NSUBJ = "nsubj"          # nominal subject
    NSUBJPASS = "nsubjpass"  # passive nominal subject
    DOBJ = "dobj"            # direct object
    POBJ = "pobj"            # prepositional object
    COMPOUND = "compound"    # compound modifier
    APPOS = "appos"          # appositional modifier
    AMOD = "amod"            # adjectival modifier
    PREP = "prep"            # prepositional modifier
    AGENT = "agent"          # agent (passive voice)
    ATTR = "attr"            # attribute
    CONJ = "conj"            # conjunction
    ACL = "acl"              # clausal modifier
    RELCL = "relcl"          # relative clause modifier


@dataclass
class DependencyEdge:
    """Represents a dependency edge between tokens."""
    head: str
    dependent: str
    relation: str
    head_idx: int
    dep_idx: int


@dataclass
class ParsedSentence:
    """Parsed sentence with tokens and dependencies."""
    text: str
    tokens: List[str]
    pos_tags: List[str]
    dependencies: List[DependencyEdge]
    entities_found: List[Tuple[str, int, int]]  # (text, start_idx, end_idx)


@dataclass
class ExtractedRelation:
    """Relation extracted from dependency parse."""
    subject: str
    predicate: str
    object: str
    relation_type: str
    confidence: float
    evidence: str
    dependency_path: List[str] = field(default_factory=list)


# Verb patterns that indicate specific relations
VERB_RELATION_MAP = {
    # Creation/Development
    "develop": "DEVELOPS",
    "developed": "DEVELOPS",
    "create": "CREATES",
    "created": "CREATES",
    "build": "CREATES",
    "built": "CREATES",
    "design": "CREATES",
    "designed": "CREATES",
    "implement": "IMPLEMENTS",
    "implemented": "IMPLEMENTS",

    # Introduction/Proposal
    "introduce": "INTRODUCES",
    "introduced": "INTRODUCES",
    "propose": "PROPOSES",
    "proposed": "PROPOSES",
    "present": "PRESENTS",
    "presented": "PRESENTS",
    "publish": "PUBLISHES",
    "published": "PUBLISHES",

    # Usage/Application
    "use": "USES",
    "used": "USES",
    "uses": "USES",
    "utilize": "USES",
    "utilized": "USES",
    "employ": "USES",
    "employed": "USES",
    "apply": "APPLIES",
    "applied": "APPLIES",
    "leverage": "USES",
    "leveraged": "USES",

    # Training/Evaluation
    "train": "TRAINED_ON",
    "trained": "TRAINED_ON",
    "evaluate": "EVALUATED_ON",
    "evaluated": "EVALUATED_ON",
    "test": "EVALUATED_ON",
    "tested": "EVALUATED_ON",
    "benchmark": "EVALUATED_ON",
    "benchmarked": "EVALUATED_ON",
    "fine-tune": "FINE_TUNED_ON",
    "fine-tuned": "FINE_TUNED_ON",

    # Comparison
    "outperform": "OUTPERFORMS",
    "outperformed": "OUTPERFORMS",
    "exceed": "OUTPERFORMS",
    "exceeded": "OUTPERFORMS",
    "surpass": "OUTPERFORMS",
    "surpassed": "OUTPERFORMS",
    "rival": "COMPETES_WITH",
    "rivals": "COMPETES_WITH",

    # Extension/Modification
    "extend": "EXTENDS",
    "extended": "EXTENDS",
    "modify": "MODIFIES",
    "modified": "MODIFIES",
    "improve": "IMPROVES",
    "improved": "IMPROVES",
    "enhance": "ENHANCES",
    "enhanced": "ENHANCES",

    # Composition
    "contain": "CONTAINS",
    "contains": "CONTAINS",
    "include": "CONTAINS",
    "includes": "CONTAINS",
    "consist": "CONSISTS_OF",
    "consists": "CONSISTS_OF",
    "comprise": "COMPRISES",
    "comprises": "COMPRISES",
}

# Preposition patterns
PREP_RELATION_MAP = {
    "on": "EVALUATED_ON",
    "with": "USES",
    "for": "USED_FOR",
    "by": "DEVELOPED_BY",
    "at": "AFFILIATED_WITH",
    "from": "DERIVED_FROM",
    "in": "APPEARS_IN",
    "into": "TRANSFORMS_INTO",
    "using": "USES",
    "via": "USES",
    "through": "USES",
}


class DependencyParser:
    """
    Dependency parser for relation extraction.
    Uses spaCy when available, falls back to pattern-based parsing.
    """

    def __init__(self, use_spacy: bool = True):
        self.nlp = None
        self.use_spacy = use_spacy
        self.simulated_mode = False

        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                print("  [DependencyParser] Using spaCy for parsing")
            except (ImportError, OSError) as e:
                print(f"  [DependencyParser] spaCy unavailable: {e}")
                print("  [DependencyParser] Falling back to pattern-based parsing")
                self.simulated_mode = True
        else:
            self.simulated_mode = True
            print("  [DependencyParser] Using pattern-based parsing (simulated mode)")

    def parse(self, text: str) -> List[ParsedSentence]:
        """Parse text into sentences with dependency information."""
        if self.nlp and not self.simulated_mode:
            return self._parse_with_spacy(text)
        else:
            return self._parse_with_patterns(text)

    def _parse_with_spacy(self, text: str) -> List[ParsedSentence]:
        """Parse using spaCy's dependency parser."""
        doc = self.nlp(text)
        parsed_sentences = []

        for sent in doc.sents:
            tokens = [token.text for token in sent]
            pos_tags = [token.pos_ for token in sent]

            dependencies = []
            for token in sent:
                if token.head != token:  # Skip root
                    edge = DependencyEdge(
                        head=token.head.text,
                        dependent=token.text,
                        relation=token.dep_,
                        head_idx=token.head.i - sent.start,
                        dep_idx=token.i - sent.start
                    )
                    dependencies.append(edge)

            # Extract entities
            entities_found = []
            for ent in sent.ents:
                entities_found.append((ent.text, ent.start - sent.start, ent.end - sent.start))

            parsed_sentences.append(ParsedSentence(
                text=sent.text,
                tokens=tokens,
                pos_tags=pos_tags,
                dependencies=dependencies,
                entities_found=entities_found
            ))

        return parsed_sentences

    def _parse_with_patterns(self, text: str) -> List[ParsedSentence]:
        """Pattern-based parsing fallback when spaCy is unavailable."""
        # Protect abbreviations
        protected = text
        abbrevs = [
            (r'\bet al\.', 'ET_AL_PROT'),
            (r'\bMr\.', 'MR_PROT'),
            (r'\bDr\.', 'DR_PROT'),
            (r'\bvs\.', 'VS_PROT'),
            (r'\bFig\.', 'FIG_PROT'),
            (r'\bEq\.', 'EQ_PROT'),
            (r'\bi\.e\.', 'IE_PROT'),
            (r'\be\.g\.', 'EG_PROT'),
        ]
        for pattern, replacement in abbrevs:
            protected = re.sub(pattern, replacement, protected)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', protected)

        # Restore abbreviations
        restored_sentences = []
        for sent in sentences:
            for pattern, replacement in abbrevs:
                original = pattern.replace(r'\b', '').replace('\\', '')
                sent = sent.replace(replacement, original)
            restored_sentences.append(sent)

        parsed_sentences = []
        for sent in restored_sentences:
            parsed = self._pattern_parse_sentence(sent)
            parsed_sentences.append(parsed)

        return parsed_sentences

    def _pattern_parse_sentence(self, sentence: str) -> ParsedSentence:
        """Parse a single sentence using patterns."""
        # Simple tokenization
        tokens = re.findall(r'\b[\w\-]+\b|[.,;:!?]', sentence)

        # Simple POS tagging based on patterns
        pos_tags = self._simple_pos_tag(tokens)

        # Generate synthetic dependencies based on patterns
        dependencies = self._generate_pattern_dependencies(tokens, pos_tags)

        return ParsedSentence(
            text=sentence,
            tokens=tokens,
            pos_tags=pos_tags,
            dependencies=dependencies,
            entities_found=[]
        )

    def _simple_pos_tag(self, tokens: List[str]) -> List[str]:
        """Simple rule-based POS tagging."""
        pos_tags = []

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Punctuation
            if token in '.,;:!?':
                pos_tags.append('PUNCT')
            # Verbs (based on common patterns)
            elif token_lower in VERB_RELATION_MAP:
                pos_tags.append('VERB')
            # Determiners
            elif token_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                pos_tags.append('DET')
            # Prepositions
            elif token_lower in PREP_RELATION_MAP or token_lower in {'of', 'to'}:
                pos_tags.append('ADP')
            # Auxiliaries
            elif token_lower in {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'has', 'have', 'had'}:
                pos_tags.append('AUX')
            # Conjunctions
            elif token_lower in {'and', 'or', 'but', 'nor', 'yet', 'so'}:
                pos_tags.append('CCONJ')
            # Proper nouns (capitalized, not first word)
            elif token[0].isupper() and i > 0:
                pos_tags.append('PROPN')
            # First word capitalized might be noun or proper noun
            elif token[0].isupper() and i == 0:
                pos_tags.append('NOUN')
            # Words ending in -ly are often adverbs
            elif token_lower.endswith('ly'):
                pos_tags.append('ADV')
            # Words ending in -ing might be verbs
            elif token_lower.endswith('ing'):
                pos_tags.append('VERB')
            # Words ending in -ed might be verbs
            elif token_lower.endswith('ed'):
                pos_tags.append('VERB')
            # Default to noun
            else:
                pos_tags.append('NOUN')

        return pos_tags

    def _generate_pattern_dependencies(self, tokens: List[str], pos_tags: List[str]) -> List[DependencyEdge]:
        """Generate synthetic dependencies based on sentence patterns."""
        dependencies = []

        # Find main verb
        verb_idx = None
        for i, pos in enumerate(pos_tags):
            if pos == 'VERB':
                verb_idx = i
                break

        if verb_idx is None:
            return dependencies

        verb = tokens[verb_idx]

        # Subject: noun/proper noun before verb
        for i in range(verb_idx - 1, -1, -1):
            if pos_tags[i] in ('NOUN', 'PROPN'):
                dependencies.append(DependencyEdge(
                    head=verb,
                    dependent=tokens[i],
                    relation='nsubj',
                    head_idx=verb_idx,
                    dep_idx=i
                ))
                break

        # Object: noun/proper noun after verb
        for i in range(verb_idx + 1, len(tokens)):
            if pos_tags[i] in ('NOUN', 'PROPN'):
                # Check if preceded by preposition
                if i > 0 and pos_tags[i-1] == 'ADP':
                    dependencies.append(DependencyEdge(
                        head=tokens[i-1],  # preposition is head
                        dependent=tokens[i],
                        relation='pobj',
                        head_idx=i-1,
                        dep_idx=i
                    ))
                    dependencies.append(DependencyEdge(
                        head=verb,
                        dependent=tokens[i-1],
                        relation='prep',
                        head_idx=verb_idx,
                        dep_idx=i-1
                    ))
                else:
                    dependencies.append(DependencyEdge(
                        head=verb,
                        dependent=tokens[i],
                        relation='dobj',
                        head_idx=verb_idx,
                        dep_idx=i
                    ))
                break

        return dependencies

    def extract_relations(self, text: str, entities: List[Dict]) -> List[ExtractedRelation]:
        """
        Extract relations between entities using dependency parsing.

        Args:
            text: Input text
            entities: List of entity dicts with 'text' and 'type' keys

        Returns:
            List of extracted relations
        """
        entity_texts = {e['text'].lower(): e for e in entities}
        entity_set = set(entity_texts.keys())

        parsed = self.parse(text)
        relations = []

        for sentence in parsed:
            sentence_relations = self._extract_from_sentence(
                sentence, entity_texts, entity_set
            )
            relations.extend(sentence_relations)

        return relations

    def _extract_from_sentence(
        self,
        sentence: ParsedSentence,
        entity_texts: Dict[str, Dict],
        entity_set: Set[str]
    ) -> List[ExtractedRelation]:
        """Extract relations from a parsed sentence."""
        relations = []
        sent_lower = sentence.text.lower()

        # Find entities in this sentence
        entities_in_sent = []
        for ent_text in entity_set:
            if ent_text in sent_lower:
                entities_in_sent.append(ent_text)

        if len(entities_in_sent) < 2:
            return relations

        # Look for verb-mediated relations
        for dep in sentence.dependencies:
            if dep.relation == 'nsubj':
                # Found subject-verb relation
                subject = dep.dependent
                verb = dep.head

                # Find object
                for dep2 in sentence.dependencies:
                    if dep2.head == verb and dep2.relation in ('dobj', 'attr'):
                        obj = dep2.dependent

                        # Check if subject and object are entities
                        subj_match = self._match_entity(subject, entities_in_sent)
                        obj_match = self._match_entity(obj, entities_in_sent)

                        if subj_match and obj_match:
                            rel_type = VERB_RELATION_MAP.get(verb.lower(), 'RELATED_TO')
                            relations.append(ExtractedRelation(
                                subject=entity_texts[subj_match]['text'],
                                predicate=verb,
                                object=entity_texts[obj_match]['text'],
                                relation_type=rel_type,
                                confidence=0.8,
                                evidence=sentence.text,
                                dependency_path=[dep.relation, dep2.relation]
                            ))

            elif dep.relation == 'pobj':
                # Prepositional object - look for prep relation
                prep = None
                for dep2 in sentence.dependencies:
                    if dep2.dependent == dep.head and dep2.relation == 'prep':
                        prep = dep.head
                        verb = dep2.head
                        obj = dep.dependent

                        # Find subject of verb
                        for dep3 in sentence.dependencies:
                            if dep3.head == verb and dep3.relation == 'nsubj':
                                subject = dep3.dependent

                                subj_match = self._match_entity(subject, entities_in_sent)
                                obj_match = self._match_entity(obj, entities_in_sent)

                                if subj_match and obj_match:
                                    # Combine verb and prep for relation type
                                    verb_rel = VERB_RELATION_MAP.get(verb.lower())
                                    prep_rel = PREP_RELATION_MAP.get(prep.lower())
                                    rel_type = verb_rel or prep_rel or 'RELATED_TO'

                                    relations.append(ExtractedRelation(
                                        subject=entity_texts[subj_match]['text'],
                                        predicate=f"{verb} {prep}",
                                        object=entity_texts[obj_match]['text'],
                                        relation_type=rel_type,
                                        confidence=0.75,
                                        evidence=sentence.text,
                                        dependency_path=[dep3.relation, dep2.relation, dep.relation]
                                    ))
                        break

        # Pattern-based extraction for common constructs
        pattern_relations = self._extract_pattern_relations(sentence.text, entities_in_sent, entity_texts)
        relations.extend(pattern_relations)

        return relations

    def _match_entity(self, token: str, entities: List[str]) -> Optional[str]:
        """Match a token to an entity (handles partial matches)."""
        token_lower = token.lower()

        # Exact match
        if token_lower in entities:
            return token_lower

        # Partial match (token is part of entity or vice versa)
        for ent in entities:
            if token_lower in ent or ent in token_lower:
                return ent

        return None

    def _extract_pattern_relations(
        self,
        sentence: str,
        entities_in_sent: List[str],
        entity_texts: Dict[str, Dict]
    ) -> List[ExtractedRelation]:
        """Extract relations using regex patterns."""
        relations = []
        sent_lower = sentence.lower()

        # Pattern: X introduced Y
        for pattern, rel_type in [
            (r'(\w+(?:\s+et\s+al\.)?)\s+introduced\s+(\w+)', 'INTRODUCES'),
            (r'(\w+(?:\s+et\s+al\.)?)\s+developed\s+(\w+)', 'DEVELOPS'),
            (r'(\w+(?:\s+et\s+al\.)?)\s+created\s+(\w+)', 'CREATES'),
            (r'(\w+)\s+uses?\s+(\w+)', 'USES'),
            (r'(\w+)\s+trained\s+on\s+(\w+)', 'TRAINED_ON'),
            (r'(\w+)\s+evaluated\s+on\s+(\w+)', 'EVALUATED_ON'),
            (r'(\w+)\s+at\s+(\w+)', 'AFFILIATED_WITH'),
            (r'(\w+)\s+outperforms?\s+(\w+)', 'OUTPERFORMS'),
            (r'(\w+)\s+rivals?\s+(\w+)', 'COMPETES_WITH'),
        ]:
            matches = re.finditer(pattern, sent_lower)
            for match in matches:
                subj_text = match.group(1)
                obj_text = match.group(2)

                # Match to entities
                subj_match = None
                obj_match = None

                for ent in entities_in_sent:
                    if subj_text in ent or ent in subj_text:
                        subj_match = ent
                    if obj_text in ent or ent in obj_text:
                        obj_match = ent

                if subj_match and obj_match and subj_match != obj_match:
                    relations.append(ExtractedRelation(
                        subject=entity_texts[subj_match]['text'],
                        predicate=pattern.split(r'\s+')[1].replace('\\', '').replace('s?', ''),
                        object=entity_texts[obj_match]['text'],
                        relation_type=rel_type,
                        confidence=0.7,
                        evidence=sentence,
                        dependency_path=['pattern_match']
                    ))

        return relations


def create_parser(use_spacy: bool = True) -> DependencyParser:
    """Factory function to create a dependency parser."""
    return DependencyParser(use_spacy=use_spacy)
