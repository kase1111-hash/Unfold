#!/usr/bin/env python3
"""
Comprehensive test of all knowledge graph extraction features.
Tests each module independently using direct imports to avoid neo4j dependency.
"""

import sys
import os
import importlib.util

# Test text (sample from academic paper)
TEST_TEXT = """
Vaswani et al. introduced the Transformer architecture in their seminal 2017 paper.
The model uses self-attention mechanisms to process sequences in parallel.
Google Brain and Google Research developed this approach, which has since been
adopted by many organizations. BERT, developed by Google, extends the Transformer
with bidirectional training. It was trained on BookCorpus and Wikipedia datasets.
The model outperforms previous approaches on GLUE benchmark. Unlike RNNs,
Transformers can process entire sequences simultaneously.
"""

# Expected entities for testing
ENTITIES = [
    {"text": "Vaswani et al.", "type": "PERSON"},
    {"text": "Transformer", "type": "TECHNOLOGY"},
    {"text": "Google Brain", "type": "ORGANIZATION"},
    {"text": "Google Research", "type": "ORGANIZATION"},
    {"text": "BERT", "type": "TECHNOLOGY"},
    {"text": "Google", "type": "ORGANIZATION"},
    {"text": "BookCorpus", "type": "DATASET"},
    {"text": "Wikipedia", "type": "DATASET"},
    {"text": "GLUE", "type": "DATASET"},
    {"text": "RNNs", "type": "TECHNOLOGY"},
]


def load_module_directly(module_name, file_path):
    """Load a module directly from file path, bypassing package imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_coreference():
    """Test coreference resolution module."""
    print("\n" + "="*60)
    print("TEST 1: Coreference Resolution")
    print("="*60)

    try:
        module_path = "backend/app/services/graph/coreference.py"
        coref = load_module_directly("coreference", module_path)

        CoreferenceResolver = coref.CoreferenceResolver
        create_resolver = coref.create_resolver

        resolver = create_resolver(use_llm=False)
        print("✓ CoreferenceResolver imported and created")

        # Test resolution
        resolved = resolver.resolve(TEST_TEXT, ENTITIES)
        print(f"✓ Resolved {len(resolved.resolutions)} coreferences")

        for ref, antecedent in resolved.resolutions[:3]:
            print(f"  - '{ref}' -> '{antecedent}'")

        # Test find_references
        refs = resolver.find_references(TEST_TEXT)
        print(f"✓ Found {len(refs)} references in text")

        return True

    except Exception as e:
        print(f"✗ Coreference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependency_parsing():
    """Test dependency parsing module."""
    print("\n" + "="*60)
    print("TEST 2: Dependency Parsing")
    print("="*60)

    try:
        module_path = "backend/app/services/graph/dependency_parsing.py"
        dep = load_module_directly("dependency_parsing", module_path)

        DependencyParser = dep.DependencyParser
        create_parser = dep.create_parser

        parser = create_parser(use_spacy=True)
        print(f"✓ DependencyParser created (simulated_mode={parser.simulated_mode})")

        # Test parsing
        parsed = parser.parse(TEST_TEXT)
        print(f"✓ Parsed {len(parsed)} sentences")

        # Test relation extraction
        relations = parser.extract_relations(TEST_TEXT, ENTITIES)
        print(f"✓ Extracted {len(relations)} relations")

        for rel in relations[:3]:
            print(f"  - {rel.subject} --[{rel.relation_type}]--> {rel.object}")

        return True

    except Exception as e:
        print(f"✗ Dependency parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_relations():
    """Test LLM relations module."""
    print("\n" + "="*60)
    print("TEST 3: LLM Relations Module")
    print("="*60)

    try:
        module_path = "backend/app/services/graph/llm_relations.py"
        llm = load_module_directly("llm_relations", module_path)

        LLMRelationExtractor = llm.LLMRelationExtractor
        create_llm_extractor = llm.create_llm_extractor
        OllamaProvider = llm.OllamaProvider
        LlamaCppProvider = llm.LlamaCppProvider
        OpenAIProvider = llm.OpenAIProvider
        AnthropicProvider = llm.AnthropicProvider
        EntityPair = llm.EntityPair
        RelationType = llm.RelationType

        print("✓ All LLM relations classes imported")

        # Test Ollama provider
        ollama = OllamaProvider(model="llama3.2")
        ollama_available = ollama.is_available()
        print(f"  - Ollama provider: {'available' if ollama_available else 'not available'}")

        # Test llama.cpp provider
        llama_cpp = LlamaCppProvider()
        llama_available = llama_cpp.is_available()
        print(f"  - llama.cpp provider: {'available' if llama_available else 'not available'}")

        # Test OpenAI provider (without key)
        openai = OpenAIProvider()
        openai_available = openai.is_available()
        print(f"  - OpenAI provider: {'available' if openai_available else 'not available (no key)'}")

        # Test Anthropic provider (without key)
        anthropic = AnthropicProvider()
        anthropic_available = anthropic.is_available()
        print(f"  - Anthropic provider: {'available' if anthropic_available else 'not available (no key)'}")

        # Test extractor creation
        extractor = create_llm_extractor()
        print(f"✓ LLMRelationExtractor created")
        print(f"  - Available: {extractor.is_available()}")
        print(f"  - Preferred provider: {extractor.preferred_provider}")

        # Test entity pair creation
        pair = EntityPair(
            source_text="Vaswani et al.",
            source_type="PERSON",
            target_text="Transformer",
            target_type="TECHNOLOGY",
            context="Vaswani et al. introduced the Transformer architecture"
        )
        print("✓ EntityPair created successfully")

        return True

    except Exception as e:
        print(f"✗ LLM relations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spacy_loader():
    """Test spaCy loader module."""
    print("\n" + "="*60)
    print("TEST 4: spaCy Loader")
    print("="*60)

    try:
        module_path = "backend/app/services/graph/spacy_loader.py"
        loader = load_module_directly("spacy_loader", module_path)

        is_spacy_available = loader.is_spacy_available
        get_nlp = loader.get_nlp
        parse_text = loader.parse_text

        print("✓ spacy_loader imported")

        available = is_spacy_available()
        print(f"  - spaCy available: {available}")

        if available:
            nlp = get_nlp()
            print(f"  - Model loaded: {nlp.meta['name']}")

            doc = parse_text("This is a test sentence.")
            print(f"  - Parsed: {len(list(doc.sents))} sentence(s)")

        return True

    except Exception as e:
        print(f"✗ spaCy loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_pipeline():
    """Test the integrated extraction pipeline."""
    print("\n" + "="*60)
    print("TEST 5: Integrated Pipeline")
    print("="*60)

    try:
        # First load dependencies
        coref_path = "backend/app/services/graph/coreference.py"
        coref = load_module_directly("app.services.graph.coreference", coref_path)

        dep_path = "backend/app/services/graph/dependency_parsing.py"
        dep = load_module_directly("app.services.graph.dependency_parsing", dep_path)

        llm_path = "backend/app/services/graph/llm_relations.py"
        llm = load_module_directly("app.services.graph.llm_relations", llm_path)

        # Mock the models.graph module
        class MockRelationType:
            EXPLAINS = "EXPLAINS"
            CITES = "CITES"
            CONTRASTS_WITH = "CONTRASTS_WITH"
            DERIVES_FROM = "DERIVES_FROM"
            RELATED_TO = "RELATED_TO"
            USES_METHOD = "USES_METHOD"
            USES_DATASET = "USES_DATASET"
            PART_OF = "PART_OF"
            AFFILIATED_WITH = "AFFILIATED_WITH"

            def __init__(self, value):
                self.value = value

        # Create mock module
        import types
        mock_graph = types.ModuleType('app.models.graph')
        mock_graph.RelationType = MockRelationType
        sys.modules['app.models.graph'] = mock_graph

        # Mock ExtractedEntity
        mock_extractor = types.ModuleType('app.services.graph.extractor')
        class MockExtractedEntity:
            def __init__(self, text, entity_type):
                self.text = text
                self.entity_type = type('EntityType', (), {'value': entity_type})()
        mock_extractor.ExtractedEntity = MockExtractedEntity
        sys.modules['app.services.graph.extractor'] = mock_extractor

        # Now load integrated pipeline
        module_path = "backend/app/services/graph/integrated_pipeline.py"
        pipeline = load_module_directly("integrated_pipeline", module_path)

        IntegratedRelationExtractor = pipeline.IntegratedRelationExtractor
        get_integrated_extractor = pipeline.get_integrated_extractor
        reset_extractor = pipeline.reset_extractor
        ExtractionMethod = pipeline.ExtractionMethod

        print("✓ Integrated pipeline imported")

        # Reset any existing instance
        reset_extractor()

        # Create extractor
        extractor = IntegratedRelationExtractor(
            use_coreference=True,
            use_dependency=True,
            use_llm=True,
            use_patterns=True,
            llm_provider="ollama",
        )
        print("✓ IntegratedRelationExtractor created")

        # Check status
        status = extractor.get_status()
        print("  Component status:")
        for component, available in status.items():
            print(f"    - {component}: {'✓' if available else '✗'}")

        # Create mock entities
        entities = [MockExtractedEntity(e["text"], e["type"]) for e in ENTITIES]

        # Test extraction
        relations = extractor.extract_relations(TEST_TEXT, entities)
        print(f"✓ Extracted {len(relations)} relations")

        # Group by extraction method
        by_method = {}
        for rel in relations:
            method = rel.extraction_method
            by_method[method] = by_method.get(method, 0) + 1

        print("  Relations by method:")
        for method, count in by_method.items():
            print(f"    - {method}: {count}")

        # Show sample relations
        print("  Sample relations:")
        for rel in relations[:5]:
            rel_type = rel.relation_type.value if hasattr(rel.relation_type, 'value') else str(rel.relation_type)
            print(f"    - {rel.source_text} --[{rel_type}]--> {rel.target_text} ({rel.confidence:.2f})")

        return True

    except Exception as e:
        print(f"✗ Integrated pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pattern_extraction():
    """Test pattern-based extraction independently."""
    print("\n" + "="*60)
    print("TEST 6: Pattern-Based Extraction")
    print("="*60)

    try:
        import re

        patterns = [
            (r'(\w+(?:\s+et\s+al\.)?)\s+introduced\s+(?:the\s+)?(\w+)', 'INTRODUCES'),
            (r'(\w+(?:\s+et\s+al\.)?)\s+developed\s+(?:the\s+)?(\w+)', 'DEVELOPS'),
            (r'(\w+)\s+uses?\s+(\w+)', 'USES'),
            (r'(\w+)\s+trained\s+on\s+(\w+)', 'TRAINED_ON'),
            (r'(\w+)\s+outperforms?\s+(\w+)', 'OUTPERFORMS'),
        ]

        text_lower = TEST_TEXT.lower()
        entity_set = {e["text"].lower() for e in ENTITIES}

        found = []
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text_lower):
                source = match.group(1)
                target = match.group(2)

                # Check if matches entities
                source_match = any(source in ent or ent in source for ent in entity_set)
                target_match = any(target in ent or ent in target for ent in entity_set)

                if source_match or target_match:
                    found.append((source, rel_type, target, match.group(0)))

        print(f"✓ Pattern extraction found {len(found)} matches")
        for src, rel, tgt, context in found[:5]:
            print(f"  - {src} --[{rel}]--> {tgt}")
            print(f"    Context: '{context[:50]}...'")

        return True

    except Exception as e:
        print(f"✗ Pattern extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all feature tests."""
    print("\n" + "#"*60)
    print("# KNOWLEDGE GRAPH EXTRACTION FEATURE TESTS")
    print("#"*60)

    results = {
        "Coreference Resolution": test_coreference(),
        "Dependency Parsing": test_dependency_parsing(),
        "LLM Relations": test_llm_relations(),
        "spaCy Loader": test_spacy_loader(),
        "Integrated Pipeline": test_integrated_pipeline(),
        "Pattern Extraction": test_pattern_extraction(),
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0
    for test, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
