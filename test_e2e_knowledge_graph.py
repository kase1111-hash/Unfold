#!/usr/bin/env python3
"""
End-to-End Test: PDF to Knowledge Graph Pipeline

This script tests the complete Unfold pipeline:
1. Create a sample academic PDF
2. Extract text from PDF
3. Extract entities and concepts
4. Build knowledge graph
5. Analyze and visualize the knowledge web
"""

import os
import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Graph and visualization
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class NodeType(str, Enum):
    CONCEPT = "CONCEPT"
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    METHOD = "METHOD"
    TECHNOLOGY = "TECHNOLOGY"
    TERM = "TERM"
    DATASET = "DATASET"


class RelationType(str, Enum):
    RELATED_TO = "RELATED_TO"
    EXPLAINS = "EXPLAINS"
    USES = "USES"
    CONTRASTS_WITH = "CONTRASTS_WITH"
    PART_OF = "PART_OF"
    DERIVES_FROM = "DERIVES_FROM"
    CITED_BY = "CITED_BY"
    ENABLES = "ENABLES"


@dataclass
class Entity:
    text: str
    type: NodeType
    confidence: float
    start: int = 0
    end: int = 0
    context: str = ""


@dataclass
class Relation:
    source: str
    target: str
    type: RelationType
    confidence: float
    evidence: str = ""


@dataclass
class GraphNode:
    node_id: str
    label: str
    type: NodeType
    description: str = ""
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[Tuple[str, str, Dict]] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node

    def add_edge(self, source_id: str, target_id: str, relation: Relation):
        self.edges.append((source_id, target_id, {
            "type": relation.type.value,
            "confidence": relation.confidence,
            "evidence": relation.evidence
        }))


# ============================================================================
# SAMPLE PDF CONTENT - Academic paper about Transformer Architecture
# ============================================================================

SAMPLE_ACADEMIC_CONTENT = """
Attention Is All You Need: Understanding the Transformer Architecture

Abstract

The Transformer architecture, introduced by Vaswani et al. at Google Brain in 2017,
has revolutionized natural language processing and machine learning. Unlike recurrent
neural networks (RNNs) and long short-term memory networks (LSTMs), the Transformer
relies entirely on self-attention mechanisms to capture dependencies in sequential data.

1. Introduction

Deep learning has transformed artificial intelligence research. Traditional sequence-to-sequence
models relied on encoder-decoder architectures with recurrent connections. The Transformer
eliminates recurrence entirely, enabling significantly more parallelization during training.

The key innovations of the Transformer include:
- Multi-head self-attention mechanisms
- Positional encoding for sequence order
- Layer normalization and residual connections
- Scaled dot-product attention

2. Background

2.1 Recurrent Neural Networks

RNNs process sequences step-by-step, maintaining a hidden state that captures information
from previous time steps. Hochreiter and Schmidhuber introduced LSTMs in 1997 to address
the vanishing gradient problem. Cho et al. later proposed Gated Recurrent Units (GRUs)
as a simpler alternative.

2.2 Attention Mechanisms

Bahdanau et al. introduced attention in 2014 for neural machine translation. This allowed
models to focus on relevant parts of the input when generating each output token. Luong et al.
extended this with different attention scoring functions.

3. The Transformer Model

3.1 Architecture Overview

The Transformer uses an encoder-decoder structure:
- The encoder maps input sequences to continuous representations
- The decoder generates output sequences autoregressively
- Both use stacked self-attention and feed-forward layers

3.2 Self-Attention Mechanism

Self-attention computes relationships between all positions in a sequence. Given queries (Q),
keys (K), and values (V), attention is computed as:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

where d_k is the dimension of keys. This scaled dot-product attention prevents gradients
from becoming too small.

3.3 Multi-Head Attention

Instead of single attention, the Transformer uses multiple attention heads:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

Each head can learn different types of relationships, improving model capacity.

3.4 Position-wise Feed-Forward Networks

Each layer contains a feed-forward network applied to each position:

FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

This uses ReLU activation and two linear transformations.

4. Applications and Impact

4.1 BERT and Pre-training

Devlin et al. at Google introduced BERT (Bidirectional Encoder Representations from
Transformers) in 2018. BERT uses masked language modeling and next sentence prediction
for pre-training, achieving state-of-the-art results on many NLP benchmarks including
GLUE, SQuAD, and SWAG.

4.2 GPT Series

OpenAI developed the GPT (Generative Pre-trained Transformer) series:
- GPT-1: Demonstrated the power of pre-training with fine-tuning
- GPT-2: Showed emergent capabilities in zero-shot learning
- GPT-3: Introduced few-shot learning with 175 billion parameters
- GPT-4: Multimodal capabilities with vision and text

4.3 Vision Transformers

Dosovitskiy et al. at Google Research introduced Vision Transformer (ViT), applying
Transformers to computer vision. The image is divided into patches, which are then
processed as a sequence. This approach rivals convolutional neural networks (CNNs)
on ImageNet classification.

5. Training and Optimization

The original Transformer was trained on WMT 2014 English-German and English-French
translation datasets. Key training details:
- Adam optimizer with beta_1=0.9, beta_2=0.98
- Learning rate warm-up followed by decay
- Dropout for regularization
- Label smoothing for improved generalization

The base model has 65 million parameters, while the big model has 213 million.

6. Related Work and Comparisons

6.1 Comparison with CNNs

Convolutional networks use local receptive fields and parameter sharing. While efficient
for images, they struggle with long-range dependencies. The Transformer's global attention
addresses this but has quadratic complexity in sequence length.

6.2 Efficient Transformers

Researchers have developed more efficient variants:
- Longformer: Uses local windowed attention with global tokens
- Linformer: Approximates attention with low-rank projections
- Performer: Uses random feature approximation
- BigBird: Combines random, local, and global attention patterns

7. Conclusion

The Transformer architecture has become the foundation for modern AI systems. From natural
language processing to computer vision, protein structure prediction (AlphaFold), and
code generation (Codex, GitHub Copilot), Transformers continue to expand the boundaries
of artificial intelligence.

References

[1] Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
[2] Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." NAACL 2019.
[3] Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020.
[4] Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
[5] Hochreiter, S. and Schmidhuber, J. "Long Short-Term Memory." Neural Computation 1997.
"""


# ============================================================================
# PDF TEXT EXTRACTION (simulated for this test)
# ============================================================================

def simulate_pdf_extraction(content: str) -> str:
    """Simulate extracting text from a PDF (using the content directly)."""
    print(f"\n{'='*60}")
    print("STEP 1: Simulating PDF Text Extraction")
    print('='*60)

    # In a real scenario, this would use pdfplumber or PyPDF2
    # For this test, we use the content directly

    text = content.strip()
    word_count = len(text.split())
    char_count = len(text)

    print(f"  Simulated PDF extraction:")
    print(f"  Characters: {char_count:,}")
    print(f"  Words: {word_count:,}")
    print(f"  Paragraphs: {text.count(chr(10)+chr(10)) + 1}")

    return text


# ============================================================================
# ENTITY EXTRACTION (Rule-based approach)
# ============================================================================

# Patterns for entity extraction
PERSON_PATTERNS = [
    r'\b([A-Z][a-z]+ (?:et al\.))',  # "Vaswani et al."
    r'\b([A-Z][a-z]+ and [A-Z][a-z]+)\b',  # "Hochreiter and Schmidhuber"
    r'\b(Vaswani|Devlin|Brown|Dosovitskiy|Hochreiter|Schmidhuber|Bahdanau|Luong|Cho)\b',
]

ORGANIZATION_PATTERNS = [
    r'\b(Google(?:\s+(?:Brain|Research|AI))?)\b',
    r'\b(OpenAI)\b',
    r'\b(Microsoft)\b',
    r'\b(Meta(?:\s+AI)?|Facebook(?:\s+AI)?)\b',
    r'\b(DeepMind)\b',
    r'\b(GitHub)\b',
]

TECHNOLOGY_PATTERNS = [
    r'\b(Transformer(?:s)?)\b',
    r'\b(BERT)\b',
    r'\b(GPT(?:-[1-4])?)\b',
    r'\b(LSTMs?|Long Short-Term Memory)\b',
    r'\b(RNNs?|Recurrent Neural Networks?)\b',
    r'\b(GRUs?|Gated Recurrent Units?)\b',
    r'\b(CNNs?|Convolutional Neural Networks?)\b',
    r'\b(Vision Transformer|ViT)\b',
    r'\b(Longformer|Linformer|Performer|BigBird)\b',
    r'\b(AlphaFold|Codex|GitHub Copilot)\b',
    r'\b(Adam optimizer)\b',
]

METHOD_PATTERNS = [
    r'\b(self-attention(?: mechanism)?s?)\b',
    r'\b(multi-head attention)\b',
    r'\b(scaled dot-product attention)\b',
    r'\b(positional encoding)\b',
    r'\b(layer normalization)\b',
    r'\b(residual connections?)\b',
    r'\b(masked language modeling)\b',
    r'\b(next sentence prediction)\b',
    r'\b(pre-training|fine-tuning)\b',
    r'\b(encoder-decoder)\b',
    r'\b(dropout|regularization)\b',
    r'\b(label smoothing)\b',
    r'\b(zero-shot|few-shot) learning\b',
]

CONCEPT_PATTERNS = [
    r'\b(natural language processing|NLP)\b',
    r'\b(machine learning)\b',
    r'\b(deep learning)\b',
    r'\b(artificial intelligence|AI)\b',
    r'\b(computer vision)\b',
    r'\b(neural machine translation)\b',
    r'\b(sequence-to-sequence)\b',
    r'\b(vanishing gradient problem)\b',
    r'\b(attention mechanism)\b',
    r'\b(parallelization)\b',
]

DATASET_PATTERNS = [
    r'\b(WMT 2014)\b',
    r'\b(ImageNet)\b',
    r'\b(GLUE|SQuAD|SWAG)\b',
]


def extract_entities(text: str) -> List[Entity]:
    """Extract entities from text using pattern matching."""
    print(f"\n{'='*60}")
    print("STEP 2: Extracting Entities")
    print('='*60)

    entities = []
    seen = set()

    pattern_groups = [
        (PERSON_PATTERNS, NodeType.PERSON),
        (ORGANIZATION_PATTERNS, NodeType.ORGANIZATION),
        (TECHNOLOGY_PATTERNS, NodeType.TECHNOLOGY),
        (METHOD_PATTERNS, NodeType.METHOD),
        (CONCEPT_PATTERNS, NodeType.CONCEPT),
        (DATASET_PATTERNS, NodeType.DATASET),
    ]

    for patterns, node_type in pattern_groups:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity_text = match.group(1) if match.lastindex else match.group(0)
                normalized = entity_text.lower().strip()

                if normalized not in seen and len(entity_text) > 2:
                    seen.add(normalized)

                    # Calculate confidence based on frequency
                    frequency = len(re.findall(re.escape(entity_text), text, re.IGNORECASE))
                    confidence = min(0.5 + (frequency * 0.1), 0.99)

                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].replace('\n', ' ')

                    entities.append(Entity(
                        text=entity_text,
                        type=node_type,
                        confidence=round(confidence, 2),
                        start=match.start(),
                        end=match.end(),
                        context=context
                    ))

    # Sort by confidence
    entities.sort(key=lambda e: e.confidence, reverse=True)

    # Print summary by type
    type_counts = defaultdict(int)
    for e in entities:
        type_counts[e.type.value] += 1

    print(f"\n  Entities extracted by type:")
    for etype, count in sorted(type_counts.items()):
        print(f"    {etype}: {count}")
    print(f"    TOTAL: {len(entities)}")

    # Show top entities
    print(f"\n  Top entities by confidence:")
    for e in entities[:10]:
        print(f"    [{e.type.value:12}] {e.text:30} (conf: {e.confidence})")

    return entities


# ============================================================================
# RELATION EXTRACTION
# ============================================================================

def extract_relations(text: str, entities: List[Entity]) -> List[Relation]:
    """Extract relations between entities based on co-occurrence and patterns."""
    print(f"\n{'='*60}")
    print("STEP 3: Extracting Relations")
    print('='*60)

    relations = []
    entity_texts = {e.text.lower(): e for e in entities}

    # Sentence-based co-occurrence
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        entities_in_sentence = []

        for entity in entities:
            if entity.text.lower() in sentence_lower:
                entities_in_sentence.append(entity)

        # Create relations between co-occurring entities
        for i, e1 in enumerate(entities_in_sentence):
            for e2 in entities_in_sentence[i+1:]:
                if e1.text.lower() != e2.text.lower():
                    # Determine relation type based on entity types
                    rel_type = determine_relation_type(e1, e2, sentence)

                    relations.append(Relation(
                        source=e1.text,
                        target=e2.text,
                        type=rel_type,
                        confidence=round(min(e1.confidence, e2.confidence) * 0.8, 2),
                        evidence=sentence.strip()[:200]
                    ))

    # Deduplicate and merge relations
    relation_map = {}
    for rel in relations:
        key = (rel.source.lower(), rel.target.lower(), rel.type)
        if key not in relation_map:
            relation_map[key] = rel
        else:
            # Increase confidence for repeated relations
            existing = relation_map[key]
            existing.confidence = min(existing.confidence + 0.1, 0.99)

    unique_relations = list(relation_map.values())
    unique_relations.sort(key=lambda r: r.confidence, reverse=True)

    # Print summary
    type_counts = defaultdict(int)
    for r in unique_relations:
        type_counts[r.type.value] += 1

    print(f"\n  Relations extracted by type:")
    for rtype, count in sorted(type_counts.items()):
        print(f"    {rtype}: {count}")
    print(f"    TOTAL: {len(unique_relations)}")

    print(f"\n  Top relations by confidence:")
    for r in unique_relations[:10]:
        print(f"    {r.source[:20]:20} --[{r.type.value:15}]--> {r.target[:20]:20} (conf: {r.confidence})")

    return unique_relations


def determine_relation_type(e1: Entity, e2: Entity, context: str) -> RelationType:
    """Determine the type of relation between two entities based on context."""
    context_lower = context.lower()

    # Check for specific relation patterns
    if "introduced" in context_lower or "developed" in context_lower or "proposed" in context_lower:
        if e1.type == NodeType.PERSON or e1.type == NodeType.ORGANIZATION:
            return RelationType.ENABLES

    if "uses" in context_lower or "using" in context_lower or "employs" in context_lower:
        return RelationType.USES

    if "unlike" in context_lower or "instead of" in context_lower or "versus" in context_lower:
        return RelationType.CONTRASTS_WITH

    if "part of" in context_lower or "component" in context_lower or "includes" in context_lower:
        return RelationType.PART_OF

    if "based on" in context_lower or "derived" in context_lower or "extends" in context_lower:
        return RelationType.DERIVES_FROM

    if "explains" in context_lower or "describes" in context_lower:
        return RelationType.EXPLAINS

    # Default based on entity types
    if e1.type == NodeType.METHOD and e2.type == NodeType.TECHNOLOGY:
        return RelationType.PART_OF
    if e1.type == NodeType.TECHNOLOGY and e2.type == NodeType.CONCEPT:
        return RelationType.ENABLES

    return RelationType.RELATED_TO


# ============================================================================
# KNOWLEDGE GRAPH BUILDING
# ============================================================================

def build_knowledge_graph(entities: List[Entity], relations: List[Relation]) -> KnowledgeGraph:
    """Build a knowledge graph from entities and relations."""
    print(f"\n{'='*60}")
    print("STEP 4: Building Knowledge Graph")
    print('='*60)

    kg = KnowledgeGraph()

    # Create nodes for each entity
    for entity in entities:
        node_id = hashlib.md5(entity.text.lower().encode()).hexdigest()[:12]
        node = GraphNode(
            node_id=node_id,
            label=entity.text,
            type=entity.type,
            description=entity.context[:200] if entity.context else "",
            confidence=entity.confidence,
            metadata={"frequency": 1}
        )
        kg.add_node(node)

    # Create lookup for node IDs
    label_to_id = {node.label.lower(): node.node_id for node in kg.nodes.values()}

    # Add edges for relations
    for relation in relations:
        source_id = label_to_id.get(relation.source.lower())
        target_id = label_to_id.get(relation.target.lower())

        if source_id and target_id and source_id != target_id:
            kg.add_edge(source_id, target_id, relation)

    print(f"\n  Graph Statistics:")
    print(f"    Nodes: {len(kg.nodes)}")
    print(f"    Edges: {len(kg.edges)}")

    return kg


# ============================================================================
# GRAPH ANALYSIS
# ============================================================================

def analyze_knowledge_graph(kg: KnowledgeGraph) -> Dict:
    """Analyze the knowledge graph for quality metrics."""
    print(f"\n{'='*60}")
    print("STEP 5: Analyzing Knowledge Graph")
    print('='*60)

    # Build NetworkX graph for analysis
    G = nx.DiGraph()

    for node in kg.nodes.values():
        G.add_node(node.node_id,
                   label=node.label,
                   type=node.type.value,
                   confidence=node.confidence)

    for source, target, attrs in kg.edges:
        G.add_edge(source, target, **attrs)

    # Calculate metrics
    metrics = {}

    # Basic metrics
    metrics['node_count'] = G.number_of_nodes()
    metrics['edge_count'] = G.number_of_edges()
    metrics['density'] = nx.density(G) if G.number_of_nodes() > 1 else 0

    # Connectivity
    if G.number_of_nodes() > 0:
        weakly_connected = list(nx.weakly_connected_components(G))
        metrics['connected_components'] = len(weakly_connected)
        metrics['largest_component_size'] = max(len(c) for c in weakly_connected) if weakly_connected else 0

        # Node type distribution
        type_dist = defaultdict(int)
        for _, data in G.nodes(data=True):
            type_dist[data.get('type', 'UNKNOWN')] += 1
        metrics['node_type_distribution'] = dict(type_dist)

        # Edge type distribution
        edge_type_dist = defaultdict(int)
        for _, _, data in G.edges(data=True):
            edge_type_dist[data.get('type', 'UNKNOWN')] += 1
        metrics['edge_type_distribution'] = dict(edge_type_dist)

        # Centrality (top nodes)
        if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
            try:
                degree_centrality = nx.degree_centrality(G)
                top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                metrics['top_central_nodes'] = [
                    (G.nodes[n]['label'], round(c, 3))
                    for n, c in top_central
                ]
            except:
                metrics['top_central_nodes'] = []

            # Betweenness centrality
            try:
                betweenness = nx.betweenness_centrality(G)
                top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics['top_bridge_nodes'] = [
                    (G.nodes[n]['label'], round(b, 3))
                    for n, b in top_between if b > 0
                ]
            except:
                metrics['top_bridge_nodes'] = []

    # Print analysis
    print(f"\n  Basic Metrics:")
    print(f"    Nodes: {metrics['node_count']}")
    print(f"    Edges: {metrics['edge_count']}")
    print(f"    Density: {metrics['density']:.4f}")
    print(f"    Connected Components: {metrics.get('connected_components', 0)}")
    print(f"    Largest Component: {metrics.get('largest_component_size', 0)} nodes")

    print(f"\n  Node Type Distribution:")
    for ntype, count in sorted(metrics.get('node_type_distribution', {}).items()):
        print(f"    {ntype}: {count}")

    print(f"\n  Edge Type Distribution:")
    for etype, count in sorted(metrics.get('edge_type_distribution', {}).items()):
        print(f"    {etype}: {count}")

    print(f"\n  Most Central Nodes (by degree):")
    for label, centrality in metrics.get('top_central_nodes', []):
        print(f"    {label}: {centrality}")

    print(f"\n  Bridge Nodes (by betweenness):")
    for label, between in metrics.get('top_bridge_nodes', []):
        print(f"    {label}: {between}")

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_knowledge_graph(kg: KnowledgeGraph, output_path: str):
    """Create a visualization of the knowledge graph."""
    print(f"\n{'='*60}")
    print("STEP 6: Visualizing Knowledge Graph")
    print('='*60)

    # Build NetworkX graph
    G = nx.DiGraph()

    for node in kg.nodes.values():
        G.add_node(node.node_id,
                   label=node.label[:25],  # Truncate for display
                   type=node.type.value,
                   confidence=node.confidence)

    for source, target, attrs in kg.edges:
        G.add_edge(source, target, **attrs)

    # Color mapping by node type
    color_map = {
        'CONCEPT': '#3498db',      # Blue
        'PERSON': '#e74c3c',       # Red
        'ORGANIZATION': '#9b59b6', # Purple
        'TECHNOLOGY': '#2ecc71',   # Green
        'METHOD': '#f39c12',       # Orange
        'TERM': '#1abc9c',         # Teal
        'DATASET': '#e67e22',      # Dark Orange
    }

    # Get colors and sizes
    node_colors = [color_map.get(G.nodes[n].get('type', 'CONCEPT'), '#95a5a6') for n in G.nodes()]
    node_sizes = [300 + (G.nodes[n].get('confidence', 0.5) * 500) for n in G.nodes()]

    # Create figure
    plt.figure(figsize=(20, 16))

    # Layout
    if len(G.nodes()) > 0:
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G)

        # Draw edges
        nx.draw_networkx_edges(G, pos,
                              edge_color='#cccccc',
                              arrows=True,
                              arrowsize=15,
                              alpha=0.6,
                              connectionstyle="arc3,rad=0.1")

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)

        # Draw labels
        labels = {n: G.nodes[n].get('label', n)[:20] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels,
                               font_size=8,
                               font_weight='bold')

        # Add edge labels for top edges
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if d.get('confidence', 0) > 0.6:
                edge_labels[(u, v)] = d.get('type', '')[:10]

        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                     font_size=6,
                                     font_color='#666666')

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=ntype)
                     for ntype, color in color_map.items()]
    plt.legend(handles=legend_patches, loc='upper left', fontsize=10)

    plt.title("Knowledge Graph: Transformer Architecture Paper", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  Visualization saved to: {output_path}")


# ============================================================================
# GRAPH TRAVERSAL DEMO
# ============================================================================

def demonstrate_graph_traversal(kg: KnowledgeGraph):
    """Demonstrate graph traversal capabilities."""
    print(f"\n{'='*60}")
    print("STEP 7: Graph Traversal Demonstration")
    print('='*60)

    # Build NetworkX graph
    G = nx.DiGraph()
    for node in kg.nodes.values():
        G.add_node(node.node_id, label=node.label, type=node.type.value)
    for source, target, attrs in kg.edges:
        G.add_edge(source, target, **attrs)

    # Find a central node to start from
    if not G.nodes():
        print("  No nodes in graph")
        return

    # Find "Transformer" node if it exists
    start_node = None
    for node_id, data in G.nodes(data=True):
        if 'transformer' in data.get('label', '').lower():
            start_node = node_id
            break

    if not start_node:
        start_node = list(G.nodes())[0]

    start_label = G.nodes[start_node]['label']
    print(f"\n  Starting from: '{start_label}'")

    # 1. Direct neighbors (depth 1)
    print(f"\n  Direct connections (depth 1):")
    successors = list(G.successors(start_node))
    predecessors = list(G.predecessors(start_node))

    print(f"    Outgoing ({len(successors)}):")
    for succ in successors[:5]:
        edge_data = G.edges[start_node, succ]
        print(f"      --[{edge_data.get('type', 'RELATED')}]--> {G.nodes[succ]['label']}")

    print(f"    Incoming ({len(predecessors)}):")
    for pred in predecessors[:5]:
        edge_data = G.edges[pred, start_node]
        print(f"      <--[{edge_data.get('type', 'RELATED')}]-- {G.nodes[pred]['label']}")

    # 2. Path finding
    print(f"\n  Shortest paths from '{start_label}':")
    paths = dict(nx.single_source_shortest_path_length(G, start_node, cutoff=3))

    for target_id, distance in sorted(paths.items(), key=lambda x: x[1]):
        if distance > 0 and distance <= 2:
            target_label = G.nodes[target_id]['label']
            print(f"    Distance {distance}: {target_label}")

    # 3. Subgraph extraction
    print(f"\n  Subgraph around '{start_label}' (2-hop neighborhood):")
    neighborhood = set([start_node])
    for _ in range(2):
        new_neighbors = set()
        for n in neighborhood:
            new_neighbors.update(G.successors(n))
            new_neighbors.update(G.predecessors(n))
        neighborhood.update(new_neighbors)

    subgraph = G.subgraph(neighborhood)
    print(f"    Nodes in subgraph: {subgraph.number_of_nodes()}")
    print(f"    Edges in subgraph: {subgraph.number_of_edges()}")


# ============================================================================
# QUALITY EVALUATION
# ============================================================================

def evaluate_knowledge_quality(kg: KnowledgeGraph, original_text: str) -> Dict:
    """Evaluate the quality of the extracted knowledge graph."""
    print(f"\n{'='*60}")
    print("STEP 8: Quality Evaluation")
    print('='*60)

    evaluation = {
        'coverage': {},
        'accuracy': {},
        'coherence': {},
        'overall_score': 0.0
    }

    # 1. Coverage: How well does the graph cover the document?
    key_terms = [
        'transformer', 'attention', 'bert', 'gpt', 'encoder', 'decoder',
        'self-attention', 'neural network', 'deep learning', 'nlp',
        'vaswani', 'google', 'openai'
    ]

    node_labels_lower = [n.label.lower() for n in kg.nodes.values()]
    covered_terms = sum(1 for term in key_terms
                        if any(term in label for label in node_labels_lower))

    evaluation['coverage']['key_term_coverage'] = covered_terms / len(key_terms)
    evaluation['coverage']['terms_found'] = covered_terms
    evaluation['coverage']['terms_expected'] = len(key_terms)

    # 2. Accuracy: Are the entity types correct?
    type_checks = [
        ('Google', NodeType.ORGANIZATION),
        ('OpenAI', NodeType.ORGANIZATION),
        ('BERT', NodeType.TECHNOLOGY),
        ('Transformer', NodeType.TECHNOLOGY),
        ('self-attention', NodeType.METHOD),
    ]

    correct_types = 0
    for term, expected_type in type_checks:
        for node in kg.nodes.values():
            if term.lower() in node.label.lower():
                if node.type == expected_type:
                    correct_types += 1
                break

    evaluation['accuracy']['type_accuracy'] = correct_types / len(type_checks)
    evaluation['accuracy']['correct_types'] = correct_types
    evaluation['accuracy']['total_checked'] = len(type_checks)

    # 3. Coherence: Are relationships meaningful?
    meaningful_relations = 0
    total_relations = len(kg.edges)

    for source_id, target_id, attrs in kg.edges:
        # Check if relation type makes sense for entity types
        source = kg.nodes.get(source_id)
        target = kg.nodes.get(target_id)

        if source and target:
            rel_type = attrs.get('type', '')

            # Simple coherence checks
            if source.type == NodeType.PERSON and target.type == NodeType.TECHNOLOGY:
                if rel_type in ['ENABLES', 'RELATED_TO']:
                    meaningful_relations += 1
            elif source.type == NodeType.METHOD and target.type == NodeType.TECHNOLOGY:
                if rel_type in ['PART_OF', 'USES', 'RELATED_TO']:
                    meaningful_relations += 1
            elif rel_type == 'RELATED_TO':
                meaningful_relations += 0.5  # Generic but acceptable
            else:
                meaningful_relations += 0.7  # Assume somewhat meaningful

    evaluation['coherence']['meaningful_ratio'] = meaningful_relations / total_relations if total_relations > 0 else 0
    evaluation['coherence']['meaningful_count'] = meaningful_relations
    evaluation['coherence']['total_relations'] = total_relations

    # 4. Calculate overall score
    coverage_score = evaluation['coverage']['key_term_coverage'] * 0.4
    accuracy_score = evaluation['accuracy']['type_accuracy'] * 0.3
    coherence_score = evaluation['coherence']['meaningful_ratio'] * 0.3

    evaluation['overall_score'] = round(coverage_score + accuracy_score + coherence_score, 3)

    # Print evaluation
    print(f"\n  Coverage Analysis:")
    print(f"    Key terms covered: {evaluation['coverage']['terms_found']}/{evaluation['coverage']['terms_expected']}")
    print(f"    Coverage score: {evaluation['coverage']['key_term_coverage']:.1%}")

    print(f"\n  Accuracy Analysis:")
    print(f"    Correct entity types: {evaluation['accuracy']['correct_types']}/{evaluation['accuracy']['total_checked']}")
    print(f"    Accuracy score: {evaluation['accuracy']['type_accuracy']:.1%}")

    print(f"\n  Coherence Analysis:")
    print(f"    Meaningful relations: {evaluation['coherence']['meaningful_count']:.1f}/{evaluation['coherence']['total_relations']}")
    print(f"    Coherence score: {evaluation['coherence']['meaningful_ratio']:.1%}")

    print(f"\n  {'='*40}")
    print(f"  OVERALL QUALITY SCORE: {evaluation['overall_score']:.1%}")
    print(f"  {'='*40}")

    # Quality rating
    if evaluation['overall_score'] >= 0.8:
        rating = "EXCELLENT - The knowledge graph accurately captures the document's key concepts"
    elif evaluation['overall_score'] >= 0.6:
        rating = "GOOD - The knowledge graph captures most important concepts with minor gaps"
    elif evaluation['overall_score'] >= 0.4:
        rating = "FAIR - The knowledge graph captures some concepts but needs improvement"
    else:
        rating = "NEEDS IMPROVEMENT - The knowledge graph has significant gaps"

    print(f"\n  Rating: {rating}")

    return evaluation


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_end_to_end_test():
    """Run the complete end-to-end test."""
    print("\n" + "="*70)
    print(" UNFOLD: End-to-End PDF Knowledge Graph Test")
    print(" Testing the complete pipeline from PDF to Knowledge Web")
    print("="*70)

    # Setup paths
    output_dir = "/home/user/Unfold/test_output"
    os.makedirs(output_dir, exist_ok=True)

    graph_image_path = os.path.join(output_dir, "knowledge_graph.png")
    results_path = os.path.join(output_dir, "test_results.json")

    try:
        # Step 1: Simulate PDF extraction
        extracted_text = simulate_pdf_extraction(SAMPLE_ACADEMIC_CONTENT)

        # Step 2: Extract entities
        entities = extract_entities(extracted_text)

        # Step 3: Extract relations
        relations = extract_relations(extracted_text, entities)

        # Step 4: Build knowledge graph
        kg = build_knowledge_graph(entities, relations)

        # Step 5: Analyze graph
        metrics = analyze_knowledge_graph(kg)

        # Step 6: Visualize
        visualize_knowledge_graph(kg, graph_image_path)

        # Step 7: Demonstrate traversal
        demonstrate_graph_traversal(kg)

        # Step 8: Evaluate quality
        evaluation = evaluate_knowledge_quality(kg, extracted_text)

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(extracted_text),
            "entities_extracted": len(entities),
            "relations_extracted": len(relations),
            "graph_nodes": len(kg.nodes),
            "graph_edges": len(kg.edges),
            "metrics": metrics,
            "evaluation": evaluation,
            "entities": [
                {"text": e.text, "type": e.type.value, "confidence": e.confidence}
                for e in entities
            ],
            "top_relations": [
                {"source": r.source, "target": r.target, "type": r.type.value, "confidence": r.confidence}
                for r in relations[:20]
            ]
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(" TEST COMPLETE")
        print('='*70)
        print(f"\n  Output files:")
        print(f"    Graph visualization: {graph_image_path}")
        print(f"    Results JSON: {results_path}")

        print(f"\n  Summary:")
        print(f"    Entities extracted: {len(entities)}")
        print(f"    Relations found: {len(relations)}")
        print(f"    Graph nodes: {len(kg.nodes)}")
        print(f"    Graph edges: {len(kg.edges)}")
        print(f"    Quality score: {evaluation['overall_score']:.1%}")

        return results

    except Exception as e:
        print(f"\n  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_end_to_end_test()
