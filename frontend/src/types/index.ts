// User types
export interface User {
  user_id: string;
  email: string;
  username: string;
  full_name?: string;
  orcid_id?: string;
  role: UserRole;
  is_active: boolean;
  is_verified: boolean;
  last_login?: string;
  created_at: string;
  updated_at: string;
}

export type UserRole = "user" | "educator" | "researcher" | "admin";

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface AuthResponse {
  user: User;
  tokens: Token;
}

// Document types
export interface Document {
  doc_id: string;
  title: string;
  authors: string[];
  doi?: string;
  abstract?: string;
  license?: DocumentLicense;
  source: DocumentSource;
  status: DocumentStatus;
  vector_id?: string;
  graph_nodes: string[];
  file_path?: string;
  file_size_bytes?: number;
  page_count?: number;
  word_count?: number;
  created_at: string;
  updated_at: string;
}

export type DocumentLicense =
  | "CC-BY-4.0"
  | "CC-BY-SA-4.0"
  | "CC-BY-NC-4.0"
  | "CC0-1.0"
  | "MIT"
  | "unknown"
  | "proprietary";

export type DocumentSource = "upload" | "arXiv" | "PubMed" | "DOI" | "URL";

export type DocumentStatus =
  | "pending"
  | "processing"
  | "validated"
  | "indexed"
  | "failed";

// Knowledge Graph types
export interface GraphNode {
  node_id: string;
  label: string;
  type: NodeType;
  description?: string;
  source_doc_id: string;
  embedding?: number[];
  confidence: number;
  external_links: Record<string, string>;
  metadata?: Record<string, unknown>;
}

export type NodeType =
  | "Concept"
  | "Author"
  | "Paper"
  | "Method"
  | "Dataset"
  | "Institution"
  | "Term";

export type RelationType =
  | "EXPLAINS"
  | "CITES"
  | "CONTRASTS_WITH"
  | "DERIVES_FROM"
  | "AUTHORED_BY"
  | "AFFILIATED_WITH"
  | "USES_METHOD"
  | "USES_DATASET"
  | "RELATED_TO"
  | "PART_OF";

export interface GraphRelation {
  relation_id: string;
  source_node_id: string;
  target_node_id: string;
  type: RelationType;
  weight: number;
  metadata?: Record<string, unknown>;
}

// Reading interface types
export interface ReadingState {
  documentId: string | null;
  complexityLevel: number; // 0-100
  viewMode: "technical" | "conceptual" | "hybrid";
  selectedText: string | null;
  activeNodes: GraphNode[];
  highlights: TextHighlight[];
}

export interface TextHighlight {
  id: string;
  startOffset: number;
  endOffset: number;
  color: string;
  note?: string;
}

// Paraphrase types
export interface ParaphraseRequest {
  text: string;
  complexity: number;
  context?: string;
}

export interface ParaphraseResponse {
  original: string;
  paraphrased: string;
  complexity: number;
  concepts: string[];
}

// API response types
export interface ApiResponse<T> {
  status: "success" | "error";
  data: T;
  meta?: Record<string, unknown>;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  status: "success";
  data: T[];
  pagination: {
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
  };
}

// Graph visualization types
export interface GraphVisualizationNode extends GraphNode {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphVisualizationLink {
  source: string | GraphVisualizationNode;
  target: string | GraphVisualizationNode;
  type: RelationType;
  weight: number;
}

export interface GraphVisualizationData {
  nodes: GraphVisualizationNode[];
  links: GraphVisualizationLink[];
}

// Scholar Mode types (Phase 5)
export interface CitationNode {
  paper_id: string;
  title: string;
  authors: string[];
  year?: number;
  venue?: string;
  citation_count: number;
  abstract?: string;
  doi?: string;
  url?: string;
  depth: number;
}

export interface CitationEdge {
  source: string;
  target: string;
  type: "cites" | "cited_by" | "related";
}

export interface CitationTree {
  root: CitationNode;
  nodes: CitationNode[];
  edges: CitationEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    max_depth: number;
  };
}

export interface CredibilityScore {
  overall_score: number;
  level: "high" | "medium" | "low" | "unknown";
  components: {
    citation_score: number;
    venue_score: number;
    author_score: number;
    recency_score: number;
    altmetric_score: number;
  };
  metadata: {
    citation_count: number;
    journal_impact_factor?: number;
    altmetric_attention?: number;
    is_peer_reviewed: boolean;
    is_retracted: boolean;
  };
  warnings: string[];
  notes: string[];
}

export interface ReadingSnapshot {
  snapshot_id: string;
  document_id: string;
  created_at: string;
  reflection_type: string;
  complexity_level: number;
  comprehension_score: number;
  time_spent_minutes: number;
  summary?: string;
  key_takeaways: string[];
  questions: string[];
  connections: string[];
  highlights_count: number;
  sections_read: number;
  scroll_depth: number;
}

export interface LearningJourney {
  snapshot_count: number;
  first_read: ReadingSnapshot;
  latest_read?: ReadingSnapshot;
  diffs: LearningDiff[];
  summary: {
    total_time_minutes: number;
    time_span_days?: number;
    comprehension_growth?: number;
    current_comprehension: number;
    questions_remaining: number;
    total_takeaways: number;
    total_connections?: number;
    total_highlights?: number;
  };
}

export interface LearningDiff {
  from_snapshot: string;
  to_snapshot: string;
  time_between_hours: number;
  changes: {
    complexity_change: number;
    comprehension_change: number;
    total_time_added: number;
  };
  content: {
    new_takeaways: string[];
    resolved_questions: string[];
    new_questions: string[];
    new_connections: string[];
    new_highlights: number;
  };
  progress: {
    new_sections: number;
    scroll_depth_change: number;
  };
  insights: {
    learning_velocity: number;
    engagement_trend: string;
  };
}

export type AnnotationType =
  | "highlight"
  | "comment"
  | "question"
  | "answer"
  | "link"
  | "tag";

export type AnnotationVisibility = "private" | "group" | "public";

export interface Annotation {
  annotation_id: string;
  document_id: string;
  user_id: string;
  user_name: string;
  type: AnnotationType;
  visibility: AnnotationVisibility;
  content: string;
  selected_text?: string;
  position: {
    start: number;
    end: number;
    section_id?: string;
  };
  created_at: string;
  updated_at: string;
  parent_id?: string;
  tags: string[];
  reactions: Record<string, string[]>;
  is_deleted: boolean;
}
