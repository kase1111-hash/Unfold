import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from "axios";
import type {
  ApiError,
  Document,
  GraphNode,
  PaginatedResponse,
  User,
  CitationTree,
  CitationNode,
  CredibilityScore,
  ReadingSnapshot,
  LearningJourney,
  Annotation,
} from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// New auth response type (refresh token is in httpOnly cookie)
interface AuthResponse {
  user: User;
  access_token: string;
  token_type: string;
  expires_in: number;
}

// Access token response for refresh
interface AccessTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

class ApiClient {
  private client: AxiosInstance;
  private accessToken: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        "Content-Type": "application/json",
      },
      // Enable cookies for cross-origin requests (needed for httpOnly refresh token)
      withCredentials: true,
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & {
          _retry?: boolean;
        };

        // If 401 and not already retrying, try to refresh
        if (
          error.response?.status === 401 &&
          !originalRequest._retry &&
          originalRequest.url !== "/auth/refresh" &&
          originalRequest.url !== "/auth/login"
        ) {
          originalRequest._retry = true;

          try {
            // Refresh token is sent automatically via httpOnly cookie
            const newToken = await this.refreshTokens();
            this.setAccessToken(newToken.access_token);

            // Retry original request
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${newToken.access_token}`;
            }
            return this.client(originalRequest);
          } catch (refreshError) {
            // Refresh failed, clear tokens
            this.clearTokens();
            throw refreshError;
          }
        }

        return Promise.reject(error);
      }
    );

    // Load access token from localStorage on init
    if (typeof window !== "undefined") {
      this.loadAccessTokenFromStorage();
    }
  }

  // Token management - only access token in localStorage now
  setAccessToken(token: string) {
    this.accessToken = token;

    if (typeof window !== "undefined") {
      localStorage.setItem("access_token", token);
    }
  }

  clearTokens() {
    this.accessToken = null;

    if (typeof window !== "undefined") {
      localStorage.removeItem("access_token");
    }
  }

  private loadAccessTokenFromStorage() {
    this.accessToken = localStorage.getItem("access_token");
  }

  isAuthenticated(): boolean {
    return !!this.accessToken;
  }

  // Auth endpoints
  async register(
    email: string,
    username: string,
    password: string,
    fullName?: string
  ): Promise<{ user: User }> {
    const response = await this.client.post<AuthResponse>("/auth/register", {
      email,
      username,
      password,
      full_name: fullName,
    });
    // Refresh token is automatically set as httpOnly cookie by the server
    this.setAccessToken(response.data.access_token);
    return { user: response.data.user };
  }

  async login(email: string, password: string): Promise<{ user: User }> {
    const response = await this.client.post<AuthResponse>("/auth/login", {
      email,
      password,
    });
    // Refresh token is automatically set as httpOnly cookie by the server
    this.setAccessToken(response.data.access_token);
    return { user: response.data.user };
  }

  async logout(): Promise<void> {
    try {
      // Tell server to clear the refresh token cookie
      await this.client.post("/auth/logout");
    } catch {
      // Ignore errors, still clear local tokens
    }
    this.clearTokens();
  }

  async refreshTokens(): Promise<AccessTokenResponse> {
    // Refresh token is sent automatically via httpOnly cookie
    const response = await this.client.post<AccessTokenResponse>("/auth/refresh", {});
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>("/auth/me");
    return response.data;
  }

  // Document endpoints
  async uploadDocument(file: File): Promise<Document> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await this.client.post<{ document: Document }>(
      "/documents/upload",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data.document;
  }

  async getDocuments(
    page = 1,
    pageSize = 20
  ): Promise<PaginatedResponse<Document>> {
    const response = await this.client.get<PaginatedResponse<Document>>(
      "/documents",
      {
        params: { page, page_size: pageSize },
      }
    );
    return response.data;
  }

  async getDocument(docId: string): Promise<Document> {
    const response = await this.client.get<Document>(`/documents/${docId}`);
    return response.data;
  }

  async deleteDocument(docId: string): Promise<void> {
    await this.client.delete(`/documents/${docId}`);
  }

  async getDocumentParaphrase(
    docId: string,
    complexity: number
  ): Promise<{ content: string }> {
    const response = await this.client.get<{ content: string }>(
      `/documents/${docId}/paraphrase`,
      {
        params: { complexity },
      }
    );
    return response.data;
  }

  // Knowledge Graph endpoints
  async buildGraph(
    text: string,
    sourceDocId: string,
    options?: {
      extractRelations?: boolean;
      generateEmbeddings?: boolean;
    }
  ): Promise<{
    nodes_created: number;
    relations_created: number;
    node_ids: string[];
  }> {
    const response = await this.client.post("/graph/build", {
      text,
      source_doc_id: sourceDocId,
      extract_relations: options?.extractRelations ?? true,
      generate_embeddings: options?.generateEmbeddings ?? true,
    });
    return response.data;
  }

  async searchNodes(params?: {
    query?: string;
    nodeType?: string;
    sourceDocId?: string;
    limit?: number;
  }): Promise<{ nodes: GraphNode[]; total: number }> {
    const response = await this.client.get<{ nodes: GraphNode[]; total: number }>(
      "/graph/nodes",
      { params }
    );
    return response.data;
  }

  async getNode(nodeId: string): Promise<GraphNode> {
    const response = await this.client.get<GraphNode>(`/graph/nodes/${nodeId}`);
    return response.data;
  }

  async getRelatedNodes(
    nodeId: string,
    options?: {
      relationTypes?: string[];
      maxDepth?: number;
      limit?: number;
    }
  ): Promise<{ nodes: GraphNode[]; total: number }> {
    const response = await this.client.get<{ nodes: GraphNode[]; total: number }>(
      `/graph/nodes/${nodeId}/related`,
      { params: options }
    );
    return response.data;
  }

  async linkToWikipedia(entity: string): Promise<{
    entity: string;
    title: string | null;
    url: string | null;
    extract: string | null;
    found: boolean;
  }> {
    const response = await this.client.get(`/graph/link/wikipedia/${encodeURIComponent(entity)}`);
    return response.data;
  }

  async searchPapers(
    query: string,
    limit = 10
  ): Promise<
    Array<{
      paper_id: string;
      title: string;
      abstract?: string;
      year?: number;
      citation_count?: number;
      authors: string[];
      url?: string;
    }>
  > {
    const response = await this.client.get("/graph/link/papers", {
      params: { query, limit },
    });
    return response.data;
  }

  // Health check
  async getHealth(): Promise<{
    status: string;
    version: string;
    environment: string;
  }> {
    const response = await this.client.get("/health");
    return response.data;
  }

  // Scholar Mode endpoints (Phase 5)

  // Citation Tree
  async buildCitationTree(
    doi: string,
    options?: {
      maxDepth?: number;
      refsPerLevel?: number;
      citesPerLevel?: number;
    }
  ): Promise<CitationTree> {
    const response = await this.client.post<CitationTree>("/scholar/citations/tree", {
      doi,
      max_depth: options?.maxDepth ?? 2,
      refs_per_level: options?.refsPerLevel ?? 10,
      cites_per_level: options?.citesPerLevel ?? 10,
    });
    return response.data;
  }

  async getPaperByDoi(doi: string): Promise<CitationNode> {
    const response = await this.client.get<CitationNode>(
      `/scholar/citations/paper/${encodeURIComponent(doi)}`
    );
    return response.data;
  }

  async getPaperReferences(
    doi: string,
    limit = 20
  ): Promise<{ doi: string; references: CitationNode[]; count: number }> {
    const response = await this.client.get(
      `/scholar/citations/references/${encodeURIComponent(doi)}`,
      { params: { limit } }
    );
    return response.data;
  }

  async getPaperCitations(
    doi: string,
    limit = 20
  ): Promise<{ doi: string; citations: CitationNode[]; count: number }> {
    const response = await this.client.get(
      `/scholar/citations/citing/${encodeURIComponent(doi)}`,
      { params: { limit } }
    );
    return response.data;
  }

  // Credibility Scoring
  async getCredibilityScore(doi: string): Promise<CredibilityScore> {
    const response = await this.client.post<CredibilityScore>(
      "/scholar/credibility/score",
      { doi }
    );
    return response.data;
  }

  async compareCredibility(
    dois: string[]
  ): Promise<{ comparisons: Array<{ doi: string; score: CredibilityScore }>; count: number }> {
    const response = await this.client.post("/scholar/credibility/compare", { dois });
    return response.data;
  }

  // Zotero Export
  async exportToZotero(
    items: Array<{
      title: string;
      authors: string[];
      date?: string;
      doi?: string;
      type?: string;
    }>,
    format: "ris" | "bibtex" | "csl-json" = "ris"
  ): Promise<Blob> {
    const response = await this.client.post(
      "/scholar/zotero/export",
      { items, format },
      { responseType: "blob" }
    );
    return response.data;
  }

  async previewZoteroExport(
    items: Array<{
      title: string;
      authors: string[];
      date?: string;
      doi?: string;
      type?: string;
    }>,
    format: "ris" | "bibtex" | "csl-json" = "ris"
  ): Promise<{ format: string; content: string; item_count: number }> {
    const response = await this.client.post("/scholar/zotero/preview", { items, format });
    return response.data;
  }

  // Reflection Engine
  async createSnapshot(
    documentId: string,
    data: {
      reflectionType?: string;
      complexityLevel?: number;
      timeSpentMinutes?: number;
      summary?: string;
      keyTakeaways?: string[];
      questions?: string[];
      connections?: string[];
      highlights?: Array<{ text: string; note?: string; position?: number }>;
      sectionsRead?: string[];
      scrollDepth?: number;
    }
  ): Promise<ReadingSnapshot> {
    const response = await this.client.post<ReadingSnapshot>("/scholar/reflection/snapshot", {
      document_id: documentId,
      reflection_type: data.reflectionType ?? "initial_reading",
      complexity_level: data.complexityLevel ?? 50,
      time_spent_minutes: data.timeSpentMinutes ?? 0,
      summary: data.summary,
      key_takeaways: data.keyTakeaways,
      questions: data.questions,
      connections: data.connections,
      highlights: data.highlights,
      sections_read: data.sectionsRead,
      scroll_depth: data.scrollDepth ?? 0,
    });
    return response.data;
  }

  async getSnapshots(
    documentId: string
  ): Promise<{ document_id: string; snapshots: ReadingSnapshot[]; count: number }> {
    const response = await this.client.get(`/scholar/reflection/snapshots/${documentId}`);
    return response.data;
  }

  async getLearningJourney(documentId: string): Promise<LearningJourney> {
    const response = await this.client.get<LearningJourney>(
      `/scholar/reflection/journey/${documentId}`
    );
    return response.data;
  }

  async getReflectionPrompts(documentId: string): Promise<{ document_id: string; prompts: string[] }> {
    const response = await this.client.get(`/scholar/reflection/prompts/${documentId}`);
    return response.data;
  }

  // Annotations
  async createAnnotation(
    documentId: string,
    data: {
      annotationType?: string;
      content?: string;
      selectedText?: string;
      startOffset?: number;
      endOffset?: number;
      sectionId?: string;
      visibility?: string;
      parentId?: string;
      tags?: string[];
    }
  ): Promise<Annotation> {
    const response = await this.client.post<Annotation>("/scholar/annotations", {
      document_id: documentId,
      annotation_type: data.annotationType ?? "highlight",
      content: data.content ?? "",
      selected_text: data.selectedText,
      start_offset: data.startOffset ?? 0,
      end_offset: data.endOffset ?? 0,
      section_id: data.sectionId,
      visibility: data.visibility ?? "private",
      parent_id: data.parentId,
      tags: data.tags,
    });
    return response.data;
  }

  async getAnnotations(
    documentId: string,
    filters?: {
      sectionId?: string;
      annotationType?: string;
      visibility?: string;
    }
  ): Promise<{ document_id: string; annotations: Annotation[]; count: number }> {
    const response = await this.client.get(`/scholar/annotations/${documentId}`, {
      params: {
        section_id: filters?.sectionId,
        annotation_type: filters?.annotationType,
        visibility: filters?.visibility,
      },
    });
    return response.data;
  }

  async updateAnnotation(
    documentId: string,
    annotationId: string,
    data: {
      content?: string;
      tags?: string[];
      visibility?: string;
    }
  ): Promise<Annotation> {
    const response = await this.client.put<Annotation>(
      `/scholar/annotations/${documentId}/${annotationId}`,
      data
    );
    return response.data;
  }

  async deleteAnnotation(documentId: string, annotationId: string): Promise<void> {
    await this.client.delete(`/scholar/annotations/${documentId}/${annotationId}`);
  }

  async addReaction(
    documentId: string,
    annotationId: string,
    emoji: string
  ): Promise<Annotation> {
    const response = await this.client.post<Annotation>(
      `/scholar/annotations/${documentId}/${annotationId}/reaction`,
      { emoji }
    );
    return response.data;
  }

  async getAnnotationThread(
    documentId: string,
    parentId: string
  ): Promise<{ parent_id: string; replies: Annotation[]; count: number }> {
    const response = await this.client.get(
      `/scholar/annotations/${documentId}/thread/${parentId}`
    );
    return response.data;
  }

  async getAnnotationStats(
    documentId: string
  ): Promise<{ total: number; by_type: Record<string, number>; by_user: Record<string, number>; by_section: Record<string, number> }> {
    const response = await this.client.get(`/scholar/annotations/${documentId}/stats`);
    return response.data;
  }
}

// Export singleton instance
export const api = new ApiClient();

// Helper to extract error message
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const apiError = error.response?.data as { detail?: { message?: string } } | undefined;
    return apiError?.detail?.message || error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "An unexpected error occurred";
}
