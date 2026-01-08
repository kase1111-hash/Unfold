import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from "axios";
import type {
  ApiError,
  AuthResponse,
  Document,
  GraphNode,
  PaginatedResponse,
  Token,
  User,
} from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

class ApiClient {
  private client: AxiosInstance;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      headers: {
        "Content-Type": "application/json",
      },
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

        // If 401 and we have refresh token, try to refresh
        if (
          error.response?.status === 401 &&
          this.refreshToken &&
          !originalRequest._retry
        ) {
          originalRequest._retry = true;

          try {
            const tokens = await this.refreshTokens();
            this.setTokens(tokens);

            // Retry original request
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${tokens.access_token}`;
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

    // Load tokens from localStorage on init
    if (typeof window !== "undefined") {
      this.loadTokensFromStorage();
    }
  }

  // Token management
  setTokens(tokens: Token) {
    this.accessToken = tokens.access_token;
    this.refreshToken = tokens.refresh_token;

    if (typeof window !== "undefined") {
      localStorage.setItem("access_token", tokens.access_token);
      localStorage.setItem("refresh_token", tokens.refresh_token);
    }
  }

  clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;

    if (typeof window !== "undefined") {
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
    }
  }

  private loadTokensFromStorage() {
    this.accessToken = localStorage.getItem("access_token");
    this.refreshToken = localStorage.getItem("refresh_token");
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
  ): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>("/auth/register", {
      email,
      username,
      password,
      full_name: fullName,
    });
    this.setTokens(response.data.tokens);
    return response.data;
  }

  async login(email: string, password: string): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>("/auth/login", {
      email,
      password,
    });
    this.setTokens(response.data.tokens);
    return response.data;
  }

  async logout(): Promise<void> {
    this.clearTokens();
  }

  async refreshTokens(): Promise<Token> {
    const response = await this.client.post<Token>("/auth/refresh", {
      refresh_token: this.refreshToken,
    });
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
}

// Export singleton instance
export const api = new ApiClient();

// Helper to extract error message
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const apiError = error.response?.data as { error?: ApiError } | undefined;
    return apiError?.error?.message || error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "An unexpected error occurred";
}
