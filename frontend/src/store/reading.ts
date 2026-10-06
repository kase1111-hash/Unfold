import { create } from "zustand";
import type { Document, GraphNode, TextHighlight } from "@/types";
import axios from "axios";
import { api, getErrorMessage } from "@/services/api";

type ViewMode = "technical" | "conceptual" | "hybrid";

// Most recently requested document; responses for any other id are dropped
let latestDocId: string | null = null;
// Incremented by every paraphrase request and document change; only the latest
// request may write the result (or clear the spinner)
let paraphraseSeq = 0;

interface ReadingState {
  // Current document
  // Id the last loadDocument call asked for (document/error belong to it)
  requestedDocId: string | null;
  document: Document | null;
  documentContent: string | null;
  isLoading: boolean;
  error: string | null;
  // The document does not exist or belongs to someone else (404)
  notFound: boolean;

  // Reading settings
  complexityLevel: number; // 0-100
  viewMode: ViewMode;

  // Paraphrased content
  paraphrasedContent: string | null;
  isParaphrasing: boolean;
  paraphraseError: string | null;

  // Selection and interaction
  selectedText: string | null;
  activeNodes: GraphNode[];
  highlights: TextHighlight[];

  // Actions
  loadDocument: (docId: string) => Promise<void>;
  // Replace the loaded document's metadata (e.g. a newer status) if it is still open
  updateDocument: (document: Document) => void;
  setComplexity: (level: number) => void;
  setViewMode: (mode: ViewMode) => void;
  fetchParaphrase: () => Promise<void>;
  setSelectedText: (text: string | null) => void;
  setActiveNodes: (nodes: GraphNode[]) => void;
  addHighlight: (highlight: Omit<TextHighlight, "id">) => void;
  removeHighlight: (id: string) => void;
  clearDocument: () => void;
}

export const useReadingStore = create<ReadingState>((set, get) => ({
  requestedDocId: null,
  document: null,
  documentContent: null,
  isLoading: false,
  error: null,
  notFound: false,
  complexityLevel: 50,
  viewMode: "hybrid",
  paraphrasedContent: null,
  isParaphrasing: false,
  paraphraseError: null,
  selectedText: null,
  activeNodes: [],
  highlights: [],

  loadDocument: async (docId: string) => {
    latestDocId = docId;
    // A paraphrase still running for the previous document must not finish here
    paraphraseSeq++;
    set({
      requestedDocId: docId,
      isLoading: true,
      error: null,
      notFound: false,
      document: null,
      documentContent: null,
      paraphrasedContent: null,
      isParaphrasing: false,
      paraphraseError: null,
      selectedText: null,
      activeNodes: [],
      highlights: [],
    });
    try {
      const [document, contentResult] = await Promise.all([
        api.getDocument(docId),
        api.getDocumentContent(docId).catch(() => null),
      ]);
      // Ignore the response if another document was opened meanwhile
      if (latestDocId !== docId) return;
      set({
        document,
        documentContent: contentResult?.content || null,
        isLoading: false,
        paraphrasedContent: null,
      });
    } catch (error) {
      if (latestDocId !== docId) return;
      set({
        error: getErrorMessage(error),
        notFound: axios.isAxiosError(error) && error.response?.status === 404,
        isLoading: false,
      });
    }
  },

  updateDocument: (document: Document) => {
    set((state) =>
      state.document?.doc_id === document.doc_id ? { document } : {}
    );
  },

  setComplexity: (level: number) => {
    set({ complexityLevel: Math.max(0, Math.min(100, level)) });
  },

  setViewMode: (mode: ViewMode) => {
    set({ viewMode: mode });
  },

  fetchParaphrase: async () => {
    const { document, complexityLevel } = get();
    if (!document) return;

    const seq = ++paraphraseSeq;
    set({ isParaphrasing: true, paraphraseError: null });
    try {
      const result = await api.getDocumentParaphrase(
        document.doc_id,
        complexityLevel
      );
      // Superseded (another document was opened, or a newer request started):
      // whoever superseded it owns isParaphrasing now
      if (seq !== paraphraseSeq) return;
      set({
        paraphrasedContent: result.content,
        isParaphrasing: false,
      });
    } catch (error) {
      if (seq !== paraphraseSeq) return;
      // Do NOT write the page-level `error`: that replaces the whole document view.
      set({
        paraphraseError: getErrorMessage(error),
        isParaphrasing: false,
      });
    }
  },

  setSelectedText: (text: string | null) => {
    set({ selectedText: text });
  },

  setActiveNodes: (nodes: GraphNode[]) => {
    set({ activeNodes: nodes });
  },

  addHighlight: (highlight: Omit<TextHighlight, "id">) => {
    const id = `highlight-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    set((state) => ({
      highlights: [...state.highlights, { ...highlight, id }],
    }));
  },

  removeHighlight: (id: string) => {
    set((state) => ({
      highlights: state.highlights.filter((h) => h.id !== id),
    }));
  },

  clearDocument: () => {
    set({
      document: null,
      documentContent: null,
      paraphrasedContent: null,
      selectedText: null,
      activeNodes: [],
      highlights: [],
      error: null,
    });
  },
}));
