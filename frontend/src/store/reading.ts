import { create } from "zustand";
import type { Document, GraphNode, TextHighlight } from "@/types";
import { api, getErrorMessage } from "@/services/api";

type ViewMode = "technical" | "conceptual" | "hybrid";

// Most recently requested document; responses for any other id are dropped
let latestDocId: string | null = null;

interface ReadingState {
  // Current document
  document: Document | null;
  documentContent: string | null;
  isLoading: boolean;
  error: string | null;

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
  document: null,
  documentContent: null,
  isLoading: false,
  error: null,
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
    set({
      isLoading: true,
      error: null,
      document: null,
      documentContent: null,
      paraphrasedContent: null,
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

    set({ isParaphrasing: true, paraphraseError: null });
    try {
      const result = await api.getDocumentParaphrase(
        document.doc_id,
        complexityLevel
      );
      if (get().document?.doc_id !== document.doc_id) return;
      set({
        paraphrasedContent: result.content,
        isParaphrasing: false,
      });
    } catch (error) {
      if (get().document?.doc_id !== document.doc_id) return;
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
