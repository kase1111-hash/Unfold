import { create } from "zustand";
import type { Document, GraphNode, TextHighlight } from "@/types";
import { api } from "@/services/api";

type ViewMode = "technical" | "conceptual" | "hybrid";

interface ReadingState {
  // Current document
  document: Document | null;
  isLoading: boolean;
  error: string | null;

  // Reading settings
  complexityLevel: number; // 0-100
  viewMode: ViewMode;

  // Paraphrased content
  paraphrasedContent: string | null;
  isParaphrasing: boolean;

  // Selection and interaction
  selectedText: string | null;
  activeNodes: GraphNode[];
  highlights: TextHighlight[];

  // Actions
  loadDocument: (docId: string) => Promise<void>;
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
  isLoading: false,
  error: null,
  complexityLevel: 50,
  viewMode: "hybrid",
  paraphrasedContent: null,
  isParaphrasing: false,
  selectedText: null,
  activeNodes: [],
  highlights: [],

  loadDocument: async (docId: string) => {
    set({ isLoading: true, error: null });
    try {
      const document = await api.getDocument(docId);
      set({
        document,
        isLoading: false,
        paraphrasedContent: null,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to load document",
        isLoading: false,
      });
    }
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

    set({ isParaphrasing: true });
    try {
      const result = await api.getDocumentParaphrase(
        document.doc_id,
        complexityLevel
      );
      set({
        paraphrasedContent: result.content,
        isParaphrasing: false,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to paraphrase",
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
      paraphrasedContent: null,
      selectedText: null,
      activeNodes: [],
      highlights: [],
      error: null,
    });
  },
}));
