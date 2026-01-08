import { create } from "zustand";
import type {
  CitationTree,
  CitationNode,
  CredibilityScore,
  ReadingSnapshot,
  LearningJourney,
  Annotation,
} from "@/types";
import { api } from "@/services/api";

interface ScholarState {
  // Citation Tree
  citationTree: CitationTree | null;
  citationLoading: boolean;
  citationError: string | null;

  // Credibility
  credibilityScores: Map<string, CredibilityScore>;
  credibilityLoading: boolean;

  // Reflection
  snapshots: ReadingSnapshot[];
  journey: LearningJourney | null;
  reflectionPrompts: string[];

  // Annotations
  annotations: Annotation[];
  selectedAnnotation: Annotation | null;
  annotationsLoading: boolean;

  // Actions - Citations
  buildCitationTree: (
    doi: string,
    options?: { maxDepth?: number; refsPerLevel?: number; citesPerLevel?: number }
  ) => Promise<void>;
  clearCitationTree: () => void;

  // Actions - Credibility
  loadCredibilityScore: (doi: string) => Promise<CredibilityScore | null>;
  compareCredibility: (dois: string[]) => Promise<void>;

  // Actions - Reflection
  createSnapshot: (
    documentId: string,
    data: {
      reflectionType?: string;
      complexityLevel?: number;
      timeSpentMinutes?: number;
      summary?: string;
      keyTakeaways?: string[];
      questions?: string[];
      connections?: string[];
      scrollDepth?: number;
    }
  ) => Promise<void>;
  loadSnapshots: (documentId: string) => Promise<void>;
  loadJourney: (documentId: string) => Promise<void>;
  loadPrompts: (documentId: string) => Promise<void>;

  // Actions - Annotations
  loadAnnotations: (documentId: string) => Promise<void>;
  createAnnotation: (
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
  ) => Promise<Annotation | null>;
  updateAnnotation: (
    documentId: string,
    annotationId: string,
    data: { content?: string; tags?: string[]; visibility?: string }
  ) => Promise<void>;
  deleteAnnotation: (documentId: string, annotationId: string) => Promise<void>;
  addReaction: (documentId: string, annotationId: string, emoji: string) => Promise<void>;
  selectAnnotation: (annotation: Annotation | null) => void;
}

export const useScholarStore = create<ScholarState>((set, get) => ({
  // Initial state
  citationTree: null,
  citationLoading: false,
  citationError: null,
  credibilityScores: new Map(),
  credibilityLoading: false,
  snapshots: [],
  journey: null,
  reflectionPrompts: [],
  annotations: [],
  selectedAnnotation: null,
  annotationsLoading: false,

  // Citation Actions
  buildCitationTree: async (doi, options) => {
    set({ citationLoading: true, citationError: null });
    try {
      const tree = await api.buildCitationTree(doi, options);
      set({ citationTree: tree, citationLoading: false });
    } catch (error) {
      set({
        citationError: error instanceof Error ? error.message : "Failed to build citation tree",
        citationLoading: false,
      });
    }
  },

  clearCitationTree: () => {
    set({ citationTree: null, citationError: null });
  },

  // Credibility Actions
  loadCredibilityScore: async (doi) => {
    set({ credibilityLoading: true });
    try {
      const score = await api.getCredibilityScore(doi);
      const scores = new Map(get().credibilityScores);
      scores.set(doi, score);
      set({ credibilityScores: scores, credibilityLoading: false });
      return score;
    } catch (error) {
      set({ credibilityLoading: false });
      return null;
    }
  },

  compareCredibility: async (dois) => {
    set({ credibilityLoading: true });
    try {
      const result = await api.compareCredibility(dois);
      const scores = new Map(get().credibilityScores);
      for (const comparison of result.comparisons) {
        scores.set(comparison.doi, comparison.score);
      }
      set({ credibilityScores: scores, credibilityLoading: false });
    } catch (error) {
      set({ credibilityLoading: false });
    }
  },

  // Reflection Actions
  createSnapshot: async (documentId, data) => {
    try {
      const snapshot = await api.createSnapshot(documentId, data);
      set((state) => ({ snapshots: [...state.snapshots, snapshot] }));
    } catch (error) {
      console.error("Failed to create snapshot:", error);
    }
  },

  loadSnapshots: async (documentId) => {
    try {
      const result = await api.getSnapshots(documentId);
      set({ snapshots: result.snapshots });
    } catch (error) {
      console.error("Failed to load snapshots:", error);
    }
  },

  loadJourney: async (documentId) => {
    try {
      const journey = await api.getLearningJourney(documentId);
      set({ journey });
    } catch (error) {
      console.error("Failed to load journey:", error);
    }
  },

  loadPrompts: async (documentId) => {
    try {
      const result = await api.getReflectionPrompts(documentId);
      set({ reflectionPrompts: result.prompts });
    } catch (error) {
      console.error("Failed to load prompts:", error);
    }
  },

  // Annotation Actions
  loadAnnotations: async (documentId) => {
    set({ annotationsLoading: true });
    try {
      const result = await api.getAnnotations(documentId);
      set({ annotations: result.annotations, annotationsLoading: false });
    } catch (error) {
      set({ annotationsLoading: false });
      console.error("Failed to load annotations:", error);
    }
  },

  createAnnotation: async (documentId, data) => {
    try {
      const annotation = await api.createAnnotation(documentId, data);
      set((state) => ({ annotations: [...state.annotations, annotation] }));
      return annotation;
    } catch (error) {
      console.error("Failed to create annotation:", error);
      return null;
    }
  },

  updateAnnotation: async (documentId, annotationId, data) => {
    try {
      const updated = await api.updateAnnotation(documentId, annotationId, data);
      set((state) => ({
        annotations: state.annotations.map((a) =>
          a.annotation_id === annotationId ? updated : a
        ),
      }));
    } catch (error) {
      console.error("Failed to update annotation:", error);
    }
  },

  deleteAnnotation: async (documentId, annotationId) => {
    try {
      await api.deleteAnnotation(documentId, annotationId);
      set((state) => ({
        annotations: state.annotations.filter((a) => a.annotation_id !== annotationId),
      }));
    } catch (error) {
      console.error("Failed to delete annotation:", error);
    }
  },

  addReaction: async (documentId, annotationId, emoji) => {
    try {
      const updated = await api.addReaction(documentId, annotationId, emoji);
      set((state) => ({
        annotations: state.annotations.map((a) =>
          a.annotation_id === annotationId ? updated : a
        ),
      }));
    } catch (error) {
      console.error("Failed to add reaction:", error);
    }
  },

  selectAnnotation: (annotation) => {
    set({ selectedAnnotation: annotation });
  },
}));
