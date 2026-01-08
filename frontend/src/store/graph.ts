import { create } from "zustand";
import type {
  GraphNode,
  GraphVisualizationData,
  GraphVisualizationNode,
  GraphVisualizationLink,
} from "@/types";
import { api } from "@/services/api";

interface GraphState {
  // Graph data
  nodes: GraphVisualizationNode[];
  links: GraphVisualizationLink[];
  isLoading: boolean;
  error: string | null;

  // Selection
  selectedNodeId: string | null;
  hoveredNodeId: string | null;

  // Visualization settings
  zoomLevel: number;
  centerPosition: { x: number; y: number };

  // Actions
  loadGraphForDocument: (docId: string) => Promise<void>;
  loadRelatedNodes: (nodeId: string, depth?: number) => Promise<void>;
  selectNode: (nodeId: string | null) => void;
  setHoveredNode: (nodeId: string | null) => void;
  setZoom: (level: number) => void;
  setCenter: (x: number, y: number) => void;
  clearGraph: () => void;
  addNodes: (nodes: GraphNode[]) => void;
  setGraphData: (data: GraphVisualizationData) => void;
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  links: [],
  isLoading: false,
  error: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  zoomLevel: 1,
  centerPosition: { x: 0, y: 0 },

  loadGraphForDocument: async (docId: string) => {
    set({ isLoading: true, error: null });
    try {
      const result = await api.searchNodes({ sourceDocId: docId, limit: 100 });
      const nodes: GraphVisualizationNode[] = result.nodes.map((node) => ({
        ...node,
        x: undefined,
        y: undefined,
        fx: null,
        fy: null,
      }));

      // For now, we'll create links based on related nodes
      // In a full implementation, we'd fetch relations from the backend
      const links: GraphVisualizationLink[] = [];

      set({
        nodes,
        links,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to load graph",
        isLoading: false,
      });
    }
  },

  loadRelatedNodes: async (nodeId: string, depth = 1) => {
    set({ isLoading: true });
    try {
      const result = await api.getRelatedNodes(nodeId, {
        maxDepth: depth,
        limit: 50,
      });

      const existingNodeIds = new Set(get().nodes.map((n) => n.node_id));
      const newNodes: GraphVisualizationNode[] = result.nodes
        .filter((node) => !existingNodeIds.has(node.node_id))
        .map((node) => ({
          ...node,
          x: undefined,
          y: undefined,
          fx: null,
          fy: null,
        }));

      // Create links from the central node to new nodes
      const newLinks: GraphVisualizationLink[] = newNodes.map((node) => ({
        source: nodeId,
        target: node.node_id,
        type: "RELATED_TO" as const,
        weight: 0.5,
      }));

      set((state) => ({
        nodes: [...state.nodes, ...newNodes],
        links: [...state.links, ...newLinks],
        isLoading: false,
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to load related nodes",
        isLoading: false,
      });
    }
  },

  selectNode: (nodeId: string | null) => {
    set({ selectedNodeId: nodeId });
  },

  setHoveredNode: (nodeId: string | null) => {
    set({ hoveredNodeId: nodeId });
  },

  setZoom: (level: number) => {
    set({ zoomLevel: Math.max(0.1, Math.min(4, level)) });
  },

  setCenter: (x: number, y: number) => {
    set({ centerPosition: { x, y } });
  },

  clearGraph: () => {
    set({
      nodes: [],
      links: [],
      selectedNodeId: null,
      hoveredNodeId: null,
      error: null,
    });
  },

  addNodes: (nodes: GraphNode[]) => {
    const existingNodeIds = new Set(get().nodes.map((n) => n.node_id));
    const newNodes: GraphVisualizationNode[] = nodes
      .filter((node) => !existingNodeIds.has(node.node_id))
      .map((node) => ({
        ...node,
        x: undefined,
        y: undefined,
        fx: null,
        fy: null,
      }));

    set((state) => ({
      nodes: [...state.nodes, ...newNodes],
    }));
  },

  setGraphData: (data: GraphVisualizationData) => {
    set({
      nodes: data.nodes,
      links: data.links,
    });
  },
}));
