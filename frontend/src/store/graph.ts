import { create } from "zustand";
import type {
  GraphNode,
  GraphVisualizationData,
  GraphVisualizationNode,
  GraphVisualizationLink,
} from "@/types";
import { api, getErrorMessage } from "@/services/api";

type DocumentRelation = Awaited<
  ReturnType<typeof api.getDocumentRelations>
>["relations"][number];

interface GraphState {
  // Graph data
  nodes: GraphVisualizationNode[];
  links: GraphVisualizationLink[];
  isLoading: boolean;
  error: string | null;
  // Document whose graph is shown (null: all of the user's documents / none)
  currentDocId: string | null;

  // Building a document's graph on demand
  isBuilding: boolean;
  buildError: string | null;

  // Expanding a node's neighbours (kept apart from the initial load so a
  // failure or spinner does not replace the whole graph)
  isExpanding: boolean;
  expandError: string | null;

  // Selection
  selectedNodeId: string | null;
  hoveredNodeId: string | null;

  // Visualization settings
  zoomLevel: number;
  centerPosition: { x: number; y: number };

  // Actions
  loadGraphForDocument: (docId: string) => Promise<void>;
  loadGraphForAllDocuments: () => Promise<void>;
  buildGraphForDocument: (docId: string) => Promise<void>;
  loadRelatedNodes: (nodeId: string, depth?: number) => Promise<void>;
  selectNode: (nodeId: string | null) => void;
  setHoveredNode: (nodeId: string | null) => void;
  setZoom: (level: number) => void;
  setCenter: (x: number, y: number) => void;
  clearGraph: () => void;
  addNodes: (nodes: GraphNode[]) => void;
  setGraphData: (data: GraphVisualizationData) => void;
}

// Incremented by every load/clear; responses from an older load are dropped so
// a slow request cannot overwrite the graph of the document now selected.
let loadSeq = 0;

function toGraphData(
  graphNodes: GraphNode[],
  relations: DocumentRelation[]
): GraphVisualizationData {
  const nodes: GraphVisualizationNode[] = graphNodes.map((node) => ({
    ...node,
    x: undefined,
    y: undefined,
    fx: null,
    fy: null,
  }));

  // D3's forceLink throws on links whose endpoints are not in the node set
  const nodeIds = new Set(nodes.map((n) => n.node_id));
  const links: GraphVisualizationLink[] = relations
    .filter((r) => nodeIds.has(r.source_node_id) && nodeIds.has(r.target_node_id))
    .map((r) => ({
      source: r.source_node_id,
      target: r.target_node_id,
      type: r.type as GraphVisualizationLink["type"],
      weight: r.weight,
    }));

  return { nodes, links };
}

// State at the start of every load: the previous graph and selection are dropped
function resetForLoad(currentDocId: string | null): Partial<GraphState> {
  return {
    currentDocId,
    isLoading: true,
    error: null,
    nodes: [],
    links: [],
    selectedNodeId: null,
    hoveredNodeId: null,
    buildError: null,
    isExpanding: false,
    expandError: null,
  };
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  links: [],
  isLoading: false,
  error: null,
  currentDocId: null,
  isBuilding: false,
  buildError: null,
  isExpanding: false,
  expandError: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  zoomLevel: 1,
  centerPosition: { x: 0, y: 0 },

  loadGraphForDocument: async (docId: string) => {
    const seq = ++loadSeq;
    set(resetForLoad(docId));

    // Relations are optional: if only they fail, still show the nodes
    const [nodesResult, relationsResult] = await Promise.allSettled([
      api.searchNodes({ sourceDocId: docId, limit: 100 }),
      api.getDocumentRelations(docId),
    ]);
    if (seq !== loadSeq) return;

    if (nodesResult.status === "rejected") {
      set({ error: getErrorMessage(nodesResult.reason), isLoading: false });
      return;
    }
    const relations =
      relationsResult.status === "fulfilled" ? relationsResult.value.relations : [];
    set({ ...toGraphData(nodesResult.value.nodes, relations), isLoading: false });
  },

  loadGraphForAllDocuments: async () => {
    const seq = ++loadSeq;
    set(resetForLoad(null));

    try {
      // Without source_doc_id the backend returns nodes of the caller's documents
      const { nodes } = await api.searchNodes({ limit: 200 });
      // Relations are served per document
      const docIds = Array.from(new Set(nodes.map((n) => n.source_doc_id)));
      const relationResults = await Promise.allSettled(
        docIds.map((id) => api.getDocumentRelations(id))
      );
      if (seq !== loadSeq) return;

      const relations = relationResults.flatMap((r) =>
        r.status === "fulfilled" ? r.value.relations : []
      );
      set({ ...toGraphData(nodes, relations), isLoading: false });
    } catch (error) {
      if (seq !== loadSeq) return;
      set({ error: getErrorMessage(error), isLoading: false });
    }
  },

  buildGraphForDocument: async (docId: string) => {
    set({ isBuilding: true, buildError: null });
    try {
      const result = await api.buildDocumentGraph(docId);
      set({ isBuilding: false });
      // The user may have switched documents while the build was running
      if (get().currentDocId !== docId) return;
      if (result.nodes_created === 0) {
        set({
          buildError:
            result.errors[0] || "No concepts could be extracted from this document.",
        });
        return;
      }
      await get().loadGraphForDocument(docId);
    } catch (error) {
      set({ isBuilding: false, buildError: getErrorMessage(error) });
    }
  },

  loadRelatedNodes: async (nodeId: string, depth = 1) => {
    const seq = loadSeq;
    set({ isExpanding: true, expandError: null });
    try {
      const result = await api.getRelatedNodes(nodeId, {
        maxDepth: depth,
        limit: 50,
      });
      // Drop the result if the graph was reloaded or cleared meanwhile
      if (seq !== loadSeq) return;

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
        isExpanding: false,
      }));
    } catch (error) {
      if (seq !== loadSeq) return;
      set({ expandError: getErrorMessage(error), isExpanding: false });
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
    loadSeq++;
    set({
      nodes: [],
      links: [],
      currentDocId: null,
      isLoading: false,
      selectedNodeId: null,
      hoveredNodeId: null,
      error: null,
      buildError: null,
      isExpanding: false,
      expandError: null,
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
