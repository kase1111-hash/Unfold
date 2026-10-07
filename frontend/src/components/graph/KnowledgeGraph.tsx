"use client";

import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { useGraphStore } from "@/store";
import { cn } from "@/utils/cn";
import type { GraphVisualizationNode, GraphVisualizationLink, NodeType } from "@/types";
import { Loader2, ZoomIn, ZoomOut, Maximize2, AlertCircle, RefreshCw } from "lucide-react";
import { ErrorBoundary } from "@/components/ErrorBoundary";

interface KnowledgeGraphProps {
  documentId?: string;
  className?: string;
}

const NODE_COLORS: Record<NodeType, string> = {
  Concept: "#7c3aed",
  Author: "#2563eb",
  Paper: "#059669",
  Method: "#d97706",
  Dataset: "#dc2626",
  Institution: "#0891b2",
  Term: "#64748b",
};

const NODE_RADIUS: Record<NodeType, number> = {
  Concept: 12,
  Author: 10,
  Paper: 14,
  Method: 11,
  Dataset: 11,
  Institution: 13,
  Term: 8,
};

// The boundary must sit ABOVE the component whose effects run D3; a boundary
// rendered by KnowledgeGraphInner itself cannot catch KnowledgeGraphInner's errors.
export function KnowledgeGraph(props: KnowledgeGraphProps) {
  return (
    <ErrorBoundary key={props.documentId ?? "all"}>
      <KnowledgeGraphInner {...props} />
    </ErrorBoundary>
  );
}

function KnowledgeGraphInner({ documentId, className }: KnowledgeGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const selectedNodeIdRef = useRef<string | null>(null);

  const {
    nodes,
    links,
    totalNodes,
    isLoading,
    error,
    selectedNodeId,
    hoveredNodeId,
    zoomLevel,
    buildingDocIds,
    buildError: lastBuildError,
    isExpanding,
    expandError,
    loadGraphForDocument,
    loadGraphForAllDocuments,
    buildGraphForDocument,
    selectNode,
    setHoveredNode,
    setZoom,
    loadRelatedNodes,
  } = useGraphStore();

  selectedNodeIdRef.current = selectedNodeId;

  // Build state of this document only (a build of another one may be running)
  const isBuilding = !!documentId && buildingDocIds.includes(documentId);
  const buildError =
    documentId && lastBuildError?.docId === documentId ? lastBuildError.message : null;

  // Load graph data when documentId changes (no document: all of the user's
  // documents). Each load replaces the previous graph, so nothing stale remains.
  const loadGraph = useCallback(() => {
    if (documentId) {
      loadGraphForDocument(documentId);
    } else {
      loadGraphForAllDocuments();
    }
  }, [documentId, loadGraphForDocument, loadGraphForAllDocuments]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    // Clear previous content (also when the graph became empty)
    svg.selectAll("*").remove();
    if (nodes.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight || 500;

    // Set up SVG
    svg.attr("width", width).attr("height", height);

    // Create zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        setZoom(event.transform.k);
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Create main group for zoom/pan
    const g = svg.append("g");

    // Create simulation
    const simulation = d3
      .forceSimulation<GraphVisualizationNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphVisualizationNode, GraphVisualizationLink>(links)
          .id((d) => d.node_id)
          .distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    // Create links
    const link = g
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "#94a3b8")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", (d) => Math.sqrt(d.weight) * 2);

    // Create link labels
    const linkLabel = g
      .append("g")
      .attr("class", "link-labels")
      .selectAll("text")
      .data(links)
      .enter()
      .append("text")
      .attr("font-size", "8px")
      .attr("fill", "#64748b")
      .attr("text-anchor", "middle")
      .text((d) => d.type.replace(/_/g, " "));

    // Create nodes
    const node = g
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .style("cursor", "pointer")
      .call(
        d3
          .drag<SVGGElement, GraphVisualizationNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Node circles
    node
      .append("circle")
      .attr("r", (d) => NODE_RADIUS[d.type] || 10)
      .attr("fill", (d) => NODE_COLORS[d.type] || "#64748b")
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .on("click", (event, d) => {
        event.stopPropagation();
        selectNode(d.node_id === selectedNodeIdRef.current ? null : d.node_id);
      })
      .on("dblclick", (event, d) => {
        event.stopPropagation();
        loadRelatedNodes(d.node_id);
      })
      .on("mouseenter", (event, d) => {
        setHoveredNode(d.node_id);
      })
      .on("mouseleave", () => {
        setHoveredNode(null);
      });

    // Node labels
    node
      .append("text")
      .attr("dy", (d) => (NODE_RADIUS[d.type] || 10) + 12)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "#334155")
      .text((d) => d.label.length > 20 ? d.label.slice(0, 20) + "..." : d.label);

    // Update positions on tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as GraphVisualizationNode).x || 0)
        .attr("y1", (d) => (d.source as GraphVisualizationNode).y || 0)
        .attr("x2", (d) => (d.target as GraphVisualizationNode).x || 0)
        .attr("y2", (d) => (d.target as GraphVisualizationNode).y || 0);

      linkLabel
        .attr("x", (d) => {
          const source = d.source as GraphVisualizationNode;
          const target = d.target as GraphVisualizationNode;
          return ((source.x || 0) + (target.x || 0)) / 2;
        })
        .attr("y", (d) => {
          const source = d.source as GraphVisualizationNode;
          const target = d.target as GraphVisualizationNode;
          return ((source.y || 0) + (target.y || 0)) / 2;
        });

      node.attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
    });

    // Click on background to deselect
    svg.on("click", () => {
      selectNode(null);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [nodes, links, selectNode, setHoveredNode, setZoom, loadRelatedNodes]);

  // Zoom controls drive the zoom behavior attached in the D3 effect (a fresh
  // d3.zoom() would change the stored transform without moving the graph)
  const handleZoomIn = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      if (zoomRef.current) svg.transition().call(zoomRef.current.scaleBy, 1.5);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      if (zoomRef.current) svg.transition().call(zoomRef.current.scaleBy, 0.67);
    }
  }, []);

  const handleReset = useCallback(() => {
    if (svgRef.current && containerRef.current) {
      const svg = d3.select(svgRef.current);
      if (zoomRef.current) svg.transition().call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  if (isLoading) {
    return (
      <div
        className={cn(
          "flex items-center justify-center h-96 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700",
          className
        )}
      >
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
          <span className="text-slate-500 dark:text-slate-400">
            Loading knowledge graph...
          </span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className={cn(
          "flex items-center justify-center h-96 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700",
          className
        )}
      >
        <div role="alert" className="flex flex-col items-center gap-3 text-center px-6">
          <AlertCircle className="w-8 h-8 text-red-500" />
          {/* Backend message, e.g. 503 GRAPH_UNAVAILABLE when Neo4j is down */}
          <span className="text-slate-600 dark:text-slate-300 text-sm">
            {error}
          </span>
          <button
            onClick={loadGraph}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden",
        className
      )}
    >
      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-white dark:bg-slate-700 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4 text-slate-600 dark:text-slate-300" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-white dark:bg-slate-700 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4 text-slate-600 dark:text-slate-300" />
        </button>
        <button
          onClick={handleReset}
          className="p-2 bg-white dark:bg-slate-700 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-600 transition-colors"
          title="Reset view"
        >
          <Maximize2 className="w-4 h-4 text-slate-600 dark:text-slate-300" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/90 dark:bg-slate-800/90 p-3 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 z-10">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-300 mb-2 block">
          Node Types
        </span>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(NODE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-slate-600 dark:text-slate-400">
                {type}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredNodeId && (
        <div className="absolute top-4 left-4 bg-white dark:bg-slate-800 p-3 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 z-10 max-w-xs">
          {(() => {
            const hoveredNode = nodes.find((n) => n.node_id === hoveredNodeId);
            if (!hoveredNode) return null;
            return (
              <>
                <div className="font-medium text-slate-900 dark:text-white text-sm">
                  {hoveredNode.label}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {hoveredNode.type}
                </div>
                {hoveredNode.description && (
                  <p className="text-xs text-slate-600 dark:text-slate-300 mt-2">
                    {hoveredNode.description}
                  </p>
                )}
                <div className="text-xs text-slate-400 mt-2">
                  Double-click to expand
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* Graph too large to load in full */}
      {totalNodes > nodes.length && nodes.length > 0 && (
        <div
          data-testid="graph-truncated"
          className="absolute bottom-4 left-4 z-10 max-w-xs bg-white dark:bg-slate-800 px-3 py-2 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 text-xs text-slate-600 dark:text-slate-300"
        >
          Showing the {nodes.length.toLocaleString()} most connected of{" "}
          {totalNodes.toLocaleString()} concepts
        </div>
      )}

      {/* Expanding related nodes (does not replace the graph) */}
      {(isExpanding || expandError) && (
        <div className="absolute bottom-4 right-4 z-10 max-w-xs bg-white dark:bg-slate-800 px-3 py-2 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 text-xs">
          {isExpanding ? (
            <span className="flex items-center gap-2 text-slate-600 dark:text-slate-300">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading related nodes...
            </span>
          ) : (
            <span role="alert" className="text-red-600 dark:text-red-400">
              Could not load related nodes: {expandError}
            </span>
          )}
        </div>
      )}

      {/* Graph SVG */}
      <svg
        ref={svgRef}
        data-testid="knowledge-graph"
        className="w-full h-full min-h-[500px]"
      />

      {/* Empty state */}
      {nodes.length === 0 && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center max-w-sm px-6">
            <div className="text-slate-400 dark:text-slate-500 mb-2">
              No graph data available
            </div>
            {documentId ? (
              <>
                <div className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  The knowledge graph is built in the background after upload. If it
                  does not appear, build it now.
                </div>
                <button
                  onClick={() => buildGraphForDocument(documentId)}
                  disabled={isBuilding}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-60 rounded-lg transition-colors"
                >
                  {isBuilding ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <RefreshCw className="w-4 h-4" />
                  )}
                  {isBuilding ? "Building knowledge graph..." : "Build knowledge graph"}
                </button>
                {buildError && (
                  <p role="alert" className="mt-3 text-sm text-red-600 dark:text-red-400">
                    {buildError}
                  </p>
                )}
              </>
            ) : (
              <div className="text-sm text-slate-500 dark:text-slate-400">
                Pick a document to build its knowledge graph
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
