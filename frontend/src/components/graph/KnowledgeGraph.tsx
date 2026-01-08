"use client";

import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import { useGraphStore } from "@/store";
import { cn } from "@/utils/cn";
import type { GraphVisualizationNode, GraphVisualizationLink, NodeType } from "@/types";
import { Loader2, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";

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

export function KnowledgeGraph({ documentId, className }: KnowledgeGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const {
    nodes,
    links,
    isLoading,
    selectedNodeId,
    hoveredNodeId,
    zoomLevel,
    loadGraphForDocument,
    selectNode,
    setHoveredNode,
    setZoom,
    loadRelatedNodes,
  } = useGraphStore();

  // Load graph data when documentId changes
  useEffect(() => {
    if (documentId) {
      loadGraphForDocument(documentId);
    }
  }, [documentId, loadGraphForDocument]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight || 500;

    // Clear previous content
    svg.selectAll("*").remove();

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
        selectNode(d.node_id === selectedNodeId ? null : d.node_id);
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
  }, [nodes, links, selectedNodeId, selectNode, setHoveredNode, setZoom, loadRelatedNodes]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as unknown as (
          selection: d3.Selection<SVGSVGElement, unknown, null, undefined>,
          k: number
        ) => void,
        1.5
      );
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as unknown as (
          selection: d3.Selection<SVGSVGElement, unknown, null, undefined>,
          k: number
        ) => void,
        0.67
      );
    }
  }, []);

  const handleReset = useCallback(() => {
    if (svgRef.current && containerRef.current) {
      const svg = d3.select(svgRef.current);
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight || 500;

      svg.transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform as unknown as (
          selection: d3.Selection<SVGSVGElement, unknown, null, undefined>,
          transform: d3.ZoomTransform
        ) => void,
        d3.zoomIdentity.translate(width / 2, height / 2).scale(1)
      );
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

      {/* Graph SVG */}
      <svg ref={svgRef} className="w-full h-full min-h-[500px]" />

      {/* Empty state */}
      {nodes.length === 0 && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-slate-400 dark:text-slate-500 mb-2">
              No graph data available
            </div>
            <div className="text-sm text-slate-500 dark:text-slate-400">
              Upload a document to generate a knowledge graph
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
