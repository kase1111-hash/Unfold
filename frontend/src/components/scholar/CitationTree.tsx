"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";
import { useScholarStore } from "@/store";
import { cn } from "@/utils/cn";
import type { CitationNode, CitationEdge } from "@/types";
import { Loader2, ZoomIn, ZoomOut, Maximize2, Search } from "lucide-react";

interface CitationTreeProps {
  className?: string;
}

interface VisualizationNode extends CitationNode {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

const DEPTH_COLORS = ["#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626"];

export function CitationTree({ className }: CitationTreeProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [searchDoi, setSearchDoi] = useState("");
  const [hoveredNode, setHoveredNode] = useState<CitationNode | null>(null);

  const { citationTree, citationLoading, citationError, buildCitationTree, clearCitationTree } =
    useScholarStore();

  // Build citation tree
  const handleSearch = useCallback(async () => {
    if (searchDoi.trim()) {
      await buildCitationTree(searchDoi.trim());
    }
  }, [searchDoi, buildCitationTree]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !citationTree) return;

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
      });

    svg.call(zoom);

    // Create main group for zoom/pan
    const g = svg.append("g");

    // Convert nodes to visualization format
    const nodes: VisualizationNode[] = citationTree.nodes.map((node) => ({
      ...node,
      x: undefined,
      y: undefined,
      fx: null,
      fy: null,
    }));

    const nodeMap = new Map(nodes.map((n) => [n.paper_id, n]));

    // Create links
    const links: Array<{ source: VisualizationNode; target: VisualizationNode; type: string }> = [];
    for (const edge of citationTree.edges) {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (source && target) {
        links.push({ source, target, type: edge.type });
      }
    }

    // Create simulation
    const simulation = d3
      .forceSimulation<VisualizationNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<VisualizationNode, { source: VisualizationNode; target: VisualizationNode; type: string }>(links)
          .id((d) => d.paper_id)
          .distance(150)
      )
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(50));

    // Create arrow markers
    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .append("path")
      .attr("d", "M 0,-5 L 10,0 L 0,5")
      .attr("fill", "#94a3b8");

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
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)");

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
          .drag<SVGGElement, VisualizationNode>()
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
      .attr("r", (d) => (d.paper_id === citationTree.root.paper_id ? 16 : 12))
      .attr("fill", (d) => DEPTH_COLORS[Math.min(d.depth, DEPTH_COLORS.length - 1)])
      .attr("stroke", "#fff")
      .attr("stroke-width", 2)
      .on("mouseenter", (event, d) => setHoveredNode(d))
      .on("mouseleave", () => setHoveredNode(null));

    // Citation count badges
    node
      .filter((d) => d.citation_count > 0)
      .append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("fill", "#fff")
      .attr("font-weight", "bold")
      .text((d) => (d.citation_count > 999 ? "999+" : d.citation_count.toString()));

    // Node labels
    node
      .append("text")
      .attr("dy", 25)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "#334155")
      .text((d) => {
        const title = d.title || "Unknown";
        return title.length > 25 ? title.slice(0, 25) + "..." : title;
      });

    // Update positions on tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as VisualizationNode).x || 0)
        .attr("y1", (d) => (d.source as VisualizationNode).y || 0)
        .attr("x2", (d) => (d.target as VisualizationNode).x || 0)
        .attr("y2", (d) => (d.target as VisualizationNode).y || 0);

      node.attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [citationTree]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      const zoom = d3.zoom<SVGSVGElement, unknown>();
      const transition = svg.transition();
      (zoom.scaleBy as Function)(transition, 1.5);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current) {
      const svg = d3.select(svgRef.current);
      const zoom = d3.zoom<SVGSVGElement, unknown>();
      const transition = svg.transition();
      (zoom.scaleBy as Function)(transition, 0.67);
    }
  }, []);

  const handleReset = useCallback(() => {
    if (svgRef.current && containerRef.current) {
      const svg = d3.select(svgRef.current);
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight || 500;
      const zoom = d3.zoom<SVGSVGElement, unknown>();
      const transition = svg.transition();
      const transform = d3.zoomIdentity.translate(width / 2, height / 2).scale(1);
      (zoom.transform as Function)(transition, transform);
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden",
        className
      )}
    >
      {/* Search bar */}
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        <input
          type="text"
          value={searchDoi}
          onChange={(e) => setSearchDoi(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Enter DOI (e.g., 10.1234/example)"
          className="px-3 py-2 text-sm bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg w-64 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        />
        <button
          onClick={handleSearch}
          disabled={citationLoading}
          className="px-3 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50 transition-colors"
        >
          {citationLoading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Search className="w-4 h-4" />
          )}
        </button>
      </div>

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
          Citation Depth
        </span>
        <div className="flex flex-col gap-1">
          {DEPTH_COLORS.map((color, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-slate-600 dark:text-slate-400">
                {i === 0 ? "Root" : `Depth ${i}`}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Stats */}
      {citationTree && (
        <div className="absolute bottom-4 right-4 bg-white/90 dark:bg-slate-800/90 p-3 rounded-lg shadow-sm border border-slate-200 dark:border-slate-600 z-10">
          <span className="text-xs font-medium text-slate-600 dark:text-slate-300 mb-2 block">
            Statistics
          </span>
          <div className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
            <div>Papers: {citationTree.stats.total_nodes}</div>
            <div>Connections: {citationTree.stats.total_edges}</div>
            <div>Max depth: {citationTree.stats.max_depth}</div>
          </div>
        </div>
      )}

      {/* Hover tooltip */}
      {hoveredNode && (
        <div className="absolute top-16 left-4 bg-white dark:bg-slate-800 p-3 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 z-10 max-w-sm">
          <div className="font-medium text-slate-900 dark:text-white text-sm">
            {hoveredNode.title}
          </div>
          {hoveredNode.authors && hoveredNode.authors.length > 0 && (
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {hoveredNode.authors.slice(0, 3).join(", ")}
              {hoveredNode.authors.length > 3 && " et al."}
            </div>
          )}
          <div className="flex gap-3 mt-2 text-xs text-slate-600 dark:text-slate-300">
            {hoveredNode.year && <span>Year: {hoveredNode.year}</span>}
            <span>Citations: {hoveredNode.citation_count}</span>
          </div>
          {hoveredNode.venue && (
            <div className="text-xs text-slate-400 mt-1">{hoveredNode.venue}</div>
          )}
          {hoveredNode.doi && (
            <a
              href={`https://doi.org/${hoveredNode.doi}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-primary-500 hover:underline mt-2 block"
            >
              View paper
            </a>
          )}
        </div>
      )}

      {/* Graph SVG */}
      <svg ref={svgRef} className="w-full h-full min-h-[500px]" />

      {/* Loading state */}
      {citationLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-slate-800/80">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
            <span className="text-slate-500 dark:text-slate-400">Building citation tree...</span>
          </div>
        </div>
      )}

      {/* Error state */}
      {citationError && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center p-4">
            <div className="text-red-500 mb-2">Error: {citationError}</div>
            <button
              onClick={clearCitationTree}
              className="text-sm text-primary-500 hover:underline"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!citationTree && !citationLoading && !citationError && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-slate-400 dark:text-slate-500 mb-2">
              Enter a DOI to build a citation tree
            </div>
            <div className="text-sm text-slate-500 dark:text-slate-400">
              Visualize how papers cite and reference each other
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
