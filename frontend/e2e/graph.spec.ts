import { test, expect, makeNode, graphCircles, graphLines, appAlerts } from "./fixtures";

test.describe("Knowledge graph page", () => {
  test("draws the selected document's nodes and edges", async ({ page, api }) => {
    await page.goto("/graph");

    await expect(page.getByRole("heading", { name: "Knowledge Graph" })).toBeVisible();
    await expect(page.getByLabel("Document")).toHaveValue("doc-1");
    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
    await expect(page.locator('[data-testid="knowledge-graph"] g.nodes text')).toHaveText([
      "Marie Curie",
      "Radioactivity",
      "Polonium",
    ]);

    // snake_case filter params, so only this document's nodes come back
    const nodesCall = api.callsTo("GET", "/graph/nodes").at(-1)!;
    expect(nodesCall.query.get("source_doc_id")).toBe("doc-1");
    expect(nodesCall.query.get("limit")).toBe("100");
    expect(nodesCall.query.has("sourceDocId")).toBe(false);
    expect(api.callsTo("GET", "/graph/documents/doc-1/relations")).not.toHaveLength(0);
  });

  test("selecting a node shows its details without redrawing the graph", async ({ page }) => {
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);
    await page.locator('[data-testid="knowledge-graph"] g.nodes').evaluate((el) =>
      el.setAttribute("data-e2e-marker", "original")
    );

    await graphCircles(page).nth(1).dispatchEvent("click");

    await expect(page.getByRole("heading", { level: 3, name: "Radioactivity" })).toBeVisible();
    await expect(page.locator('[data-e2e-marker="original"]')).toHaveCount(1);
  });

  test("zoom buttons zoom the drawn graph", async ({ page }) => {
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);

    await page.getByTitle("Zoom in").click();

    await expect(page.locator('[data-testid="knowledge-graph"] > g')).toHaveAttribute(
      "transform",
      /scale\(1\.5\)/
    );
  });

  test("still draws the nodes when only the relations request fails", async ({ page, api }) => {
    api.on("GET", "/graph/documents/doc-1/relations", {
      status: 500,
      body: { detail: "Internal Server Error" },
    });
    await page.goto("/graph");

    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(0);
    await expect(appAlerts(page)).toHaveCount(0);
  });

  test("shows the backend message when the graph database is down", async ({ page, api }) => {
    const unavailable = {
      status: 503,
      body: {
        detail: {
          code: "GRAPH_UNAVAILABLE",
          message: "The knowledge graph database is unavailable. Please try again later.",
        },
      },
    };
    api.on("GET", "/graph/nodes", unavailable);
    api.on("GET", "/graph/documents/doc-1/relations", unavailable);
    await page.goto("/graph");

    await expect(appAlerts(page)).toContainText(
      "The knowledge graph database is unavailable. Please try again later."
    );
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
  });

  test("builds the graph of a document that has none yet", async ({ page, api }) => {
    const doc1Nodes = api.data.nodes.filter((n) => n.source_doc_id === "doc-1");
    api.data.nodes.splice(0, api.data.nodes.length, ...api.data.nodes.filter((n) => n.source_doc_id !== "doc-1"));
    api.on("POST", "/graph/documents/doc-1/build", () => {
      api.data.nodes.push(...doc1Nodes);
      return { body: { doc_id: "doc-1", nodes_created: 3, relations_created: 2, errors: [] } };
    });
    await page.goto("/graph");
    await expect(page.getByText("No graph data available")).toBeVisible();
    await expect(graphCircles(page)).toHaveCount(0);

    await page.getByRole("button", { name: "Build knowledge graph" }).click();

    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
    const buildCalls = api.callsTo("POST", "/graph/documents/doc-1/build");
    expect(buildCalls).toHaveLength(1);
    expect(buildCalls[0].body).toBeNull();
  });

  test("explains a build that produced no nodes", async ({ page, api }) => {
    api.data.nodes.splice(0);
    api.on("POST", "/graph/documents/doc-1/build", {
      body: {
        doc_id: "doc-1",
        nodes_created: 0,
        relations_created: 0,
        errors: ["Entity extraction found no concepts"],
      },
    });
    await page.goto("/graph");

    await page.getByRole("button", { name: "Build knowledge graph" }).click();

    await expect(appAlerts(page)).toHaveText("Entity extraction found no concepts");
    await expect(graphCircles(page)).toHaveCount(0);
  });

  test("'All documents' replaces the previous graph with all of the user's nodes", async ({
    page,
    api,
  }) => {
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);

    await page.getByLabel("Document").selectOption("");

    await expect(graphCircles(page)).toHaveCount(4);
    await expect(graphLines(page)).toHaveCount(2);
    const lastNodesCall = api.callsTo("GET", "/graph/nodes").at(-1)!;
    expect(lastNodesCall.query.has("source_doc_id")).toBe(false);
    // Relations are fetched per document found among the nodes
    expect(api.callsTo("GET", "/graph/documents/doc-2/relations").length).toBeGreaterThan(0);

    await page.getByLabel("Document").selectOption("doc-2");

    await expect(graphCircles(page)).toHaveCount(1);
    await expect(page.locator('[data-testid="knowledge-graph"] g.nodes text')).toHaveText([
      "Chlorophyll",
    ]);
  });

  test("double-click expands related nodes without reloading the graph", async ({ page, api }) => {
    api.on("GET", "/graph/nodes/node_polonium/related", {
      body: {
        nodes: [makeNode({ node_id: "node_radium", label: "Radium", type: "Term" })],
        total: 1,
      },
    });
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);

    await graphCircles(page).nth(2).dispatchEvent("dblclick");

    await expect(graphCircles(page)).toHaveCount(4);
    await expect(graphLines(page)).toHaveCount(3);
    const related = api.callsTo("GET", "/graph/nodes/node_polonium/related")[0];
    expect(related.query.get("max_depth")).toBe("1");
    expect(related.query.get("limit")).toBe("50");
  });

  test("a failed expansion keeps the graph and reports the error", async ({ page, api }) => {
    api.on("GET", "/graph/nodes/node_curie/related", {
      status: 404,
      body: { detail: { code: "NOT_FOUND", message: "Node not found" } },
    });
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);

    await graphCircles(page).first().dispatchEvent("dblclick");

    await expect(appAlerts(page)).toHaveText("Could not load related nodes: Node not found");
    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
  });
});

test.describe("Reader graph panel", () => {
  test("shows the open document's graph", async ({ page }) => {
    await page.goto("/read/doc-1");

    await expect(page.getByRole("heading", { name: "Knowledge Graph" })).toBeVisible();
    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
  });
});
