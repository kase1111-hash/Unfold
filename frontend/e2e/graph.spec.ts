import {
  test,
  expect,
  makeNode,
  relation,
  deferred,
  graphCircles,
  graphLines,
  appAlerts,
  readerUrl,
  DOC1,
  DOC2,
  UNHANDLED_500,
} from "./fixtures";

const CURIE_NODE = "node_a1cedb2f2704";
const POLONIUM_NODE = "node_c06fe41637f2";

test.describe("Knowledge graph page", () => {
  test("draws the selected document's nodes and edges", async ({ page, api }) => {
    await page.goto("/graph");

    await expect(page.getByRole("heading", { name: "Knowledge Graph" })).toBeVisible();
    await expect(page.getByLabel("Document")).toHaveValue(DOC1);
    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
    await expect(page.locator('[data-testid="knowledge-graph"] g.nodes text')).toHaveText([
      "Marie Curie",
      "Radioactivity",
      "Polonium",
    ]);

    // snake_case filter params, so only this document's nodes come back
    const nodesCall = api.callsTo("GET", "/graph/nodes").at(-1)!;
    expect(nodesCall.query.get("source_doc_id")).toBe(DOC1);
    expect(nodesCall.query.has("sourceDocId")).toBe(false);
    // The whole graph: the backend maximum, not a silent cut at 100
    expect(nodesCall.query.get("limit")).toBe("1000");
    expect(api.callsTo("GET", `/graph/documents/${DOC1}/relations`).at(-1)!.query.get("limit")).toBe(
      "1000"
    );
  });

  test("shows a large document's whole graph", async ({ page, api }) => {
    // 150 concepts in a chain: every node and every edge must be drawn
    const nodes = Array.from({ length: 150 }, (_, i) =>
      makeNode({ node_id: `node_big_${i}`, label: `Concept ${i}` })
    );
    api.data.nodes.splice(0, api.data.nodes.length, ...nodes);
    api.data.relations[DOC1] = nodes
      .slice(1)
      .map((node, i) => relation(`rel_big_${i}`, nodes[i].node_id, node.node_id));
    await page.goto("/graph");

    await expect(graphCircles(page)).toHaveCount(150);
    await expect(graphLines(page)).toHaveCount(149);
    await expect(page.getByText("Total Nodes").locator("..")).toContainText("150");
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
    api.on("GET", `/graph/documents/${DOC1}/relations`, UNHANDLED_500);
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
    api.on("GET", `/graph/documents/${DOC1}/relations`, unavailable);
    await page.goto("/graph");

    await expect(appAlerts(page)).toContainText(
      "The knowledge graph database is unavailable. Please try again later."
    );
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
  });

  test("builds the graph of a document that has none yet", async ({ page, api }) => {
    const doc1Nodes = api.data.nodes.filter((n) => n.source_doc_id === DOC1);
    api.data.nodes.splice(0, api.data.nodes.length, ...api.data.nodes.filter((n) => n.source_doc_id !== DOC1));
    api.on("POST", `/graph/documents/${DOC1}/build`, () => {
      api.data.nodes.push(...doc1Nodes);
      return { body: { doc_id: DOC1, nodes_created: 3, relations_created: 2, errors: [] } };
    });
    await page.goto("/graph");
    await expect(page.getByText("No graph data available")).toBeVisible();
    await expect(graphCircles(page)).toHaveCount(0);

    await page.getByRole("button", { name: "Build knowledge graph", exact: true }).click();

    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
    const buildCalls = api.callsTo("POST", `/graph/documents/${DOC1}/build`);
    expect(buildCalls).toHaveLength(1);
    expect(buildCalls[0].body).toBeNull();
  });

  test("explains a build that produced no nodes", async ({ page, api }) => {
    api.data.nodes.splice(0);
    api.on("POST", `/graph/documents/${DOC1}/build`, {
      body: {
        doc_id: DOC1,
        nodes_created: 0,
        relations_created: 0,
        errors: ["Entity extraction found no concepts"],
      },
    });
    await page.goto("/graph");

    await page.getByRole("button", { name: "Build knowledge graph", exact: true }).click();

    await expect(appAlerts(page)).toHaveText("Entity extraction found no concepts");
    await expect(graphCircles(page)).toHaveCount(0);
  });

  test("a build keeps to its own document", async ({ page, api }) => {
    api.data.nodes.splice(0); // neither document has a graph
    const buildAnswer = deferred();
    api.on("POST", `/graph/documents/${DOC1}/build`, {
      waitFor: buildAnswer.promise,
      status: 503,
      body: {
        detail: {
          code: "GRAPH_UNAVAILABLE",
          message: "The knowledge graph database is unavailable. Please try again later.",
        },
      },
    });
    await page.goto("/graph");
    await page.getByRole("button", { name: "Build knowledge graph", exact: true }).click();
    await expect(page.getByRole("button", { name: "Building knowledge graph..." })).toBeDisabled();

    await page.getByLabel("Document").selectOption(DOC2);

    // The other document is not "building", and does not get the first one's error
    await expect(page.getByRole("button", { name: "Build knowledge graph", exact: true })).toBeEnabled();
    const buildFailed = page.waitForResponse((r) => r.url().includes("/build"));
    buildAnswer.release();
    await buildFailed;
    await expect(page.getByRole("button", { name: "Build knowledge graph", exact: true })).toBeEnabled();
    await expect(appAlerts(page)).toHaveCount(0);
  });

  test("waits for the selected document's build before drawing its graph", async ({
    page,
    api,
  }) => {
    // DOC1 is being built: some nodes exist, relations do not yet
    const doc = api.data.documents[0];
    doc.status = "processing";
    const allNodes = api.data.nodes.filter((n) => n.source_doc_id === DOC1);
    let polls = 0;
    api.on("GET", `/documents/${DOC1}`, () => {
      polls += 1;
      if (polls >= 2) doc.status = "indexed";
      return { body: doc };
    });
    api.on("GET", "/graph/nodes", (call) => {
      const nodes =
        doc.status === "indexed"
          ? api.data.nodes.filter((n) => n.source_doc_id === call.query.get("source_doc_id"))
          : allNodes.slice(0, 2);
      return { body: { nodes, total: nodes.length } };
    });
    api.on("GET", `/graph/documents/${DOC1}/relations`, () => {
      const relations = doc.status === "indexed" ? api.data.relations[DOC1] : [];
      return { body: { relations, total: relations.length } };
    });
    await page.goto("/graph");

    await expect(page.getByText("Building knowledge graph…")).toBeVisible();
    await expect(page.getByRole("button", { name: "Build knowledge graph", exact: true })).toHaveCount(0);

    await expect(graphCircles(page)).toHaveCount(3, { timeout: 20_000 });
    await expect(graphLines(page)).toHaveCount(2);
    await expect(page.getByText("Total Links").locator("..")).toContainText("2");
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
    expect(lastNodesCall.query.get("limit")).toBe("500");
    // Relations are fetched per document found among the nodes
    expect(api.callsTo("GET", `/graph/documents/${DOC2}/relations`).length).toBeGreaterThan(0);

    await page.getByLabel("Document").selectOption(DOC2);

    await expect(graphCircles(page)).toHaveCount(1);
    await expect(page.locator('[data-testid="knowledge-graph"] g.nodes text')).toHaveText([
      "Chlorophyll",
    ]);
  });

  test("double-click expands related nodes without reloading the graph", async ({ page, api }) => {
    api.on("GET", `/graph/nodes/${POLONIUM_NODE}/related`, {
      body: {
        nodes: [makeNode({ node_id: "node_e5f6a7b8c9d0", label: "Radium", type: "Term" })],
        total: 1,
      },
    });
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);

    await graphCircles(page).nth(2).dispatchEvent("dblclick");

    await expect(graphCircles(page)).toHaveCount(4);
    await expect(graphLines(page)).toHaveCount(3);
    const related = api.callsTo("GET", `/graph/nodes/${POLONIUM_NODE}/related`)[0];
    expect(related.query.get("max_depth")).toBe("1");
    expect(related.query.get("limit")).toBe("50");
  });

  test("a failed expansion keeps the graph and reports the error", async ({ page, api }) => {
    api.on("GET", `/graph/nodes/${CURIE_NODE}/related`, {
      status: 404,
      body: { detail: { code: "NODE_NOT_FOUND", message: `Node ${CURIE_NODE} not found` } },
    });
    await page.goto("/graph");
    await expect(graphCircles(page)).toHaveCount(3);

    await graphCircles(page).first().dispatchEvent("dblclick");

    await expect(appAlerts(page)).toHaveText(
      `Could not load related nodes: Node ${CURIE_NODE} not found`
    );
    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
  });
});

test.describe("Reader graph panel", () => {
  test("shows the open document's graph", async ({ page }) => {
    await page.goto(readerUrl(DOC1));

    await expect(page.getByRole("heading", { name: "Knowledge Graph" })).toBeVisible();
    await expect(graphCircles(page)).toHaveCount(3);
    await expect(graphLines(page)).toHaveCount(2);
  });
});
