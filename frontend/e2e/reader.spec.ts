import {
  test,
  expect,
  appAlerts,
  deferred,
  graphCircles,
  graphLines,
  readerUrl,
  toasts,
  DOC1,
  DOC2,
  RATE_LIMITED,
  type MockApi,
} from "./fixtures";
import type { Page } from "@playwright/test";

const CURIE = "Marie Curie and Radioactivity";

test.describe("Reader", () => {
  test("shows the document text with its line breaks", async ({ page, api }) => {
    await page.goto(readerUrl(DOC1));

    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();
    await expect(page.getByText("Eve Curie")).toBeVisible();
    const original = page.getByText(/She discovered the elements polonium and radium\./).first();
    await expect(original).toBeVisible();
    await expect(original).toHaveCSS("white-space", "pre-wrap");
    await expect(page.getByText("120 words")).toBeVisible();
    expect(api.callsTo("GET", `/documents/${DOC1}/content`)).not.toHaveLength(0);
  });

  test("decodes the document id from the URL before calling the API", async ({ page, api }) => {
    // The upload page links with the id URL-encoded ("sha256%3A...")
    await page.goto(`/read/${encodeURIComponent(DOC1)}`);

    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();
    // The graph filter must carry the real id, not the still-encoded one
    await expect.poll(() => api.callsTo("GET", "/graph/nodes").length).toBeGreaterThan(0);
    expect(
      new Set(api.callsTo("GET", "/graph/nodes").map((c) => c.query.get("source_doc_id")))
    ).toEqual(new Set([DOC1]));
  });

  test("a failed simplification is shown inline and keeps the document", async ({
    page,
    api,
  }) => {
    api.on("GET", `/documents/${DOC1}/paraphrase`, RATE_LIMITED);
    await page.goto(readerUrl(DOC1));
    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();

    await page.getByRole("button", { name: "Apply Complexity" }).click();

    const inlineError = appAlerts(page).filter({ hasText: "Simplification failed" });
    await expect(inlineError).toContainText("Too many requests. Please try again later.");
    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();
    await expect(page.getByText("Document not found")).toHaveCount(0);

    // Also visible in the conceptual-only view
    await page.getByRole("button", { name: "Conceptual" }).click();
    await expect(inlineError).toContainText("Too many requests. Please try again later.");
  });

  test("a simplification still running when another document is opened does not block it", async ({
    page,
    api,
  }) => {
    const answer = deferred();
    api.on("GET", `/documents/${DOC1}/paraphrase`, {
      waitFor: answer.promise,
      body: { doc_id: DOC1, complexity: 50, content: "Simplified Curie text." },
    });
    await page.goto(readerUrl(DOC1));
    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();
    await page.getByRole("button", { name: "Apply Complexity" }).click();
    await expect(page.getByRole("button", { name: "Generating..." })).toBeDisabled();

    // Client-side navigation: the stores keep their state
    await page.getByRole("navigation").getByRole("link", { name: "Documents" }).click();
    await page.getByRole("link", { name: "Photosynthesis Basics" }).click();
    await expect(page.getByRole("heading", { name: "Photosynthesis Basics" })).toBeVisible();
    // Only now does the first document's simplification come back
    const paraphraseAnswered = page.waitForResponse((r) => r.url().includes("/paraphrase"));
    answer.release();
    await paraphraseAnswered;

    await expect(page.getByRole("button", { name: "Apply Complexity" })).toBeEnabled();
    await expect(page.getByText("Generating simplified version...")).toHaveCount(0);
    await expect(page.getByText("Simplified Curie text.")).toHaveCount(0);
  });

  test("a missing or foreign document shows one not-found state and nothing to act on", async ({
    page,
    api,
  }) => {
    await page.goto("/read/someone-elses-doc");

    const notFound = appAlerts(page);
    await expect(notFound.getByRole("heading", { name: "Document not found" })).toBeVisible();
    await expect(notFound).toContainText("It may have been deleted, or it belongs to another account.");
    // Said once, without the backend's long id-echoing message
    await expect(page.getByText("Document not found")).toHaveCount(1);
    await expect(page.getByText("someone-elses-doc")).toHaveCount(0);
    // No graph panel, generator or complexity controls
    await expect(page.getByRole("heading", { name: "Knowledge Graph" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Generate flashcards" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Apply Complexity" })).toHaveCount(0);
    expect(api.callsTo("GET", "/graph/nodes")).toHaveLength(0);

    await notFound.getByRole("link", { name: "Back to Documents" }).click();
    await expect(page).toHaveURL(/\/documents$/);
  });

  test("generates flashcards once, then says they exist and offers the review", async ({
    page,
    api,
  }) => {
    await page.goto(readerUrl(DOC1));
    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();

    await page.getByRole("button", { name: "Generate flashcards" }).click();

    // Two of the three generated questions were stored before
    await expect(page.getByText("1 new card created.")).toBeVisible();
    await expect(toasts(page).filter({ hasText: "Created 1 new flashcard" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Review flashcards" })).toHaveAttribute(
      "href",
      "/flashcards"
    );
    await expect(page.getByRole("button", { name: "Generate flashcards" })).toHaveCount(0);

    // Generating again stores nothing
    await page.reload();
    await page.getByRole("button", { name: "Generate flashcards" }).click();

    await expect(page.getByText("Flashcards for this document already exist.")).toBeVisible();
    await expect(page.getByRole("link", { name: "Review flashcards" })).toHaveAttribute(
      "href",
      "/flashcards"
    );
    expect(api.callsTo("POST", "/learning/flashcards/generate").map((c) => c.body)).toEqual([
      { document_id: DOC1 },
      { document_id: DOC1 },
    ]);
    expect(api.data.cards.filter((c) => c.document_id === DOC1)).toHaveLength(3);
  });
});

/**
 * Model the backend's graph build for DOC1: the document's status advances one
 * entry of `statuses` per GET /documents/{id}. While "processing", nodes are
 * being written one by one (only part of them exist) and relations come last;
 * "validated" (not built yet, or the build failed) has no graph at all.
 */
function modelBuild(api: MockApi, statuses: string[]) {
  const doc = api.data.documents.find((d) => d.doc_id === DOC1)!;
  const allNodes = api.data.nodes.filter((n) => n.source_doc_id === DOC1);
  const allRelations = api.data.relations[DOC1];
  let reads = 0;
  let countedAt = 0;
  const stage = () => statuses[Math.max(0, Math.min(reads, statuses.length) - 1)];
  api.on("GET", `/documents/${DOC1}`, () => {
    // Reads less than a second apart are one read: in development React runs
    // the page's mount effect (and so the first load) twice; polls are 3 s apart
    if (reads === 0 || Date.now() - countedAt >= 1000) {
      reads += 1;
      countedAt = Date.now();
    }
    return { body: { ...doc, status: stage() } };
  });
  const graphRequestsAt: string[] = [];
  api.on("GET", "/graph/nodes", (call) => {
    graphRequestsAt.push(stage());
    expect(call.query.get("source_doc_id")).toBe(DOC1);
    const nodes =
      stage() === "indexed" ? allNodes : stage() === "processing" ? allNodes.slice(0, 1) : [];
    return { body: { nodes, total: nodes.length } };
  });
  api.on("GET", `/graph/documents/${DOC1}/relations`, () => {
    const relations = stage() === "indexed" ? allRelations : [];
    return { body: { relations, total: relations.length } };
  });
  return { graphRequestsAt, reads: () => reads };
}

const building = (page: Page) => page.getByText("Building knowledge graph…");
const buildButton = (page: Page) => page.getByRole("button", { name: "Build knowledge graph", exact: true });

test.describe("Reader while the graph is being built", () => {
  test("opened during the build: waits for it, then shows the whole graph", async ({
    page,
    api,
  }) => {
    const build = modelBuild(api, ["processing", "processing", "indexed"]);
    await page.goto(readerUrl(DOC1));

    await expect(page.getByRole("heading", { name: CURIE })).toBeVisible();
    await expect(building(page)).toBeVisible();
    await expect(buildButton(page)).toHaveCount(0);
    await expect(graphCircles(page)).toHaveCount(0);

    // Nodes AND edges, loaded once the document is indexed
    await expect(graphCircles(page)).toHaveCount(3, { timeout: 20_000 });
    await expect(graphLines(page)).toHaveCount(2);
    await expect(building(page)).toHaveCount(0);
    await expect(page.getByText(/^indexed$/i)).toBeVisible();
    // The graph is only requested once the build is done
    expect(new Set(build.graphRequestsAt)).toEqual(new Set(["indexed"]));
  });

  test("just uploaded (validated): shows the build, not an empty graph to rebuild", async ({
    page,
    api,
  }) => {
    const build = modelBuild(api, ["validated", "processing", "indexed"]);
    await page.goto(readerUrl(DOC1));

    await expect(building(page)).toBeVisible();
    await expect(buildButton(page)).toHaveCount(0);
    await expect(page.getByText("No graph data available")).toHaveCount(0);

    await expect(graphCircles(page)).toHaveCount(3, { timeout: 20_000 });
    await expect(graphLines(page)).toHaveCount(2);
    // The graph is only requested once the build is done
    expect(new Set(build.graphRequestsAt)).toEqual(new Set(["indexed"]));
  });

  test("a build that ends without a graph offers the Build button", async ({ page, api }) => {
    modelBuild(api, ["processing", "validated"]);
    await page.goto(readerUrl(DOC1));
    await expect(building(page)).toBeVisible();

    await expect(buildButton(page)).toBeVisible({ timeout: 20_000 });
    await expect(building(page)).toHaveCount(0);
    await expect(page.getByText("No graph data available")).toBeVisible();
  });

  test("a validated document whose build never starts offers the Build button", async ({
    page,
    api,
  }) => {
    test.slow();
    const build = modelBuild(api, ["validated"]);
    await page.goto(readerUrl(DOC1));
    await expect(building(page)).toBeVisible();

    // After the reader stops waiting for the build to start (15 s)
    await expect(buildButton(page)).toBeVisible({ timeout: 30_000 });
    await expect(building(page)).toHaveCount(0);
    expect(build.reads()).toBeGreaterThan(1);
  });

  test("Build while another build runs (409 BUILD_IN_PROGRESS) waits for that build", async ({
    page,
    api,
  }) => {
    // Indexed, but its graph was deleted: the empty panel offers Build
    const build = modelBuild(api, ["indexed", "processing", "indexed"]);
    let nodesExist = false;
    api.on("GET", "/graph/nodes", () => {
      const nodes = nodesExist ? api.data.nodes.filter((n) => n.source_doc_id === DOC1) : [];
      return { body: { nodes, total: nodes.length } };
    });
    api.on("GET", `/graph/documents/${DOC1}/relations`, () => {
      const relations = nodesExist ? api.data.relations[DOC1] : [];
      return { body: { relations, total: relations.length } };
    });
    api.on("POST", `/graph/documents/${DOC1}/build`, () => {
      nodesExist = true; // the other build finishes the graph
      return {
        status: 409,
        body: {
          detail: {
            code: "BUILD_IN_PROGRESS",
            message: "The knowledge graph for this document is already being built",
          },
        },
      };
    });
    await page.goto(readerUrl(DOC1));
    await expect(page.getByText("No graph data available")).toBeVisible();

    await buildButton(page).click();

    await expect(building(page)).toBeVisible();
    await expect(appAlerts(page)).toHaveCount(0);
    await expect(graphCircles(page)).toHaveCount(3, { timeout: 20_000 });
    await expect(graphLines(page)).toHaveCount(2);
    await expect(page.getByText("already being built")).toHaveCount(0);
    expect(build.reads()).toBeGreaterThanOrEqual(3);
  });
});
