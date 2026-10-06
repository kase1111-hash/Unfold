import { test, expect, makeDocument, makeCard, appAlerts } from "./fixtures";

test.describe("Reader", () => {
  test("shows the document text with its line breaks", async ({ page, api }) => {
    await page.goto("/read/doc-1");

    await expect(page.getByRole("heading", { name: "Marie Curie and Radioactivity" })).toBeVisible();
    await expect(page.getByText("Eve Curie")).toBeVisible();
    const original = page.getByText(/She discovered the elements polonium and radium\./).first();
    await expect(original).toBeVisible();
    await expect(original).toHaveCSS("white-space", "pre-wrap");
    await expect(page.getByText("120 words")).toBeVisible();
    expect(api.callsTo("GET", "/documents/doc-1/content")).not.toHaveLength(0);
  });

  test("decodes the document id from the URL before calling the API", async ({ page, api }) => {
    api.data.documents.push(makeDocument({ doc_id: "sha256:abc123", title: "Hashed Doc" }));
    api.data.content["sha256:abc123"] = "Content of the hashed document.";
    await page.goto("/read/sha256%3Aabc123");

    await expect(page.getByRole("heading", { name: "Hashed Doc" })).toBeVisible();
    await expect(page.getByText("Content of the hashed document.").first()).toBeVisible();
    // The graph filter must carry the real id, not the still-encoded "sha256%3Aabc123"
    await expect
      .poll(() => api.callsTo("GET", "/graph/nodes").map((c) => c.query.get("source_doc_id")))
      .toContain("sha256:abc123");
    expect(
      api.callsTo("GET", "/graph/nodes").map((c) => c.query.get("source_doc_id"))
    ).not.toContain("sha256%3Aabc123");
  });

  test("a failed simplification is shown inline and keeps the document", async ({
    page,
    api,
  }) => {
    api.on("GET", "/documents/doc-1/paraphrase", {
      status: 503,
      body: { detail: { code: "LLM_UNAVAILABLE", message: "Paraphrasing service unavailable" } },
    });
    await page.goto("/read/doc-1");
    await expect(page.getByRole("heading", { name: "Marie Curie and Radioactivity" })).toBeVisible();

    await page.getByRole("button", { name: "Apply Complexity" }).click();

    const inlineError = appAlerts(page).filter({ hasText: "Simplification failed" });
    await expect(inlineError).toContainText("Paraphrasing service unavailable");
    await expect(page.getByRole("heading", { name: "Marie Curie and Radioactivity" })).toBeVisible();
    await expect(page.getByText("Error loading document")).toHaveCount(0);

    // Also visible in the conceptual-only view
    await page.getByRole("button", { name: "Conceptual" }).click();
    await expect(inlineError).toContainText("Paraphrasing service unavailable");
  });

  test("shows the backend message for a document the user cannot open", async ({ page }) => {
    await page.goto("/read/someone-elses-doc");

    await expect(page.getByText("Error loading document")).toBeVisible();
    await expect(page.getByText("Document not found")).toBeVisible();
  });

  test("generates flashcards for the open document", async ({ page, api }) => {
    api.on("POST", "/learning/flashcards/generate", (call) => ({
      body: {
        document_id: (call.body as { document_id: string }).document_id,
        flashcards: [
          makeCard({ card_id: "c-a", question: "Q1?", answer: "A1" }),
          makeCard({ card_id: "c-b", question: "Q2?", answer: "A2" }),
          makeCard({ card_id: "c-c", question: "Q3?", answer: "A3" }),
        ],
        count: 3,
      },
    }));
    await page.goto("/read/doc-1");
    await expect(page.getByRole("heading", { name: "Marie Curie and Radioactivity" })).toBeVisible();

    await page.getByRole("button", { name: "Generate flashcards" }).click();

    await expect(page.getByRole("link", { name: "3 flashcards created. Review now" })).toHaveAttribute(
      "href",
      "/flashcards"
    );
    expect(api.callsTo("POST", "/learning/flashcards/generate").map((c) => c.body)).toEqual([
      { document_id: "doc-1" },
    ]);
  });

  test("loads the graph once the background build has indexed the document", async ({
    page,
    api,
  }) => {
    // Just uploaded: validated, no graph yet; the background task finishes later
    const doc = api.data.documents[1]; // doc-2, status "validated"
    const savedNodes = api.data.nodes.splice(0);
    let documentReads = 0;
    api.on("GET", "/documents/doc-2", () => {
      documentReads += 1;
      if (documentReads === 3) {
        doc.status = "indexed";
        api.data.nodes.push(...savedNodes);
      }
      return { body: doc };
    });
    await page.goto("/read/doc-2");

    await expect(page.getByRole("button", { name: "Build knowledge graph" })).toBeVisible();
    await expect(page.locator('[data-testid="knowledge-graph"] g.nodes circle')).toHaveCount(1, {
      timeout: 20_000,
    });
    await expect(page.getByText(/^indexed$/i)).toBeVisible();
  });
});
