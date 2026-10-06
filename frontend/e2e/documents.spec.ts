import { test, expect, makeDocument, appAlerts } from "./fixtures";

test.describe("Documents list", () => {
  test("lists the user's documents with status and pagination", async ({ page, api }) => {
    api.data.documents.splice(
      0,
      api.data.documents.length,
      ...Array.from({ length: 25 }, (_, i) =>
        makeDocument({
          doc_id: `doc-${i + 1}`,
          title: `Paper ${String(i + 1).padStart(2, "0")}`,
          status: i === 0 ? "indexed" : "validated",
        })
      )
    );
    await page.goto("/documents");

    const rows = page.locator("tbody tr");
    await expect(rows).toHaveCount(10);
    await expect(rows.first()).toContainText("Paper 01");
    await expect(rows.first()).toContainText("indexed");
    await expect(page.getByText("Page 1 of 3")).toBeVisible();
    // Trailing-slash path (no redirect) with snake_case paging params
    const first = api.callsTo("GET", "/documents/")[0];
    expect(first.query.get("page")).toBe("1");
    expect(first.query.get("page_size")).toBe("10");

    await page.getByRole("button", { name: "Next" }).click();

    await expect(page.getByText("Page 2 of 3")).toBeVisible();
    await expect(rows.first()).toContainText("Paper 11");
    expect(api.callsTo("GET", "/documents/").at(-1)?.query.get("page")).toBe("2");
  });

  test("filters rows by the search box", async ({ page }) => {
    await page.goto("/documents");
    await expect(page.locator("tbody tr")).toHaveCount(2);

    await page.getByPlaceholder("Search documents...").fill("photosynthesis");

    await expect(page.locator("tbody tr")).toHaveCount(1);
    await expect(page.locator("tbody tr")).toContainText("Photosynthesis Basics");
  });

  test("deletes a document after confirmation", async ({ page, api }) => {
    await page.goto("/documents");
    await expect(page.locator("tbody tr")).toHaveCount(2);

    page.once("dialog", (dialog) => dialog.accept());
    await page
      .locator("tbody tr", { hasText: "Marie Curie and Radioactivity" })
      .getByTitle("Delete")
      .click();

    await expect(page.locator("tbody tr")).toHaveCount(1);
    await expect(page.locator("tbody tr")).toContainText("Photosynthesis Basics");
    expect(api.callsTo("DELETE", "/documents/doc-1")).toHaveLength(1);
  });

  test("shows an empty state when there are no documents", async ({ page, api }) => {
    api.data.documents.splice(0);
    await page.goto("/documents");

    await expect(page.getByText("No documents uploaded yet")).toBeVisible();
  });
});

test.describe("Upload", () => {
  const pdf = {
    name: "curie.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4\n% test file\n"),
  };

  test("accepts PDFs only", async ({ page }) => {
    await page.goto("/upload");

    await expect(page.getByRole("heading", { name: "Upload Document" })).toBeVisible();
    await expect(page.locator('input[type="file"]')).toHaveAttribute("accept", ".pdf,application/pdf");
    await expect(page.getByText("Supported format: PDF (max 50MB)")).toBeVisible();
  });

  test("uploads, explains the background graph build and opens the reader", async ({
    page,
    api,
  }) => {
    const uploaded = makeDocument({
      doc_id: "doc-new",
      title: "Curie Notes",
      status: "validated",
      word_count: 41,
    });
    api.on("POST", "/documents/upload", () => {
      api.data.documents.push(uploaded);
      api.data.content["doc-new"] = "Notes about Marie Curie.";
      return {
        status: 201,
        body: { status: "success", message: "Document uploaded", document: uploaded },
      };
    });
    await page.goto("/upload");

    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Upload Document" }).click();

    await expect(page.getByText("The knowledge graph is being built in the background.")).toBeVisible();
    await expect(page).toHaveURL(/\/read\/doc-new$/);
    await expect(page.getByRole("heading", { name: "Curie Notes" })).toBeVisible();
    expect(api.callsTo("POST", "/documents/upload")).toHaveLength(1);
    expect(String(api.callsTo("POST", "/documents/upload")[0].body)).toContain('name="file"');
  });

  test("shows the backend's reason when a PDF is rejected", async ({ page, api }) => {
    api.on("POST", "/documents/upload", {
      status: 400,
      body: {
        detail: {
          code: "CORRUPT_PDF",
          message: "File appears to be corrupt or is not a valid PDF.",
        },
      },
    });
    await page.goto("/upload");

    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Upload Document" }).click();

    await expect(appAlerts(page).filter({ hasText: "File appears to be corrupt" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Retry Upload" })).toBeVisible();
    await expect(page).toHaveURL(/\/upload$/);
  });
});
