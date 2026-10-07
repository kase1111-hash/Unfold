import { test, expect, makeDocument, appAlerts, DOC1, RATE_LIMITED } from "./fixtures";

const hexId = (n: number) => `sha256:${n.toString(16).padStart(64, "0")}`;

test.describe("Documents list", () => {
  test("lists the user's documents with status and pagination", async ({ page, api }) => {
    api.data.documents.splice(
      0,
      api.data.documents.length,
      ...Array.from({ length: 25 }, (_, i) =>
        makeDocument({
          doc_id: hexId(i + 1),
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
    expect(api.callsTo("DELETE", `/documents/${DOC1}`)).toHaveLength(1);
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
  const otherPdf = { ...pdf, name: "darwin.pdf" };
  const NEW_DOC = hexId(0xc0ffee);

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
      doc_id: NEW_DOC,
      title: "Curie Notes",
      status: "validated",
      word_count: 41,
    });
    api.on("POST", "/documents/upload", () => {
      api.data.documents.push(uploaded);
      api.data.content[NEW_DOC] = "Notes about Marie Curie.";
      return {
        status: 201,
        body: { status: "success", message: "Document uploaded successfully", document: uploaded },
      };
    });
    await page.goto("/upload");

    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Upload Document" }).click();

    await expect(page.getByText("The knowledge graph is being built in the background.")).toBeVisible();
    await expect(page).toHaveURL(`/read/${encodeURIComponent(NEW_DOC)}`);
    await expect(page.getByRole("heading", { name: "Curie Notes" })).toBeVisible();
    expect(api.callsTo("POST", "/documents/upload")).toHaveLength(1);
    expect(String(api.callsTo("POST", "/documents/upload")[0].body)).toContain('name="file"');
  });

  test("after a rejected PDF the user can choose another file", async ({ page, api }) => {
    let uploads = 0;
    api.on("POST", "/documents/upload", () => {
      uploads += 1;
      if (uploads === 1) {
        return {
          status: 400,
          body: {
            detail: {
              code: "CORRUPT_PDF",
              message:
                "File appears to be corrupt or is not a valid PDF: Stream has ended unexpectedly",
            },
          },
        };
      }
      const doc = makeDocument({ doc_id: NEW_DOC, title: "Darwin Notes", status: "validated" });
      api.data.documents.push(doc);
      return { status: 201, body: { status: "success", message: "Document uploaded successfully", document: doc } };
    });
    await page.goto("/upload");
    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Upload Document" }).click();

    await expect(appAlerts(page).filter({ hasText: "File appears to be corrupt" })).toBeVisible();
    await expect(page).toHaveURL(/\/upload$/);
    // Sending the same rejected file again cannot help
    await expect(page.getByRole("button", { name: "Retry Upload" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Remove file" })).toBeVisible();

    await page.getByRole("button", { name: "Choose another file" }).click();
    await page.locator('input[type="file"]').setInputFiles(otherPdf);
    await expect(page.getByText("darwin.pdf")).toBeVisible();
    await page.getByRole("button", { name: "Upload Document" }).click();

    await expect(page).toHaveURL(`/read/${encodeURIComponent(NEW_DOC)}`);
    expect(api.callsTo("POST", "/documents/upload").map((c) => String(c.body).includes('filename="darwin.pdf"'))).toEqual([
      false,
      true,
    ]);
  });

  test("an upload that failed for a passing reason can be retried", async ({ page, api }) => {
    api.on("POST", "/documents/upload", RATE_LIMITED);
    await page.goto("/upload");
    await page.locator('input[type="file"]').setInputFiles(pdf);
    await page.getByRole("button", { name: "Upload Document" }).click();

    await expect(appAlerts(page).filter({ hasText: "Too many requests" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Retry Upload" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Choose another file" })).toBeVisible();

    await page.getByRole("button", { name: "Retry Upload" }).click();
    await expect.poll(() => api.callsTo("POST", "/documents/upload").length).toBe(2);
  });
});
