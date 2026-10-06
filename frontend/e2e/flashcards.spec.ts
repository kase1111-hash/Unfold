import { test, expect, makeCard, appAlerts } from "./fixtures";

const Q1 = "Which element did Marie Curie name after Poland?";
const Q2 = "In which two sciences did Marie Curie win Nobel Prizes?";

test.describe("Flashcards", () => {
  test("lists due cards with their real questions", async ({ page }) => {
    await page.goto("/flashcards");

    await expect(page.getByRole("heading", { level: 1, name: "Flashcards" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Ready to Review?" })).toBeVisible();
    await expect(page.getByText("You have 2 cards ready for review.", { exact: false })).toBeVisible();
    await expect(page.getByText(Q1)).toBeVisible();
    await expect(page.getByText(Q2)).toBeVisible();
    await expect(page.getByText(/^Card card-/)).toHaveCount(0);
    // Sidebar stats come from the caller's cards
    await expect(page.getByText("Flashcard Progress")).toBeVisible();
    await expect(page.getByText("1h 30m")).toBeVisible();
    // The dead Export button is gone
    await expect(page.getByRole("button", { name: "Export" })).toHaveCount(0);
  });

  test("reviews every due card, saving each rating before completing", async ({ page, api }) => {
    // The last review is slow: completion must wait for it
    api.on("POST", "/learning/flashcards/review", (call) => {
      const { card_id, quality } = call.body as { card_id: string; quality: number };
      return {
        delayMs: card_id === "card-2" ? 800 : 0,
        body: {
          card_id,
          quality,
          next_review: "2026-10-08T12:00:00Z",
          interval_days: quality >= 4 ? 6 : 1,
          easiness_factor: 2.6,
          repetitions: 1,
        },
      };
    });
    await page.goto("/flashcards");
    await page.getByRole("button", { name: "Start Review Session" }).click();

    await expect(page.getByText(Q1)).toBeVisible();
    await page.getByText("Click to reveal answer").click();
    await expect(page.getByText("Polonium", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Good" }).click();

    await expect(page.getByText(Q2)).toBeVisible();
    await page.getByText("Click to reveal answer").click();
    await expect(page.getByText("Physics and chemistry")).toBeVisible();
    const dueLoadsBefore = api.callsTo("GET", "/learning/flashcards/due").length;
    await page.getByRole("button", { name: "Easy" }).click();
    // While the last review is being saved the ratings cannot be clicked again
    await expect(page.getByRole("button", { name: "Easy" })).toBeDisabled();
    await expect(page.getByRole("heading", { name: "Review Complete!" })).toHaveCount(0);

    await expect(page.getByRole("heading", { name: "Review Complete!" })).toBeVisible();
    expect(api.callsTo("POST", "/learning/flashcards/review").map((c) => c.body)).toEqual([
      { card_id: "card-1", quality: 4 },
      { card_id: "card-2", quality: 5 },
    ]);
    // Due cards are reloaded only when the user leaves the summary
    expect(api.callsTo("GET", "/learning/flashcards/due")).toHaveLength(dueLoadsBefore);

    api.data.dueCards.splice(0);
    await page.getByRole("button", { name: "Done" }).click();

    await expect(page.getByRole("heading", { name: "No Cards Due" })).toBeVisible();
    expect(api.callsTo("GET", "/learning/flashcards/due").length).toBeGreaterThan(dueLoadsBefore);
  });

  test("going back to a rated card does not submit it twice", async ({ page, api }) => {
    await page.goto("/flashcards");
    await page.getByRole("button", { name: "Start Review Session" }).click();

    await page.getByText("Click to reveal answer").click();
    await page.getByRole("button", { name: "Hard" }).click();
    await expect(page.getByText(Q2)).toBeVisible();

    await page.getByRole("button", { name: "Previous" }).click();
    await expect(page.getByText(Q1)).toBeVisible();
    await page.getByText("Click to reveal answer").click();
    await page.getByRole("button", { name: "Good" }).click();
    await page.getByText("Click to reveal answer").click();
    await page.getByRole("button", { name: "Good" }).click();

    await expect(page.getByRole("heading", { name: "Review Complete!" })).toBeVisible();
    expect(api.callsTo("POST", "/learning/flashcards/review").map((c) => c.body)).toEqual([
      { card_id: "card-1", quality: 3 },
      { card_id: "card-2", quality: 4 },
    ]);
  });

  test("Create Cards generates flashcards from a chosen document", async ({ page, api }) => {
    api.data.dueCards.splice(0);
    api.on("POST", "/learning/flashcards/generate", (call) => {
      const { document_id } = call.body as { document_id: string };
      const cards = [
        makeCard({
          card_id: "gen-1",
          document_id,
          question: "What do plants convert light into?",
          answer: "Chemical energy",
        }),
        makeCard({
          card_id: "gen-2",
          document_id,
          question: "Which pigment absorbs light in plants?",
          answer: "Chlorophyll",
        }),
      ];
      api.data.dueCards.push(...cards);
      return { body: { document_id, flashcards: cards, count: cards.length } };
    });
    await page.goto("/flashcards");
    await expect(page.getByRole("heading", { name: "No Cards Due" })).toBeVisible();

    await page.getByRole("button", { name: "Create Cards" }).click();
    const dialog = page.getByRole("dialog", { name: "Generate flashcards from a document" });
    await dialog.getByLabel("Document").selectOption({ label: "Photosynthesis Basics" });
    await dialog.getByRole("button", { name: "Generate" }).click();

    await expect(page.getByText("What do plants convert light into?")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Ready to Review?" })).toBeVisible();
    await expect(dialog).toHaveCount(0);
    expect(api.callsTo("POST", "/learning/flashcards/generate").map((c) => c.body)).toEqual([
      { document_id: "doc-2" },
    ]);
  });

  test("Create Cards explains why generation failed", async ({ page, api }) => {
    api.on("POST", "/learning/flashcards/generate", {
      status: 422,
      body: { detail: { code: "NO_CONTENT", message: "This document has no text to generate flashcards from." } },
    });
    await page.goto("/flashcards");

    await page.getByRole("button", { name: "Create Cards" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Generate" }).click();

    await expect(dialog.getByRole("alert")).toHaveText(
      "This document has no text to generate flashcards from."
    );
    expect(api.callsTo("POST", "/learning/flashcards/generate").map((c) => c.body)).toEqual([
      { document_id: "doc-1" },
    ]);
  });

  test("shows the backend error when cards cannot be loaded", async ({ page, api }) => {
    api.on("GET", "/learning/flashcards/due", {
      status: 500,
      body: { detail: { code: "INTERNAL_ERROR", message: "Database unavailable" } },
    });
    await page.goto("/flashcards");

    await expect(appAlerts(page)).toContainText("Failed to load flashcards: Database unavailable");
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
  });
});
