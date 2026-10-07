import {
  test,
  expect,
  appAlerts,
  toasts,
  CARD1,
  CARD2,
  DOC1,
  DOC2,
  Q1,
  Q2,
  UNHANDLED_500,
} from "./fixtures";

test.describe("Flashcards", () => {
  test("lists due cards with their real questions", async ({ page }) => {
    await page.goto("/flashcards");

    await expect(page.getByRole("heading", { level: 1, name: "Flashcards" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Ready to Review?" })).toBeVisible();
    await expect(page.getByText("You have 2 cards ready for review.", { exact: false })).toBeVisible();
    await expect(page.getByText(Q1)).toBeVisible();
    await expect(page.getByText(Q2)).toBeVisible();
    await expect(page.getByText(/^Card /)).toHaveCount(0);
    // Sidebar stats come from the caller's cards
    await expect(page.getByText("Flashcard Progress")).toBeVisible();
    // The dead Export button is gone
    await expect(page.getByRole("button", { name: "Export" })).toHaveCount(0);
  });

  test("hides reading progress while no reading has been recorded", async ({ page }) => {
    // The real profile without engagement data: zeros and a default 50% score
    await page.goto("/flashcards");

    await expect(page.getByText("Flashcard Progress")).toBeVisible();
    await expect(page.getByText("Reading Progress")).toHaveCount(0);
    await expect(page.getByText("Comprehension Score")).toHaveCount(0);
    await expect(page.getByText("0h 0m")).toHaveCount(0);
  });

  test("shows reading progress once there is engagement data", async ({ page, api }) => {
    Object.assign(api.data.engagement, {
      total_reading_time_minutes: 90,
      documents_read: 2,
      avg_session_duration_minutes: 15,
      comprehension_score: 70,
    });
    await page.goto("/flashcards");

    await expect(page.getByText("Reading Progress")).toBeVisible();
    await expect(page.getByText("1h 30m")).toBeVisible();
    await expect(page.getByText("70%")).toBeVisible();
  });

  test("reviews every due card, saving each rating before completing", async ({ page, api }) => {
    // The last review is slow: completion must wait for it
    api.on("POST", "/learning/flashcards/review", (call) => {
      const { card_id, quality } = call.body as { card_id: string; quality: number };
      return {
        delayMs: card_id === CARD2 ? 800 : 0,
        body: {
          card_id,
          quality,
          next_review: "2026-10-08T12:00:00+00:00",
          interval_days: quality >= 4 ? 6 : 1,
          easiness_factor: 2.6,
          repetitions: 1,
          retention_rate: 100.0,
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
      { card_id: CARD1, quality: 4 },
      { card_id: CARD2, quality: 5 },
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
      { card_id: CARD1, quality: 3 },
      { card_id: CARD2, quality: 4 },
    ]);
  });

  test("Create Cards generates flashcards from a chosen document", async ({ page, api }) => {
    api.data.dueCards.splice(0);
    await page.goto("/flashcards");
    await expect(page.getByRole("heading", { name: "No Cards Due" })).toBeVisible();

    await page.getByRole("button", { name: "Create Cards" }).click();
    const dialog = page.getByRole("dialog", { name: "Generate flashcards from a document" });
    await dialog.getByLabel("Document").selectOption({ label: "Photosynthesis Basics" });
    await dialog.getByRole("button", { name: "Generate" }).click();

    await expect(page.getByText("What do plants turn light into?")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Ready to Review?" })).toBeVisible();
    await expect(toasts(page).filter({ hasText: "Created 2 new flashcards" })).toBeVisible();
    await expect(dialog).toHaveCount(0);
    expect(api.callsTo("POST", "/learning/flashcards/generate").map((c) => c.body)).toEqual([
      { document_id: DOC2 },
    ]);
  });

  test("Create Cards for a document whose cards exist says so", async ({ page, api }) => {
    await page.goto("/flashcards");

    for (let run = 0; run < 2; run++) {
      await page.getByRole("button", { name: "Create Cards" }).click();
      const dialog = page.getByRole("dialog");
      await dialog.getByLabel("Document").selectOption({ label: "Photosynthesis Basics" });
      await dialog.getByRole("button", { name: "Generate" }).click();
      await expect(dialog).toHaveCount(0);
    }

    await expect(
      toasts(page).filter({ hasText: "Flashcards for this document already exist" })
    ).toBeVisible();
    // The second run stored nothing: still two cards for the document, no duplicates
    expect(api.data.cards.filter((c) => c.document_id === DOC2)).toHaveLength(2);
    await expect(page.getByText("You have 4 cards ready for review.", { exact: false })).toBeVisible();
  });

  test("Create Cards explains why generation failed", async ({ page, api }) => {
    api.data.content[DOC1] = "";
    await page.goto("/flashcards");

    await page.getByRole("button", { name: "Create Cards" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Generate" }).click();

    await expect(dialog.getByRole("alert")).toHaveText(
      `Document ${DOC1} has no text to generate flashcards from`
    );
    expect(api.callsTo("POST", "/learning/flashcards/generate").map((c) => c.body)).toEqual([
      { document_id: DOC1 },
    ]);
  });

  test("explains a server error when cards cannot be loaded", async ({ page, api }) => {
    api.on("GET", "/learning/flashcards/due", UNHANDLED_500);
    await page.goto("/flashcards");

    await expect(appAlerts(page)).toContainText(
      "Failed to load flashcards: Could not reach the server, or it ran into an error. Please try again."
    );
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
  });

  test("explains a plain-text server error (same-origin deployment)", async ({ page, api }) => {
    // Behind a same-origin proxy the app can read the 500 itself
    api.on("GET", "/learning/flashcards/due", { status: 500, text: "Internal Server Error" });
    await page.goto("/flashcards");

    await expect(appAlerts(page)).toContainText(
      "Failed to load flashcards: The server ran into an error. Please try again later."
    );
  });
});
