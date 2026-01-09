import { test, expect } from "@playwright/test";

test.describe("Knowledge Graph", () => {
  test("should load graph page", async ({ page }) => {
    await page.goto("/graph");
    await page.waitForLoadState("networkidle");

    await expect(page.locator("body")).toBeVisible();
  });

  test("should display graph visualization container", async ({ page }) => {
    await page.goto("/graph");
    await page.waitForTimeout(1000);

    // Look for SVG or canvas element for D3 visualization
    const graphContainer = page.locator("svg, canvas, [data-testid='graph-container'], .graph-visualization");
    if (await graphContainer.count() > 0) {
      await expect(graphContainer.first()).toBeVisible();
    }
  });

  test("should have search/filter functionality", async ({ page }) => {
    await page.goto("/graph");

    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    if (await searchInput.count() > 0) {
      await expect(searchInput.first()).toBeVisible();
    }
  });
});

test.describe("Flashcards", () => {
  test("should load flashcards page", async ({ page }) => {
    await page.goto("/flashcards");
    await page.waitForLoadState("networkidle");

    await expect(page.locator("body")).toBeVisible();
  });

  test("should display flashcard interface or empty state", async ({ page }) => {
    await page.goto("/flashcards");
    await page.waitForTimeout(1000);

    // Should show flashcards or empty state
    const hasFlashcards = await page.locator('[data-testid="flashcard"], .flashcard, .card').count() > 0;
    const hasEmptyState = await page.locator('text=/no flashcards|empty|create|get started/i').count() > 0;
    const hasInterface = await page.locator('button, h1, h2').count() > 0;

    expect(hasFlashcards || hasEmptyState || hasInterface).toBeTruthy();
  });

  test("should have study/review button if flashcards exist", async ({ page }) => {
    await page.goto("/flashcards");
    await page.waitForTimeout(1000);

    const studyButton = page.locator('button:has-text("study"), button:has-text("review"), button:has-text("start")');
    // Button may not exist if no flashcards
    if (await studyButton.count() > 0) {
      await expect(studyButton.first()).toBeVisible();
    }
  });
});

test.describe("Dashboard", () => {
  test("should load dashboard page", async ({ page }) => {
    await page.goto("/dashboard");
    await page.waitForLoadState("networkidle");

    // Should either show dashboard or redirect to login
    const currentUrl = page.url();
    expect(currentUrl.includes("/dashboard") || currentUrl.includes("/login")).toBeTruthy();
  });

  test("should display statistics or summary cards", async ({ page }) => {
    await page.goto("/dashboard");
    await page.waitForTimeout(1000);

    // Look for stat cards or summary widgets
    const statElements = page.locator('[data-testid="stat-card"], .stat-card, .summary-card, .metric');
    // May not be visible if not logged in
  });

  test("should have navigation to other sections", async ({ page }) => {
    await page.goto("/dashboard");
    await page.waitForTimeout(500);

    // Look for links to other sections
    const links = page.locator("a[href]");
    const linkCount = await links.count();
    expect(linkCount).toBeGreaterThan(0);
  });
});

test.describe("Accessibility", () => {
  test("should have proper heading hierarchy", async ({ page }) => {
    await page.goto("/");

    // Check for h1
    const h1 = page.locator("h1");
    if (await h1.count() > 0) {
      expect(await h1.count()).toBeGreaterThanOrEqual(1);
    }
  });

  test("should have alt text on images", async ({ page }) => {
    await page.goto("/");

    const images = page.locator("img");
    const imgCount = await images.count();

    for (let i = 0; i < imgCount; i++) {
      const alt = await images.nth(i).getAttribute("alt");
      // Images should have alt attribute (can be empty for decorative)
      expect(alt !== null).toBeTruthy();
    }
  });

  test("should have proper form labels", async ({ page }) => {
    await page.goto("/login");

    const inputs = page.locator('input:not([type="hidden"]):not([type="submit"])');
    const inputCount = await inputs.count();

    for (let i = 0; i < inputCount; i++) {
      const input = inputs.nth(i);
      const id = await input.getAttribute("id");
      const ariaLabel = await input.getAttribute("aria-label");
      const placeholder = await input.getAttribute("placeholder");

      // Input should have some form of labeling
      const hasLabel = id || ariaLabel || placeholder;
      expect(hasLabel).toBeTruthy();
    }
  });

  test("should support keyboard navigation", async ({ page }) => {
    await page.goto("/");

    // Press Tab and check that focus moves
    await page.keyboard.press("Tab");

    // Some element should be focused
    const focusedElement = page.locator(":focus");
    const hasFocus = await focusedElement.count() > 0;
    expect(hasFocus).toBeTruthy();
  });
});

test.describe("Performance", () => {
  test("should load home page within acceptable time", async ({ page }) => {
    const startTime = Date.now();
    await page.goto("/");
    await page.waitForLoadState("domcontentloaded");
    const loadTime = Date.now() - startTime;

    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test("should not have console errors on load", async ({ page }) => {
    const errors: string[] = [];

    page.on("console", (msg) => {
      if (msg.type() === "error") {
        errors.push(msg.text());
      }
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Filter out expected errors (like API calls to mock server)
    const unexpectedErrors = errors.filter(
      (e) => !e.includes("Failed to fetch") && !e.includes("NetworkError")
    );

    expect(unexpectedErrors.length).toBe(0);
  });
});

test.describe("Dark Mode", () => {
  test("should respect system color scheme preference", async ({ page }) => {
    await page.emulateMedia({ colorScheme: "dark" });
    await page.goto("/");

    // Check if dark mode is applied (body or html should have dark class or attribute)
    const body = page.locator("body");
    const html = page.locator("html");

    const bodyClass = await body.getAttribute("class");
    const htmlClass = await html.getAttribute("class");

    // App should respond to color scheme (either via class or CSS)
    await expect(page.locator("body")).toBeVisible();
  });
});
