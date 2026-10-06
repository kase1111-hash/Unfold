import { test, expect, DOC1 } from "./fixtures";

test.describe("Public pages", () => {
  test.use({ authenticated: false });

  test("home page links to sign-in and registration only", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    await page.goto("/");

    await expect(page).toHaveTitle(/Unfold/);
    await expect(page.getByRole("heading", { level: 1, name: "Read Smarter, Not Harder" })).toBeVisible();
    const header = page.getByRole("banner");
    await expect(header.getByRole("link", { name: "Sign In" })).toHaveAttribute("href", "/login");
    await expect(header.getByRole("link", { name: "Get Started" })).toHaveAttribute("href", "/register");
    // No link to the missing /demo page
    await expect(page.locator('a[href="/demo"]')).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test("Get Started opens registration", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("banner").getByRole("link", { name: "Get Started" }).click();

    await expect(page).toHaveURL(/\/register$/);
    await expect(page.getByRole("heading", { name: "Create Your Account" })).toBeVisible();
  });

  test("pages declare an icon that exists (no /favicon.ico 404)", async ({ page, request }) => {
    await page.goto("/");

    const icon = page.locator('head link[rel="icon"]');
    await expect(icon).toHaveCount(1);
    const href = await icon.getAttribute("href");
    expect(href).toMatch(/^\/icon\.svg/);
    const response = await request.get(href!);
    expect(response.status()).toBe(200);
    expect(response.headers()["content-type"]).toContain("image/svg+xml");
  });

  test("unknown routes return 404", async ({ page }) => {
    const response = await page.goto("/nonexistent-page-12345");

    expect(response?.status()).toBe(404);
  });

  test("home page renders on a phone-sized viewport", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto("/");

    await expect(page.getByRole("heading", { level: 1, name: "Read Smarter, Not Harder" })).toBeVisible();
  });
});

test.describe("Dashboard", () => {
  test("shows real numbers from the API", async ({ page, api }) => {
    await page.goto("/dashboard");

    await expect(page.getByRole("heading", { name: "Welcome back, Alice Example!" })).toBeVisible();
    const stats = page.getByTestId("stat-card");
    await expect(stats).toHaveCount(3);
    // Documents: total from GET /documents/ (2); cards from /learning/flashcards/stats
    await expect(stats.filter({ hasText: "Documents" })).toContainText("2");
    await expect(stats.filter({ hasText: "Cards Due" })).toContainText("2");
    await expect(stats.filter({ hasText: "Flashcards" })).toContainText("5");
    await expect(page.getByText("Reading Streak")).toHaveCount(0);

    await expect(page.getByRole("link", { name: /Marie Curie and Radioactivity/ })).toHaveAttribute(
      "href",
      `/read/${DOC1}`
    );
    const listCall = api.callsTo("GET", "/documents/")[0];
    expect(listCall.query.get("page")).toBe("1");
    expect(listCall.query.get("page_size")).toBe("5");
  });

  test("sidebar links to every section and nothing else", async ({ page }) => {
    await page.goto("/dashboard");

    const nav = page.getByRole("navigation");
    await expect(nav.getByRole("link")).toHaveText([
      "Dashboard",
      "Documents",
      "Knowledge Graph",
      "Flashcards",
      "Upload",
    ]);
    await expect(page.locator('a[href="/settings"]')).toHaveCount(0);

    await nav.getByRole("link", { name: "Documents" }).click();
    await expect(page).toHaveURL(/\/documents$/);
    await expect(page.getByRole("heading", { level: 1, name: "Documents" })).toBeVisible();
  });
});
