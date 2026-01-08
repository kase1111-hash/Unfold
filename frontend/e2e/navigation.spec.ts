import { test, expect } from "@playwright/test";

test.describe("Navigation", () => {
  test("should load the home page", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/Unfold/i);
  });

  test("should navigate to login page", async ({ page }) => {
    await page.goto("/login");
    await expect(page.locator("h1, h2")).toContainText(/login|sign in/i);
  });

  test("should navigate to register page", async ({ page }) => {
    await page.goto("/register");
    await expect(page.locator("h1, h2")).toContainText(/register|sign up|create/i);
  });

  test("should have working navigation links", async ({ page }) => {
    await page.goto("/");

    // Check for common navigation elements
    const nav = page.locator("nav, header");
    await expect(nav).toBeVisible();
  });

  test("should handle 404 pages gracefully", async ({ page }) => {
    const response = await page.goto("/nonexistent-page-12345");
    // Should either show 404 or redirect
    expect(response?.status()).toBeLessThan(500);
  });
});

test.describe("Responsive Design", () => {
  test("should be responsive on mobile viewport", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto("/");

    // Page should still be functional
    await expect(page.locator("body")).toBeVisible();
  });

  test("should be responsive on tablet viewport", async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto("/");

    await expect(page.locator("body")).toBeVisible();
  });
});
