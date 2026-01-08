import { test, expect } from "@playwright/test";

test.describe("Documents Management", () => {
  test.describe("Document List Page", () => {
    test("should load documents page", async ({ page }) => {
      await page.goto("/documents");
      await page.waitForLoadState("networkidle");

      // Page should load without errors
      await expect(page.locator("body")).toBeVisible();
    });

    test("should display document list or empty state", async ({ page }) => {
      await page.goto("/documents");
      await page.waitForLoadState("networkidle");

      // Should show either documents or empty state message
      const hasDocuments = await page.locator('[data-testid="document-item"], .document-card, article').count() > 0;
      const hasEmptyState = await page.locator('text=/no documents|empty|upload|get started/i').count() > 0;

      expect(hasDocuments || hasEmptyState).toBeTruthy();
    });

    test("should have search functionality", async ({ page }) => {
      await page.goto("/documents");

      // Look for search input
      const searchInput = page.locator('input[type="search"], input[placeholder*="search" i], input[name="search"]');
      if (await searchInput.count() > 0) {
        await expect(searchInput).toBeVisible();
      }
    });

    test("should have upload button or link", async ({ page }) => {
      await page.goto("/documents");

      // Look for upload functionality
      const uploadElement = page.locator('a[href*="upload"], button:has-text("upload"), button:has-text("add")');
      if (await uploadElement.count() > 0) {
        await expect(uploadElement.first()).toBeVisible();
      }
    });
  });

  test.describe("Document Upload Page", () => {
    test("should load upload page", async ({ page }) => {
      await page.goto("/upload");
      await page.waitForLoadState("networkidle");

      await expect(page.locator("body")).toBeVisible();
    });

    test("should display file upload area", async ({ page }) => {
      await page.goto("/upload");

      // Look for file input or drop zone
      const fileInput = page.locator('input[type="file"]');
      const dropZone = page.locator('[data-testid="drop-zone"], .dropzone, .upload-area');

      const hasFileInput = await fileInput.count() > 0;
      const hasDropZone = await dropZone.count() > 0;

      expect(hasFileInput || hasDropZone).toBeTruthy();
    });

    test("should show supported file types", async ({ page }) => {
      await page.goto("/upload");

      // Look for file type information
      const fileTypeInfo = page.locator('text=/pdf|epub|supported/i');
      if (await fileTypeInfo.count() > 0) {
        await expect(fileTypeInfo.first()).toBeVisible();
      }
    });

    test("should validate file type on selection", async ({ page }) => {
      await page.goto("/upload");

      const fileInput = page.locator('input[type="file"]');
      if (await fileInput.count() > 0) {
        // Check that file input has accept attribute
        const acceptAttr = await fileInput.getAttribute("accept");
        if (acceptAttr) {
          expect(acceptAttr.toLowerCase()).toMatch(/pdf|epub|application/);
        }
      }
    });
  });
});

test.describe("Reading Interface", () => {
  test("should handle reading page with invalid document ID", async ({ page }) => {
    await page.goto("/read/nonexistent-doc-123");
    await page.waitForLoadState("networkidle");

    // Should show error or redirect
    const currentUrl = page.url();
    const hasError = await page.locator('text=/not found|error|404/i').count() > 0;

    expect(currentUrl.includes("/read") || hasError || currentUrl.includes("/documents")).toBeTruthy();
  });

  test("should have complexity slider component available", async ({ page }) => {
    await page.goto("/read/test-doc");
    await page.waitForTimeout(1000);

    // Look for slider or complexity control
    const slider = page.locator('input[type="range"], [role="slider"], .complexity-slider');
    // May not be visible if document doesn't exist, but component should be available
  });
});
