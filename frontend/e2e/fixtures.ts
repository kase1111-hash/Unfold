import { test as base, expect, type Page, type Route } from "@playwright/test";

/**
 * Shared E2E fixtures. No backend runs during the E2E suite: every request to
 * <anything>/api/v1/* is answered by MockApi, whose default handlers model the
 * backend API contract with a small in-memory data set. Tests override single
 * endpoints with `api.on(...)` and inspect what the app sent with `api.calls`.
 * A request without a handler gets a 404 and fails the test at teardown.
 */

export interface ApiCall {
  method: string;
  /** Path below /api/v1, e.g. "/documents/" or "/graph/nodes" */
  path: string;
  query: URLSearchParams;
  body: unknown;
  headers: Record<string, string>;
}

export interface MockResponse {
  status?: number;
  body?: unknown;
  /** Answer only after this many milliseconds */
  delayMs?: number;
}

type Handler = (call: ApiCall, match: RegExpMatchArray) => MockResponse;

const now = "2026-10-01T12:00:00Z";

export const USER = {
  user_id: "user-1",
  email: "alice@example.com",
  username: "alice",
  full_name: "Alice Example",
  role: "user",
  is_active: true,
  is_verified: true,
  created_at: now,
  updated_at: now,
};

export function makeDocument(overrides: Record<string, unknown> & { doc_id: string }) {
  return {
    title: `Document ${overrides.doc_id}`,
    authors: [],
    source: "upload",
    status: "indexed",
    graph_nodes: [],
    word_count: 120,
    page_count: 2,
    created_at: now,
    updated_at: now,
    ...overrides,
  };
}

export function makeNode(overrides: Record<string, unknown> & { node_id: string; label: string }) {
  return {
    type: "Concept",
    source_doc_id: "doc-1",
    confidence: 0.9,
    external_links: {},
    ...overrides,
  };
}

export function makeCard(
  overrides: Record<string, unknown> & { card_id: string; question: string; answer: string }
) {
  return {
    document_id: "doc-1",
    hint: null,
    card_type: "definition",
    difficulty: "medium",
    interval_days: 0,
    repetitions: 0,
    easiness_factor: 2.5,
    next_review: now,
    days_overdue: 0,
    ...overrides,
  };
}

export const DOC_CONTENT =
  "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.\n\nShe discovered the elements polonium and radium.";

export function defaultData() {
  return {
    documents: [
      makeDocument({
        doc_id: "doc-1",
        title: "Marie Curie and Radioactivity",
        authors: ["Eve Curie"],
      }),
      makeDocument({ doc_id: "doc-2", title: "Photosynthesis Basics", status: "validated" }),
    ],
    content: { "doc-1": DOC_CONTENT, "doc-2": "Plants turn light into chemical energy." } as Record<
      string,
      string
    >,
    nodes: [
      makeNode({ node_id: "node_curie", label: "Marie Curie", type: "Author" }),
      makeNode({ node_id: "node_radio", label: "Radioactivity", type: "Concept" }),
      makeNode({ node_id: "node_polonium", label: "Polonium", type: "Term" }),
      makeNode({ node_id: "node_chloro", label: "Chlorophyll", source_doc_id: "doc-2" }),
    ],
    relations: {
      "doc-1": [
        {
          relation_id: "rel_1",
          source_node_id: "node_curie",
          target_node_id: "node_radio",
          type: "RELATED_TO",
          weight: 0.8,
        },
        {
          relation_id: "rel_2",
          source_node_id: "node_radio",
          target_node_id: "node_polonium",
          type: "PART_OF",
          weight: 0.6,
        },
      ],
    } as Record<string, unknown[]>,
    dueCards: [
      makeCard({
        card_id: "card-1",
        question: "Which element did Marie Curie name after Poland?",
        answer: "Polonium",
        hint: "Her home country",
      }),
      makeCard({
        card_id: "card-2",
        question: "In which two sciences did Marie Curie win Nobel Prizes?",
        answer: "Physics and chemistry",
        difficulty: "hard",
      }),
    ],
    stats: {
      total_cards: 5,
      due_now: 2,
      due_today: 3,
      average_ef: 2.5,
      average_retention: 80,
      mature_cards: 1,
      learning_cards: 4,
    },
  };
}

export type MockData = ReturnType<typeof defaultData>;

export class MockApi {
  readonly calls: ApiCall[] = [];
  readonly data: MockData = defaultData();
  private handlers: { method: string; pattern: RegExp; handler: Handler }[] = [];

  constructor() {
    this.installDefaults();
  }

  /**
   * Answer `method path`. A string path must match exactly; a RegExp is matched
   * against the path (without query). Later registrations take precedence.
   */
  on(method: string, path: string | RegExp, response: MockResponse | Handler) {
    const pattern =
      typeof path === "string"
        ? new RegExp(`^${path.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`)
        : path;
    const handler: Handler = typeof response === "function" ? response : () => response;
    this.handlers.unshift({ method, pattern, handler });
  }

  /** Calls the app made to `method path` (same matching rules as `on`) */
  callsTo(method: string, path: string | RegExp): ApiCall[] {
    return this.calls.filter(
      (c) =>
        c.method === method && (typeof path === "string" ? c.path === path : path.test(c.path))
    );
  }

  /** Requests that no handler answered (they got a 404) */
  readonly unmockedCalls: string[] = [];

  async install(page: Page) {
    await page.route((url) => url.pathname.startsWith("/api/v1/"), (route) => this.handle(route));
  }

  private findHandler(call: ApiCall) {
    for (const h of this.handlers) {
      if (h.method !== call.method) continue;
      const match = call.path.match(h.pattern);
      if (match) return { handler: h.handler, match };
    }
    return null;
  }

  private async handle(route: Route) {
    const request = route.request();
    const cors = {
      "access-control-allow-origin": request.headers()["origin"] || "*",
      "access-control-allow-credentials": "true",
      "access-control-allow-headers": "authorization, content-type",
      "access-control-allow-methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    };
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: cors });
      return;
    }

    const url = new URL(request.url());
    let body: unknown = request.postData();
    try {
      body = body ? JSON.parse(body as string) : null;
    } catch {
      // Multipart upload bodies stay as raw text
    }
    const call: ApiCall = {
      method: request.method(),
      path: url.pathname.slice("/api/v1".length),
      query: url.searchParams,
      body,
      headers: request.headers(),
    };
    this.calls.push(call);

    const found = this.findHandler(call);
    if (!found) this.unmockedCalls.push(`${call.method} ${call.path}`);
    const response: MockResponse = found
      ? found.handler(call, found.match)
      : {
          status: 404,
          body: { detail: { code: "NOT_MOCKED", message: `No mock for ${call.method} ${call.path}` } },
        };
    if (response.delayMs) {
      await new Promise((resolve) => setTimeout(resolve, response.delayMs));
    }
    const status = response.status ?? 200;
    await route
      .fulfill({
        status,
        headers: { ...cors, "content-type": "application/json" },
        body: status === 204 ? "" : JSON.stringify(response.body ?? {}),
      })
      // The page may have navigated away while a delayed response was pending
      .catch(() => undefined);
  }

  private installDefaults() {
    const d = this.data;
    const notFound = { status: 404, body: { detail: { code: "NOT_FOUND", message: "Document not found" } } };
    const findDoc = (id: string) => d.documents.find((doc) => doc.doc_id === id);

    // Auth: any bearer token is valid; no refresh cookie
    this.on("GET", "/auth/me", (call) =>
      call.headers["authorization"]?.startsWith("Bearer ")
        ? { body: USER }
        : { status: 401, body: { detail: { code: "NOT_AUTHENTICATED", message: "Not authenticated" } } }
    );
    this.on("POST", "/auth/refresh", {
      status: 401,
      body: { detail: { code: "MISSING_REFRESH_TOKEN", message: "Refresh token missing" } },
    });
    this.on("POST", "/auth/logout", { body: { message: "Logged out" } });

    // Documents
    this.on("GET", "/documents/", (call) => {
      const page = Number(call.query.get("page") || 1);
      const pageSize = Number(call.query.get("page_size") || 20);
      const start = (page - 1) * pageSize;
      return {
        body: {
          status: "success",
          data: d.documents.slice(start, start + pageSize),
          total: d.documents.length,
          page,
          page_size: pageSize,
        },
      };
    });
    this.on("GET", /^\/documents\/([^/]+)$/, (_call, m) => {
      const doc = findDoc(decodeURIComponent(m[1]));
      return doc ? { body: doc } : notFound;
    });
    this.on("GET", /^\/documents\/([^/]+)\/content$/, (_call, m) => {
      const id = decodeURIComponent(m[1]);
      return findDoc(id) ? { body: { doc_id: id, content: d.content[id] ?? "" } } : notFound;
    });
    this.on("DELETE", /^\/documents\/([^/]+)$/, (_call, m) => {
      const id = decodeURIComponent(m[1]);
      const index = d.documents.findIndex((doc) => doc.doc_id === id);
      if (index === -1) return notFound;
      d.documents.splice(index, 1);
      return { status: 204 };
    });

    // Graph
    this.on("GET", "/graph/nodes", (call) => {
      const docId = call.query.get("source_doc_id");
      const nodes = docId ? d.nodes.filter((n) => n.source_doc_id === docId) : d.nodes;
      return { body: { nodes, total: nodes.length } };
    });
    this.on("GET", /^\/graph\/documents\/([^/]+)\/relations$/, (_call, m) => {
      const relations = d.relations[decodeURIComponent(m[1])] ?? [];
      return { body: { relations, total: relations.length } };
    });
    this.on("GET", /^\/graph\/link\/wikipedia\/(.+)$/, (_call, m) => ({
      body: { entity: decodeURIComponent(m[1]), title: null, url: null, extract: null, found: false },
    }));

    // Learning
    this.on("GET", "/learning/flashcards/due", () => ({
      body: { due_cards: d.dueCards, total_due: d.dueCards.length },
    }));
    this.on("GET", "/learning/flashcards/stats", () => ({ body: d.stats }));
    this.on("GET", "/learning/engagement/profile", {
      body: {
        total_reading_time_minutes: 90,
        documents_read: 2,
        avg_session_duration_minutes: 15,
        avg_scroll_depth: 60,
        preferred_complexity: 50,
        total_highlights: 3,
        total_flashcards: 5,
        comprehension_score: 70,
      },
    });
    this.on("POST", "/learning/flashcards/review", (call) => {
      const { card_id, quality } = call.body as { card_id: string; quality: number };
      return {
        body: {
          card_id,
          quality,
          next_review: "2026-10-07T12:00:00Z",
          interval_days: 1,
          easiness_factor: 2.5,
          repetitions: 1,
        },
      };
    });
  }
}

interface Fixtures {
  /** Seed an access token so protected pages load as USER (default true) */
  authenticated: boolean;
  api: MockApi;
}

export const test = base.extend<Fixtures>({
  authenticated: [true, { option: true }],

  api: [
    async ({ page, authenticated }, use) => {
      const api = new MockApi();
      await api.install(page);
      if (authenticated) {
        // Seed once per tab, so a test can observe the app clearing the token
        await page.addInitScript(() => {
          if (!window.sessionStorage.getItem("e2e-seeded")) {
            window.sessionStorage.setItem("e2e-seeded", "1");
            window.localStorage.setItem("access_token", "test-token");
          }
        });
      }
      await use(api);
      expect(api.unmockedCalls, "API calls without a mock").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };

/** The app's role="alert" elements, without Next's (empty) route announcer */
export function appAlerts(page: Page) {
  return page.locator('[role="alert"]:not(#__next-route-announcer__)');
}

/** The D3 graph's node circles and edge lines (not the icons around it) */
export function graphCircles(page: Page) {
  return page.locator('[data-testid="knowledge-graph"] g.nodes circle');
}

export function graphLines(page: Page) {
  return page.locator('[data-testid="knowledge-graph"] g.links line');
}
