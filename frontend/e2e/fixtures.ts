import { test as base, expect, type Page, type Route } from "@playwright/test";

/**
 * Shared E2E fixtures. No backend runs during the E2E suite: every request to
 * <anything>/api/v1/* is answered by MockApi, whose default handlers model the
 * backend API contract with a small in-memory data set. Tests override single
 * endpoints with `api.on(...)` and inspect what the app sent with `api.calls`.
 * A request without a handler gets a 404 and fails the test at teardown.
 *
 * Bodies, error codes and messages follow responses captured from the real
 * backend: ids are "sha256:<hex>" documents and UUID cards, app errors are
 * {detail: {code, message}}, and an unhandled exception is a plain-text 500
 * without CORS headers (UNHANDLED_500).
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
  /** JSON body */
  body?: unknown;
  /** A non-JSON body, sent as text/plain */
  text?: string;
  /**
   * false: no CORS headers. A cross-origin app (the dev and CI setup) then
   * cannot read the response and gets a network error instead.
   */
  cors?: boolean;
  /** Answer only after this many milliseconds */
  delayMs?: number;
  /** Answer only once this promise settles (the test decides when) */
  waitFor?: Promise<unknown>;
}

/** A promise the test resolves itself, e.g. to hold a response back */
export function deferred() {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => (release = resolve));
  return { promise, release };
}

type Handler = (call: ApiCall, match: RegExpMatchArray) => MockResponse;

/**
 * What the backend sends for an unhandled exception: Starlette's
 * ServerErrorMiddleware answers outside the CORS middleware, so the response is
 * plain text with no CORS headers, and the cross-origin app sees a network error.
 */
export const UNHANDLED_500: MockResponse = {
  status: 500,
  text: "Internal Server Error",
  cors: false,
};

/** The rate limiter's answer (sent with CORS headers) */
export const RATE_LIMITED: MockResponse = {
  status: 429,
  body: {
    detail: { code: "RATE_LIMIT_EXCEEDED", message: "Too many requests. Please try again later." },
  },
};

/** An expired access token */
export const TOKEN_EXPIRED: MockResponse = {
  status: 401,
  body: { detail: { code: "INVALID_TOKEN", message: "Token has expired" } },
};

/** POST /auth/refresh without a (valid) refresh cookie */
export const NO_REFRESH_TOKEN: MockResponse = {
  status: 401,
  body: { detail: { code: "MISSING_REFRESH_TOKEN", message: "Refresh token required" } },
};

const now = "2026-10-01T12:00:00Z";

export const DOC1 = "sha256:b2adb213f822641771acb898c7bfa79bdc23abee6d77c38a72d9e1a87ffcca69";
export const DOC2 = "sha256:97f64d5e88390efc86be5942bfd74fda383450279864731e09d80328b25d4672";
export const CARD1 = "b2daa461-9829-44de-8652-b94e7dccb4ca";
export const CARD2 = "641e82bd-148d-4015-a73f-83dbb5932d78";

export const USER = {
  user_id: "04440243-f32c-415d-b217-cdcc5d4815a5",
  email: "alice@example.com",
  username: "alice",
  full_name: "Alice Example",
  orcid_id: null,
  role: "user",
  is_active: true,
  is_verified: false,
  last_login: null,
  created_at: now,
  updated_at: now,
};

export function makeDocument(overrides: Record<string, unknown> & { doc_id: string }) {
  return {
    title: `Document ${overrides.doc_id}`,
    authors: [] as string[],
    doi: null,
    abstract: null,
    license: null,
    source: "upload",
    status: "indexed",
    vector_id: null,
    graph_nodes: [] as string[],
    file_path: `uploads/documents/${overrides.doc_id.replace(":", "_")}.pdf`,
    file_size_bytes: 2965,
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
    description: null,
    metadata: null,
    embedding: null,
    source_doc_id: DOC1,
    confidence: 0.85,
    external_links: {},
    created_at: now,
    updated_at: now,
    ...overrides,
  };
}

/** A card as POST /learning/flashcards/generate returns it (no days_overdue) */
export function makeCard(
  overrides: Record<string, unknown> & { card_id: string; question: string; answer: string }
) {
  return {
    document_id: DOC1,
    hint: null,
    card_type: "recall",
    difficulty: "medium",
    interval_days: 0,
    repetitions: 0,
    easiness_factor: 2.5,
    next_review: "2026-10-01T12:00:00+00:00",
    ...overrides,
  };
}

/** A card as GET /learning/flashcards/due returns it */
export function makeDueCard(
  overrides: Record<string, unknown> & { card_id: string; question: string; answer: string }
) {
  return { ...makeCard(overrides), days_overdue: 0, ...overrides };
}

export function relation(id: string, source: string, target: string, type = "RELATED_TO") {
  return { relation_id: id, source_node_id: source, target_node_id: target, type, weight: 0.5 };
}

export const DOC_CONTENT =
  "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.\n\nShe discovered the elements polonium and radium.";

export const Q1 = "Which element did Marie Curie name after Poland?";
export const Q2 = "In which two sciences did Marie Curie win Nobel Prizes?";

export function defaultData() {
  const dueCards = [
    makeDueCard({
      card_id: CARD1,
      question: Q1,
      answer: "Polonium",
      hint: "Her home country",
    }),
    makeDueCard({
      card_id: CARD2,
      question: Q2,
      answer: "Physics and chemistry",
      difficulty: "hard",
    }),
  ];
  return {
    documents: [
      makeDocument({
        doc_id: DOC1,
        title: "Marie Curie and Radioactivity",
        authors: ["Eve Curie"],
      }),
      makeDocument({ doc_id: DOC2, title: "Photosynthesis Basics" }),
    ],
    content: {
      [DOC1]: DOC_CONTENT,
      [DOC2]: "Plants turn light into chemical energy.",
    } as Record<string, string>,
    nodes: [
      makeNode({ node_id: "node_a1cedb2f2704", label: "Marie Curie", type: "Author" }),
      makeNode({ node_id: "node_b27a2a0f4c54", label: "Radioactivity", type: "Concept" }),
      makeNode({ node_id: "node_c06fe41637f2", label: "Polonium", type: "Term" }),
      makeNode({ node_id: "node_d4e5f6a7b8c9", label: "Chlorophyll", source_doc_id: DOC2 }),
    ],
    relations: {
      [DOC1]: [
        relation("rel_bce224bb2c44", "node_a1cedb2f2704", "node_b27a2a0f4c54"),
        relation("rel_a7abb2e78cc0", "node_b27a2a0f4c54", "node_c06fe41637f2", "PART_OF"),
      ],
    } as Record<string, ReturnType<typeof relation>[]>,
    /** Every stored card of the user (generation skips questions found here) */
    cards: dueCards.map((card) => ({ ...card })),
    dueCards,
    /** What the (deterministic, rule-based) generator extracts per document */
    generated: {
      [DOC1]: [
        { question: Q1, answer: "Polonium" },
        { question: Q2, answer: "Physics and chemistry" },
        { question: "_____ discovered the elements polonium and radium", answer: "Marie Curie" },
      ],
      [DOC2]: [
        { question: "What do plants turn light into?", answer: "Chemical energy" },
        { question: "_____ turn light into chemical energy", answer: "Plants" },
      ],
    } as Record<string, { question: string; answer: string }[]>,
    stats: {
      total_cards: 5,
      due_now: 2,
      due_today: 3,
      average_ef: 2.5,
      average_retention: 80,
      mature_cards: 1,
      learning_cards: 4,
    },
    // Nothing in the app records reading engagement yet, so the real profile is
    // all zeros with the default 50% comprehension score
    engagement: {
      user_id: USER.user_id,
      total_reading_time_minutes: 0,
      documents_read: 0,
      avg_session_duration_minutes: 0,
      avg_scroll_depth: 0,
      preferred_complexity: 50,
      total_highlights: 0,
      total_flashcards: 0,
      comprehension_score: 50.0,
    },
  };
}

export type MockData = ReturnType<typeof defaultData>;

let cardSeq = 0;

/** FastAPI's 422 for a query parameter above its `le` bound */
function limitTooHigh(value: string, max: number): MockResponse {
  return {
    status: 422,
    body: {
      detail: [
        {
          type: "less_than_equal",
          loc: ["query", "limit"],
          msg: `Input should be less than or equal to ${max}`,
          input: value,
          ctx: { le: max },
        },
      ],
    },
  };
}

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
      path: decodeURIComponent(url.pathname.slice("/api/v1".length)),
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
    if (response.waitFor) await response.waitFor;
    if (response.cors === false && call.headers["origin"] && call.headers["origin"] !== url.origin) {
      // Chromium does not apply CORS to responses fulfilled by Playwright, so do
      // what it does with a cross-origin response that has no CORS headers: the
      // app only sees a network error
      await route.abort("failed").catch(() => undefined);
      return;
    }
    const status = response.status ?? 200;
    const isText = response.text !== undefined;
    await route
      .fulfill({
        status,
        headers: {
          ...(response.cors === false ? {} : cors),
          "content-type": isText ? "text/plain; charset=utf-8" : "application/json",
        },
        body: status === 204 ? "" : isText ? response.text : JSON.stringify(response.body ?? {}),
      })
      // The page may have navigated away while a delayed response was pending
      .catch(() => undefined);
  }

  private installDefaults() {
    const d = this.data;
    const findDoc = (id: string) => d.documents.find((doc) => doc.doc_id === id);
    const notFound = (message: string): MockResponse => ({
      status: 404,
      body: { detail: { code: "NOT_FOUND", message } },
    });

    // Auth: any bearer token is valid; no refresh cookie
    this.on("GET", "/auth/me", (call) =>
      call.headers["authorization"]?.startsWith("Bearer ")
        ? { body: USER }
        : {
            status: 401,
            body: { detail: { code: "MISSING_TOKEN", message: "Authorization token required" } },
          }
    );
    this.on("POST", "/auth/refresh", NO_REFRESH_TOKEN);
    this.on("POST", "/auth/logout", { body: { message: "Logged out successfully" } });

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
      const doc = findDoc(m[1]);
      return doc ? { body: doc } : notFound(`Document not found: ${m[1]}`);
    });
    this.on("GET", /^\/documents\/([^/]+)\/content$/, (_call, m) => {
      const id = m[1];
      return findDoc(id) && d.content[id]
        ? { body: { doc_id: id, content: d.content[id] } }
        : notFound(`Document not found or has no content: ${id}`);
    });
    this.on("GET", /^\/documents\/([^/]+)\/paraphrase$/, (call, m) => {
      const id = m[1];
      // Without an LLM the backend returns the text itself
      return findDoc(id)
        ? {
            body: {
              doc_id: id,
              complexity: Number(call.query.get("complexity")),
              content: d.content[id] ?? "",
            },
          }
        : notFound(`Document not found: ${id}`);
    });
    this.on("DELETE", /^\/documents\/([^/]+)$/, (_call, m) => {
      const index = d.documents.findIndex((doc) => doc.doc_id === m[1]);
      if (index === -1) return notFound(`Document not found: ${m[1]}`);
      d.documents.splice(index, 1);
      return { status: 204 };
    });

    // Graph (nodes in a fixed order, at most `limit`)
    this.on("GET", "/graph/nodes", (call) => {
      const limit = call.query.get("limit") ?? "50";
      if (Number(limit) > 1000) return limitTooHigh(limit, 1000);
      const docId = call.query.get("source_doc_id");
      const nodes = (docId ? d.nodes.filter((n) => n.source_doc_id === docId) : d.nodes).slice(
        0,
        Number(limit)
      );
      return { body: { nodes, total: nodes.length } };
    });
    this.on("GET", /^\/graph\/documents\/([^/]+)\/relations$/, (call, m) => {
      const limit = call.query.get("limit") ?? "500";
      if (Number(limit) > 1000) return limitTooHigh(limit, 1000);
      if (!findDoc(m[1])) return notFound(`Document ${m[1]} not found`);
      const relations = (d.relations[m[1]] ?? []).slice(0, Number(limit));
      return { body: { relations, total: relations.length } };
    });
    this.on("GET", /^\/graph\/link\/wikipedia\/(.+)$/, (_call, m) => ({
      body: { entity: m[1], title: null, url: null, extract: null, found: false },
    }));

    // Learning
    this.on("GET", "/learning/flashcards/due", () => ({
      body: { due_cards: d.dueCards, total_due: d.dueCards.length },
    }));
    this.on("GET", "/learning/flashcards/stats", () => ({ body: d.stats }));
    this.on("GET", "/learning/engagement/profile", () => ({ body: d.engagement }));
    this.on("POST", "/learning/flashcards/review", (call) => {
      const { card_id, quality } = call.body as { card_id: string; quality: number };
      return {
        body: {
          card_id,
          quality,
          next_review: "2026-10-07T12:00:00+00:00",
          interval_days: 1,
          easiness_factor: 2.5,
          repetitions: 1,
          retention_rate: 100.0,
        },
      };
    });
    // Stores only cards whose question is new for this user and document
    this.on("POST", "/learning/flashcards/generate", (call) => {
      const { document_id } = call.body as { document_id: string };
      if (!findDoc(document_id)) return notFound(`Document ${document_id} not found`);
      if (!d.content[document_id]?.trim()) {
        return {
          status: 422,
          body: {
            detail: {
              code: "NO_CONTENT",
              message: `Document ${document_id} has no text to generate flashcards from`,
            },
          },
        };
      }
      const candidates = d.generated[document_id] ?? [];
      const existing = new Set(
        d.cards.filter((c) => c.document_id === document_id).map((c) => c.question)
      );
      const flashcards = candidates
        .filter((c) => !existing.has(c.question))
        .map((c) =>
          makeCard({
            card_id: `00000000-0000-4000-8000-${String(++cardSeq).padStart(12, "0")}`,
            document_id,
            ...c,
          })
        );
      for (const card of flashcards) {
        d.cards.push({ ...card, days_overdue: 0 });
        d.dueCards.push({ ...card, days_overdue: 0 });
      }
      return {
        body: {
          document_id,
          flashcards,
          count: flashcards.length,
          duplicates_skipped: candidates.length - flashcards.length,
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

/** The reader URL of a document (the app links to it the same way) */
export function readerUrl(docId: string) {
  return `/read/${encodeURIComponent(docId)}`;
}

/** The app's role="alert" elements, without Next's (empty) route announcer */
export function appAlerts(page: Page) {
  return page.locator('[role="alert"]:not(#__next-route-announcer__)');
}

/** The toast messages react-hot-toast is showing */
export function toasts(page: Page) {
  return page.locator('[role="status"][aria-live="polite"]');
}

/** The D3 graph's node circles and edge lines (not the icons around it) */
export function graphCircles(page: Page) {
  return page.locator('[data-testid="knowledge-graph"] g.nodes circle');
}

export function graphLines(page: Page) {
  return page.locator('[data-testid="knowledge-graph"] g.links line');
}
