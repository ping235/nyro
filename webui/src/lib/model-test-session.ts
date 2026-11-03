import type { ApiTypeId, ChatMessage, ChatParams, TestAttachment } from "@/lib/model-test";

/**
 * Browser-side persistence for the model-test playground.
 *
 * The conversation (plus composer draft, request config and request counter) is written
 * to localStorage so it survives both route switches and full page reloads. It stays
 * until the user presses 「清空」, which drops the stored entry.
 */

export type ModelTestSnapshot = {
  apiType: ApiTypeId;
  modelName: string;
  apiKeyId: string;
  params: ChatParams;
  history: ChatMessage[];
  draft: string;
  attachments: TestAttachment[];
  requestCount: number;
};

export const MODEL_TEST_SESSION_STORAGE_KEY = "nyro-model-test-session";

/** Bumped when the persisted shape changes; older payloads are simply discarded. */
const STORAGE_VERSION = 1;

/**
 * localStorage quotas sit around 5 MB per origin, and JSON payloads are measured in
 * UTF-16 code units, so keep a comfortable margin below it. Image attachments are
 * base64 data URLs (up to 5 MB each), which is why the payload is budgeted rather
 * than written as-is.
 */
const MAX_SERIALIZED_CHARS = 1_500_000;

/** Messages kept when the payload still does not fit after dropping image payloads. */
const MIN_KEPT_MESSAGES = 12;

type StoredPayload = {
  version: number;
  snapshot: ModelTestSnapshot;
};

function storage(): Storage | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Keeps attachment metadata but drops the base64 image data. */
function withoutImageData(snapshot: ModelTestSnapshot): ModelTestSnapshot {
  const history = snapshot.history.map((message) =>
    message.attachments.length === 0
      ? message
      : {
          ...message,
          attachments: message.attachments.map((attachment) => ({ ...attachment, dataUrl: "" })),
        },
  );
  return { ...snapshot, history, attachments: [] };
}

/** Drops the oldest messages, keeping the most recent `keep` turns. */
function trimHistory(snapshot: ModelTestSnapshot, keep: number): ModelTestSnapshot {
  if (snapshot.history.length <= keep) return snapshot;
  return { ...snapshot, history: snapshot.history.slice(snapshot.history.length - keep) };
}

function serialize(snapshot: ModelTestSnapshot): string {
  const payload: StoredPayload = { version: STORAGE_VERSION, snapshot };
  return JSON.stringify(payload);
}

/**
 * Persists the playground state, degrading the payload (image data → oldest messages)
 * until it fits the budget. Storage failures are swallowed: persistence is a
 * convenience and must never break the panel.
 */
export function saveModelTestSnapshot(snapshot: ModelTestSnapshot): void {
  const store = storage();
  if (!store) return;

  let candidate = snapshot;
  let json = serialize(candidate);

  if (json.length > MAX_SERIALIZED_CHARS) {
    candidate = withoutImageData(candidate);
    json = serialize(candidate);
  }

  while (json.length > MAX_SERIALIZED_CHARS && candidate.history.length > MIN_KEPT_MESSAGES) {
    const keep = Math.max(MIN_KEPT_MESSAGES, Math.ceil(candidate.history.length / 2));
    candidate = trimHistory(candidate, keep);
    json = serialize(candidate);
  }

  try {
    store.setItem(MODEL_TEST_SESSION_STORAGE_KEY, json);
  } catch {
    // Quota exceeded (or storage disabled): retry once with the smallest useful payload.
    try {
      store.setItem(
        MODEL_TEST_SESSION_STORAGE_KEY,
        serialize(trimHistory(withoutImageData(snapshot), MIN_KEPT_MESSAGES)),
      );
    } catch {
      try {
        store.removeItem(MODEL_TEST_SESSION_STORAGE_KEY);
      } catch {
        // Nothing else to do — the in-memory module snapshot still holds the session.
      }
    }
  }
}

/** Reads the stored session back, returning null when absent or from another version. */
export function loadModelTestSnapshot(): ModelTestSnapshot | null {
  const store = storage();
  if (!store) return null;
  try {
    const raw = store.getItem(MODEL_TEST_SESSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredPayload> | null;
    if (!parsed || parsed.version !== STORAGE_VERSION) return null;
    const snapshot = parsed.snapshot;
    if (!snapshot || !Array.isArray(snapshot.history) || !Array.isArray(snapshot.attachments)) {
      return null;
    }
    return snapshot;
  } catch {
    return null;
  }
}

export function clearModelTestSnapshot(): void {
  const store = storage();
  if (!store) return;
  try {
    store.removeItem(MODEL_TEST_SESSION_STORAGE_KEY);
  } catch {
    // Ignore: an unreadable store cannot hold stale data we could do anything about.
  }
}