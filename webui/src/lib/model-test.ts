import { getAdminToken } from "./auth";
import { backend } from "./backend";
import type { LogPage, RequestLog } from "./types";

/**
 * Ingress API types exposed by the gateway. Every entry maps to a real proxy
 * endpoint, so the playground can compare how the same model behaves across
 * OpenAI Chat/Responses, Anthropic Messages and Gemini GenerateContent.
 */
export type ApiTypeId =
  | "chat-completions"
  | "responses"
  | "anthropic-messages"
  | "gemini-generate-content";

export type ApiTypeOption = {
  id: ApiTypeId;
  name: string;
  path: string;
  supported: boolean;
};

export const API_TYPES: readonly ApiTypeOption[] = [
  {
    id: "chat-completions",
    name: "OpenAI · Chat Completions",
    path: "/v1/chat/completions",
    supported: true,
  },
  {
    id: "responses",
    name: "OpenAI · Responses",
    path: "/v1/responses",
    supported: true,
  },
  {
    id: "anthropic-messages",
    name: "Anthropic · Messages",
    path: "/v1/messages",
    supported: true,
  },
  {
    id: "gemini-generate-content",
    name: "Google · GenerateContent",
    path: "/v1beta/models/{model}:generateContent",
    supported: true,
  },
];

/** Ingress types that accept image parts; the Responses ingress is text-only. */
export function supportsImageAttachments(apiType: ApiTypeId): boolean {
  return apiType !== "responses";
}

/**
 * Resolves the ingress path for a request. Gemini carries the model in the path
 * (and streams through `:streamGenerateContent`), so it cannot use the static
 * `path` from `API_TYPES` at runtime.
 */
export function buildRequestPath(apiType: ApiTypeId, model: string): string {
  if (apiType === "gemini-generate-content") {
    return `/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`;
  }
  const option = API_TYPES.find((item) => item.id === apiType) ?? API_TYPES[0];
  return option.path;
}

/** Splits a `data:` URL into its mime type and raw base64 payload. */
function dataUrlPayload(dataUrl: string): { mime: string; data: string } | null {
  const comma = dataUrl.indexOf(",");
  if (comma < 0) return null;
  const header = dataUrl.slice(0, comma);
  const data = dataUrl.slice(comma + 1);
  if (!data) return null;
  const mime = /data:([^;]+)/.exec(header)?.[1] ?? "image/png";
  return { mime, data };
}

export const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

export type TestAttachment = {
  id: string;
  name: string;
  mime: string;
  size: number;
  dataUrl: string;
};

export type MessageMeta = {
  latencyMs?: number;
  firstTokenMs?: number;
  inputTokens?: number;
  outputTokens?: number;
  tps?: number;
};

export type ChatError = {
  status?: number;
  message: string;
  detail?: string;
  logId?: string | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Chain-of-thought / reasoning text, kept apart from the final answer. */
  thinking?: string;
  attachments: TestAttachment[];
  pending?: boolean;
  error?: ChatError;
  meta?: MessageMeta;
  /** Request log id produced by this turn, when the log collector caught up. */
  logId?: string | null;
};

export type ChatParams = {
  temperature: number;
  topP: number;
  /** Raw context/max token text, accepts shorthands such as `8k` or `1m`. */
  maxTokens: string;
  topK: number;
};

export const DEFAULT_PARAMS: ChatParams = {
  temperature: 1,
  topP: 0.95,
  maxTokens: "8k",
  topK: 20,
};

const TOKEN_UNITS: Record<string, number> = {
  k: 1024,
  m: 1024 * 1024,
};

/**
 * Parses a token budget such as `8192`, `8k` or `1m` (case-insensitive,
 * optional spaces). Returns `undefined` for empty or invalid input so the
 * field can be left out of the payload entirely.
 */
export function parseTokenBudget(input: string | number): number | undefined {
  if (typeof input === "number") {
    return Number.isFinite(input) && input > 0 ? Math.round(input) : undefined;
  }
  const text = input.trim().toLowerCase().replace(/[\s,_]/g, "");
  if (!text) return undefined;
  const match = /^(\d+(?:\.\d+)?)([km]?)$/.exec(text);
  if (!match) return undefined;
  const value = Number(match[1]) * (TOKEN_UNITS[match[2]] ?? 1);
  if (!Number.isFinite(value) || value <= 0) return undefined;
  return Math.round(value);
}

/** Formats a token budget back into the compact `8k` / `1m` notation. */
export function formatTokenBudget(value: number): string {
  if (value >= 1024 * 1024 && value % (1024 * 1024) === 0) return `${value / (1024 * 1024)}m`;
  if (value >= 1024 && value % 1024 === 0) return `${value / 1024}k`;
  return String(value);
}

let sequence = 0;

export function newId(prefix = "mt"): string {
  sequence += 1;
  return `${prefix}-${Date.now().toString(36)}-${sequence}`;
}

function imageAttachments(message: ChatMessage): TestAttachment[] {
  return message.attachments.filter((item) => item.mime.startsWith("image/"));
}

/** Builds an OpenAI Chat Completions payload from the visible transcript. */
export function buildChatPayload(input: {
  model: string;
  history: ChatMessage[];
  params: ChatParams;
}): Record<string, unknown> {
  const { model, history, params } = input;
  const messages: Array<Record<string, unknown>> = [];

  for (const message of history) {
    if (message.pending || message.error) continue;
    if (message.role === "user") {
      const images = imageAttachments(message);
      const text = message.text.trim();
      if (images.length === 0) {
        if (!text) continue;
        messages.push({ role: "user", content: message.text });
        continue;
      }
      const parts: Array<Record<string, unknown>> = [];
      if (message.text) {
        parts.push({ type: "text", text: message.text });
      }
      for (const image of images) {
        parts.push({ type: "image_url", image_url: { url: image.dataUrl } });
      }
      messages.push({ role: "user", content: parts });
      continue;
    }
    if (message.text.trim()) {
      messages.push({ role: "assistant", content: message.text });
    }
  }

  const payload: Record<string, unknown> = { model, messages };

  if (Number.isFinite(params.temperature)) payload.temperature = params.temperature;
  if (Number.isFinite(params.topP)) payload.top_p = params.topP;
  const maxTokens = parseTokenBudget(params.maxTokens);
  if (maxTokens) payload.max_tokens = maxTokens;
  if (Number.isFinite(params.topK) && params.topK > 0) {
    payload.top_k = Math.round(params.topK);
  }
  payload.stream = true;
  payload.stream_options = { include_usage: true };
  return payload;
}

export type BuildPayloadInput = {
  model: string;
  history: ChatMessage[];
  params: ChatParams;
};

/**
 * Builds an OpenAI Responses payload. The Responses ingress decodes text items
 * only, so image attachments are skipped here (the UI warns about it).
 */
export function buildResponsesPayload(input: BuildPayloadInput): Record<string, unknown> {
  const { model, history, params } = input;
  const items: Array<Record<string, unknown>> = [];

  for (const message of history) {
    if (message.pending || message.error) continue;
    const text = message.text.trim();
    if (!text) continue;
    items.push({
      role: message.role,
      content: [
        message.role === "user"
          ? { type: "input_text", text: message.text }
          : { type: "output_text", text: message.text },
      ],
    });
  }

  const payload: Record<string, unknown> = { model, input: items, stream: true };
  if (Number.isFinite(params.temperature)) payload.temperature = params.temperature;
  if (Number.isFinite(params.topP)) payload.top_p = params.topP;
  const maxTokens = parseTokenBudget(params.maxTokens);
  if (maxTokens) payload.max_output_tokens = maxTokens;
  return payload;
}

/** Builds an Anthropic Messages payload (images as base64 blocks). */
export function buildAnthropicPayload(input: BuildPayloadInput): Record<string, unknown> {
  const { model, history, params } = input;
  const messages: Array<Record<string, unknown>> = [];

  for (const message of history) {
    if (message.pending || message.error) continue;
    if (message.role === "user") {
      const blocks: Array<Record<string, unknown>> = [];
      if (message.text.trim()) blocks.push({ type: "text", text: message.text });
      for (const image of imageAttachments(message)) {
        const encoded = dataUrlPayload(image.dataUrl);
        if (!encoded) continue;
        blocks.push({
          type: "image",
          source: { type: "base64", media_type: encoded.mime, data: encoded.data },
        });
      }
      if (blocks.length === 0) continue;
      messages.push({ role: "user", content: blocks });
      continue;
    }
    if (message.text.trim()) {
      messages.push({ role: "assistant", content: message.text });
    }
  }

  const payload: Record<string, unknown> = {
    model,
    messages,
    // `max_tokens` is mandatory for the Messages ingress.
    max_tokens: parseTokenBudget(params.maxTokens) ?? 8192,
    stream: true,
  };
  if (Number.isFinite(params.temperature)) payload.temperature = params.temperature;
  if (Number.isFinite(params.topP)) payload.top_p = params.topP;
  if (Number.isFinite(params.topK) && params.topK > 0) {
    payload.top_k = Math.round(params.topK);
  }
  return payload;
}

/** Builds a Gemini GenerateContent payload (`camelCase` wire format). */
export function buildGeminiPayload(input: BuildPayloadInput): Record<string, unknown> {
  const { history, params } = input;
  const contents: Array<Record<string, unknown>> = [];

  for (const message of history) {
    if (message.pending || message.error) continue;
    const parts: Array<Record<string, unknown>> = [];
    if (message.text.trim()) parts.push({ text: message.text });
    if (message.role === "user") {
      for (const image of imageAttachments(message)) {
        const encoded = dataUrlPayload(image.dataUrl);
        if (!encoded) continue;
        parts.push({ inlineData: { mimeType: encoded.mime, data: encoded.data } });
      }
    }
    if (parts.length === 0) continue;
    contents.push({ role: message.role === "user" ? "user" : "model", parts });
  }

  const generationConfig: Record<string, unknown> = {};
  if (Number.isFinite(params.temperature)) generationConfig.temperature = params.temperature;
  if (Number.isFinite(params.topP)) generationConfig.topP = params.topP;
  if (Number.isFinite(params.topK) && params.topK > 0) {
    generationConfig.topK = Math.round(params.topK);
  }
  const maxTokens = parseTokenBudget(params.maxTokens);
  if (maxTokens) generationConfig.maxOutputTokens = maxTokens;

  const payload: Record<string, unknown> = { contents };
  if (Object.keys(generationConfig).length > 0) {
    payload.generationConfig = generationConfig;
  }
  return payload;
}

/** Dispatches to the payload builder matching the selected ingress type. */
export function buildPayload(
  apiType: ApiTypeId,
  input: BuildPayloadInput,
): Record<string, unknown> {
  switch (apiType) {
    case "responses":
      return buildResponsesPayload(input);
    case "anthropic-messages":
      return buildAnthropicPayload(input);
    case "gemini-generate-content":
      return buildGeminiPayload(input);
    default:
      return buildChatPayload(input);
  }
}

export type StreamDelta = {
  text?: string;
  /** Reasoning / chain-of-thought delta, rendered separately from the answer. */
  thinking?: string;
  usage?: { promptTokens?: number; completionTokens?: number };
};

function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let out = "";
  for (const part of content) {
    if (typeof part === "string") {
      out += part;
    } else if (part && typeof part === "object") {
      const text = (part as { text?: unknown }).text;
      if (typeof text === "string") out += text;
    }
  }
  return out;
}

/** Parses one Chat Completions SSE payload. */
function parseChatData(data: string, onDelta: (delta: StreamDelta) => void): void {
  if (!data || data === "[DONE]") return;

  let event: unknown;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  if (!event || typeof event !== "object") return;
  const record = event as Record<string, unknown>;
  const choices = record.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const choice = choices[0] as Record<string, unknown> | null;
    if (choice) {
      const delta = (choice.delta ?? choice.message) as Record<string, unknown> | undefined;
      const text = textFromContent(delta?.content);
      if (text) onDelta({ text });
      const reasoning = textFromContent(
        delta?.reasoning_content ?? delta?.reasoning ?? delta?.thinking,
      );
      if (reasoning) onDelta({ thinking: reasoning });
    }
  }
  const usage = record.usage as Record<string, unknown> | null | undefined;
  if (usage && typeof usage === "object") {
    const promptTokens = Number(usage.prompt_tokens);
    const completionTokens = Number(usage.completion_tokens);
    onDelta({
      usage: {
        promptTokens: Number.isFinite(promptTokens) ? promptTokens : undefined,
        completionTokens: Number.isFinite(completionTokens) ? completionTokens : undefined,
      },
    });
  }
}

/**
 * Reads an SSE body and hands each `data:` payload to `onData`. Shared by every
 * ingress type because only the payload shape differs between them.
 */
async function readSseData(
  response: Response,
  onData: (data: string) => void,
): Promise<void> {
  if (!response.body) {
    throw new Error("empty response body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flush = (block: string) => {
    for (const line of block.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (data) onData(data);
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) flush(block);
  }

  buffer += decoder.decode();
  if (buffer.trim()) flush(buffer);
}

/** Parses one OpenAI Responses SSE payload. */
function parseResponsesData(data: string, onDelta: (delta: StreamDelta) => void): void {
  if (!data || data === "[DONE]") return;

  let event: unknown;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  if (!event || typeof event !== "object") return;
  const record = event as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : "";

  if (type === "response.output_text.delta") {
    const delta = record.delta;
    if (typeof delta === "string" && delta) onDelta({ text: delta });
    return;
  }
  if (
    type === "response.reasoning_summary_text.delta" ||
    type === "response.reasoning_text.delta"
  ) {
    const delta = record.delta;
    if (typeof delta === "string" && delta) onDelta({ thinking: delta });
    return;
  }
  if (type === "response.completed" || type === "response.incomplete") {
    const response = record.response as Record<string, unknown> | undefined;
    const usage = response?.usage as Record<string, unknown> | undefined;
    if (usage && typeof usage === "object") {
      const promptTokens = Number(usage.input_tokens);
      const completionTokens = Number(usage.output_tokens);
      onDelta({
        usage: {
          promptTokens: Number.isFinite(promptTokens) ? promptTokens : undefined,
          completionTokens: Number.isFinite(completionTokens) ? completionTokens : undefined,
        },
      });
    }
    return;
  }
  if (type === "error") {
    const error = record.error as Record<string, unknown> | undefined;
    const message = typeof error?.message === "string" ? error.message : "stream error";
    throw new Error(message);
  }
}

/** Parses one Anthropic Messages SSE payload. */
function parseAnthropicData(data: string, onDelta: (delta: StreamDelta) => void): void {
  if (!data || data === "[DONE]") return;

  let event: unknown;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  if (!event || typeof event !== "object") return;
  const record = event as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : "";

  if (type === "content_block_delta") {
    const delta = record.delta as Record<string, unknown> | undefined;
    if (!delta) return;
    const kind = typeof delta.type === "string" ? delta.type : "";
    if (kind === "text_delta" && typeof delta.text === "string" && delta.text) {
      onDelta({ text: delta.text });
      return;
    }
    if (kind === "thinking_delta" && typeof delta.thinking === "string" && delta.thinking) {
      onDelta({ thinking: delta.thinking });
    }
    return;
  }

  const usageSource =
    type === "message_start"
      ? ((record.message as Record<string, unknown> | undefined)?.usage as
          | Record<string, unknown>
          | undefined)
      : type === "message_delta"
        ? (record.usage as Record<string, unknown> | undefined)
        : undefined;
  if (usageSource && typeof usageSource === "object") {
    const promptTokens = Number(usageSource.input_tokens);
    const completionTokens = Number(usageSource.output_tokens);
    const hasPrompt = Number.isFinite(promptTokens) && promptTokens > 0;
    const hasCompletion = Number.isFinite(completionTokens) && completionTokens > 0;
    if (hasPrompt || hasCompletion) {
      onDelta({
        usage: {
          promptTokens: hasPrompt ? promptTokens : undefined,
          completionTokens: hasCompletion ? completionTokens : undefined,
        },
      });
    }
    return;
  }

  if (type === "error") {
    const error = record.error as Record<string, unknown> | undefined;
    const message = typeof error?.message === "string" ? error.message : "stream error";
    throw new Error(message);
  }
}

/** Parses one Gemini GenerateContent SSE payload. */
function parseGeminiData(data: string, onDelta: (delta: StreamDelta) => void): void {
  if (!data || data === "[DONE]") return;

  let event: unknown;
  try {
    event = JSON.parse(data);
  } catch {
    return;
  }
  if (!event || typeof event !== "object") return;
  const record = event as Record<string, unknown>;

  const candidates = record.candidates;
  if (Array.isArray(candidates) && candidates.length > 0) {
    const candidate = candidates[0] as Record<string, unknown> | null;
    const content = candidate?.content as Record<string, unknown> | undefined;
    const parts = content?.parts;
    if (Array.isArray(parts)) {
      let text = "";
      for (const part of parts) {
        if (part && typeof part === "object") {
          const value = (part as { text?: unknown }).text;
          if (typeof value === "string") text += value;
        }
      }
      if (text) onDelta({ text });
    }
  }

  const usage = record.usageMetadata as Record<string, unknown> | undefined;
  if (usage && typeof usage === "object") {
    const promptTokens = Number(usage.promptTokenCount);
    const completionTokens = Number(usage.candidatesTokenCount);
    if (Number.isFinite(promptTokens) || Number.isFinite(completionTokens)) {
      onDelta({
        usage: {
          promptTokens: Number.isFinite(promptTokens) ? promptTokens : undefined,
          completionTokens: Number.isFinite(completionTokens) ? completionTokens : undefined,
        },
      });
    }
  }

  const error = record.error as Record<string, unknown> | undefined;
  if (error && typeof error === "object") {
    const message = typeof error.message === "string" ? error.message : "stream error";
    throw new Error(message);
  }
}

/** Reads an SSE response, invoking `onDelta` for every content/usage chunk. */
export async function consumeChatStream(
  response: Response,
  onDelta: (delta: StreamDelta) => void,
): Promise<void> {
  await readSseData(response, (data) => parseChatData(data, onDelta));
}

/** Reads the SSE stream of the selected ingress type. */
export async function consumeStream(
  apiType: ApiTypeId,
  response: Response,
  onDelta: (delta: StreamDelta) => void,
): Promise<void> {
  switch (apiType) {
    case "responses":
      await readSseData(response, (data) => parseResponsesData(data, onDelta));
      return;
    case "anthropic-messages":
      await readSseData(response, (data) => parseAnthropicData(data, onDelta));
      return;
    case "gemini-generate-content":
      await readSseData(response, (data) => parseGeminiData(data, onDelta));
      return;
    default:
      await consumeChatStream(response, onDelta);
  }
}

export type ParsedResponse = {
  text: string;
  usage?: { promptTokens?: number; completionTokens?: number };
};

/** Extracts content/usage from a non-streaming Chat Completions response. */
export function parseChatResponse(body: unknown): ParsedResponse {
  if (!body || typeof body !== "object") {
    return { text: "" };
  }
  const record = body as Record<string, unknown>;
  const choices = record.choices;
  let text = "";
  if (Array.isArray(choices) && choices.length > 0) {
    const choice = choices[0] as Record<string, unknown> | null;
    const message = choice?.message as Record<string, unknown> | undefined;
    text = textFromContent(message?.content);
    const reasoning = textFromContent(message?.reasoning_content ?? message?.reasoning);
    if (reasoning) text = `${reasoning}${text}`;
  }
  const usage = record.usage as Record<string, unknown> | null | undefined;
  let parsedUsage: ParsedResponse["usage"];
  if (usage && typeof usage === "object") {
    const promptTokens = Number(usage.prompt_tokens);
    const completionTokens = Number(usage.completion_tokens);
    parsedUsage = {
      promptTokens: Number.isFinite(promptTokens) ? promptTokens : undefined,
      completionTokens: Number.isFinite(completionTokens) ? completionTokens : undefined,
    };
  }
  return { text, usage: parsedUsage };
}

const ERROR_DETAIL_LIMIT = 8000;

function truncate(value: string): string {
  return value.length > ERROR_DETAIL_LIMIT ? `${value.slice(0, ERROR_DETAIL_LIMIT)}\n…` : value;
}

/** Normalizes a gateway error body into a short message plus raw details. */
export async function readErrorResponse(
  response: Response,
): Promise<{ status: number; message: string; detail?: string }> {
  const raw = await response.text().catch(() => "");
  const detail = raw ? truncate(raw) : undefined;
  let message = `HTTP ${response.status}`;

  if (raw) {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (parsed && typeof parsed === "object") {
        const record = parsed as Record<string, unknown>;
        const error = record.error;
        if (typeof error === "string" && error.trim()) {
          message = error.trim();
        } else if (error && typeof error === "object") {
          const inner = error as Record<string, unknown>;
          if (typeof inner.message === "string" && inner.message.trim()) {
            message = inner.message.trim();
          }
        } else if (typeof record.message === "string" && record.message.trim()) {
          message = record.message.trim();
        }
      }
    } catch {
      const firstLine = raw.split(/\r?\n/).find((line) => line.trim());
      if (firstLine) message = `${message} — ${firstLine.trim().slice(0, 200)}`;
    }
  }

  return { status: response.status, message, detail };
}

/** Posts a Chat Completions payload through the admin bridge to the proxy. */
export async function sendChatRequest(input: {
  apiKey?: string | null;
  payload: unknown;
  /** Ingress path to exercise; defaults to Chat Completions server-side. */
  path?: string;
  signal: AbortSignal;
}): Promise<Response> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = getAdminToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  return fetch("/api/v1/playground/chat", {
    method: "POST",
    headers,
    body: JSON.stringify({
      api_key: input.apiKey?.trim() ? input.apiKey.trim() : null,
      path: input.path ?? null,
      payload: input.payload,
    }),
    signal: input.signal,
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

/**
 * Finds the request log entry produced by a playground call so the UI can jump
 * straight to it and backfill real token counts. The log collector is
 * asynchronous, hence the short retries.
 */
export async function findRequestLog(model: string, since: number): Promise<RequestLog | null> {
  for (let attempt = 0; attempt < 4; attempt += 1) {
    await sleep(attempt === 0 ? 300 : 600);
    try {
      const page = await backend<LogPage>("query_logs", {
        query: { limit: 10, offset: 0 },
      });
      const hit = page.items.find(
        (item) => (item.client_model ?? "") === model && item.created_at >= since - 2000,
      );
      if (hit) return hit;
    } catch {
      // Log lookup is best-effort; keep the request result intact.
    }
  }
  return null;
}

/** Approximate output throughput, measured after the first streamed token. */
export function computeMetaTps(meta: MessageMeta): number | undefined {
  if (!meta.outputTokens || !meta.latencyMs) return undefined;
  const start =
    meta.firstTokenMs && meta.firstTokenMs < meta.latencyMs ? meta.firstTokenMs : meta.latencyMs;
  const window = meta.latencyMs - start;
  const seconds = (window > 0 ? window : meta.latencyMs) / 1000;
  if (seconds <= 0) return undefined;
  return meta.outputTokens / seconds;
}
