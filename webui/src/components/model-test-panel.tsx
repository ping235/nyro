import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  ArrowUp,
  ChevronDown,
  ExternalLink,
  Paperclip,
  RotateCcw,
  Settings,
  Square,
  Trash2,
  X,
} from "lucide-react";

import { Combobox } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import MarkdownView from "@/components/markdown-view";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDuration, formatTokenCount, formatTps } from "@/lib/format";
import { useLocale } from "@/lib/i18n";
import {
  API_TYPES,
  DEFAULT_PARAMS,
  MAX_IMAGE_BYTES,
  buildPayload,
  buildRequestPath,
  computeMetaTps,
  consumeStream,
  findRequestLog,
  newId,
  readErrorResponse,
  sendChatRequest,
  supportsImageAttachments,
} from "@/lib/model-test";
import type {
  ApiTypeId,
  ChatMessage,
  ChatParams,
  MessageMeta,
  StreamDelta,
  TestAttachment,
} from "@/lib/model-test";
import type { ApiKey, Model as ModelType } from "@/lib/types";

type ModelTestPanelProps = {
  models: ModelType[];
  apiKeys: ApiKey[];
};

type KeyOption = {
  id: string;
  name: string;
  key?: string;
};

type PendingUsage = StreamDelta["usage"];

type ConfigDraft = {
  apiType: ApiTypeId;
  modelName: string;
  apiKeyId: string;
  params: ChatParams;
};

const EMPTY_HISTORY: ChatMessage[] = [];

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("read failed"));
    reader.readAsDataURL(file);
  });
}

/** Collapsible chain-of-thought block, visually separated from the answer. */
function ThinkingBlock({
  text,
  pending,
  isZh,
}: {
  text: string;
  pending?: boolean;
  isZh: boolean;
}) {
  const [open, setOpen] = useState(Boolean(pending));
  const wasPending = useRef(Boolean(pending));

  useEffect(() => {
    if (pending) {
      wasPending.current = true;
      setOpen(true);
      return;
    }
    if (wasPending.current) {
      wasPending.current = false;
      setOpen(false);
    }
  }, [pending]);

  return (
    <div
      className="mb-2 rounded-xl border px-3 py-2"
      style={{
        borderColor: "rgba(148,163,184,0.32)",
        background: "rgba(148,163,184,0.12)",
      }}
    >
      <button
        type="button"
        className="model-test-muted flex w-full items-center gap-1.5 text-xs font-medium"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        {pending ? (
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400" />
        ) : (
          <span className="h-1.5 w-1.5 rounded-full bg-slate-400/70" />
        )}
        <span>{pending ? (isZh ? "思考中…" : "Thinking…") : isZh ? "思考过程" : "Thinking"}</span>
        <ChevronDown
          className={
            open
              ? "ml-auto h-3 w-3 rotate-180 transition-transform"
              : "ml-auto h-3 w-3 transition-transform"
          }
        />
      </button>
      {open && text ? (
        <div className="model-test-muted mt-1.5 whitespace-normal break-words text-xs leading-relaxed">
          <MarkdownView text={text} tone="muted" />
        </div>
      ) : null}
    </div>
  );
}

/** Model playground rendered inside the "模型测试" tab of the connect page. */
export function ModelTestPanel({ models, apiKeys }: ModelTestPanelProps) {
  const { locale } = useLocale();
  const isZh = locale === "zh-CN";
  const navigate = useNavigate();

  const [apiType, setApiType] = useState<ApiTypeId>("chat-completions");
  const [modelName, setModelName] = useState("");
  const [apiKeyId, setApiKeyId] = useState("");
  const [params, setParams] = useState<ChatParams>(DEFAULT_PARAMS);
  const [history, setHistory] = useState<ChatMessage[]>(EMPTY_HISTORY);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<TestAttachment[]>([]);
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [requestCount, setRequestCount] = useState(0);
  const [expandedErrorId, setExpandedErrorId] = useState<string | null>(null);
  const [configOpen, setConfigOpen] = useState(false);
  const [configDraft, setConfigDraft] = useState<ConfigDraft>({
    apiType: "chat-completions",
    modelName: "",
    apiKeyId: "",
    params: DEFAULT_PARAMS,
  });

  const abortRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const enabledModels = useMemo(() => models.filter((item) => item.is_enabled), [models]);
  const selectedModel = useMemo(
    () => models.find((item) => item.name === modelName) ?? null,
    [models, modelName],
  );
  const selectedApiType = useMemo(
    () => API_TYPES.find((item) => item.id === apiType) ?? API_TYPES[0],
    [apiType],
  );
  const keyOptions = useMemo<KeyOption[]>(
    () => [
      { id: "", name: isZh ? "不使用密钥" : "No API key" },
      ...apiKeys.filter((item) => item.is_enabled).map((item) => ({ id: item.id, name: item.name, key: item.key })),
    ],
    [apiKeys, isZh],
  );
  const selectedKeyOption = useMemo(
    () => keyOptions.find((item) => item.id === apiKeyId) ?? null,
    [keyOptions, apiKeyId],
  );

  const draftApiType = useMemo(
    () => API_TYPES.find((item) => item.id === configDraft.apiType) ?? API_TYPES[0],
    [configDraft.apiType],
  );
  const draftModel = useMemo(
    () => models.find((item) => item.name === configDraft.modelName) ?? null,
    [models, configDraft.modelName],
  );

  const openConfig = useCallback(() => {
    setConfigDraft({ apiType, modelName, apiKeyId, params });
    setConfigOpen(true);
  }, [apiType, modelName, apiKeyId, params]);

  const closeConfig = useCallback(() => {
    setConfigOpen(false);
  }, []);

  const applyConfig = useCallback(() => {
    setApiType(configDraft.apiType);
    setModelName(configDraft.modelName);
    setApiKeyId(configDraft.apiKeyId);
    setParams(configDraft.params);
    setConfigOpen(false);
    setNotice(null);
  }, [configDraft]);

  useEffect(() => {
    if (modelName && models.some((item) => item.name === modelName)) return;
    const fallback = enabledModels[0] ?? models[0];
    setModelName(fallback?.name ?? "");
  }, [models, enabledModels, modelName]);

  useEffect(() => {
    if (!selectedModel?.enable_auth || apiKeyId) return;
    const fallback = keyOptions.find((item) => item.id !== "");
    if (fallback) setApiKeyId(fallback.id);
  }, [selectedModel, keyOptions, apiKeyId]);

  const streaming = history.some((message) => message.pending);
  const streamingRef = useRef(streaming);
  streamingRef.current = streaming;

  // While a response streams in, pin the pane to the bottom every frame so the newest
  // line stays visible even when markdown blocks re-layout in the same frame.
  useEffect(() => {
    const node = messagesRef.current;
    if (!node) return;
    let frame = 0;
    const pin = () => {
      if (streamingRef.current) {
        const gap = node.scrollHeight - node.clientHeight - node.scrollTop;
        if (gap > 0) node.scrollTop = node.scrollHeight;
      }
      frame = requestAnimationFrame(pin);
    };
    frame = requestAnimationFrame(pin);
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    const node = messagesRef.current;
    if (!node) return;
    const frame = requestAnimationFrame(() => {
      node.scrollTop = node.scrollHeight;
    });
    return () => cancelAnimationFrame(frame);
  }, [history]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const updateDraftParams = useCallback((patch: Partial<ChatParams>) => {
    setConfigDraft((prev) => ({ ...prev, params: { ...prev.params, ...patch } }));
  }, []);

  const addFiles = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files) return;
      const list = Array.from(files);
      if (list.length === 0) return;
      for (const file of list) {
        if (!file.type.startsWith("image/")) {
          setNotice(isZh ? "目前仅支持图片附件" : "Only image attachments are supported");
          continue;
        }
        if (file.size > MAX_IMAGE_BYTES) {
          setNotice(isZh ? "单张图片不能超过 5 MB" : "Each image must be under 5 MB");
          continue;
        }
        const dataUrl = await readAsDataUrl(file);
        setAttachments((prev) => [
          ...prev,
          { id: newId("att"), name: file.name, mime: file.type, size: file.size, dataUrl },
        ]);
        setNotice(null);
      }
    },
    [isZh],
  );

  const patchMessage = useCallback((id: string, updater: (message: ChatMessage) => ChatMessage) => {
    setHistory((prev) => prev.map((message) => (message.id === id ? updater(message) : message)));
  }, []);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleSend = useCallback(async () => {
    if (sending) return;
    if (!selectedModel) {
      setNotice(isZh ? "请选择一个模型" : "Select a model first");
      return;
    }
    if (!selectedApiType?.supported) {
      setNotice(isZh ? "该 API 类型即将支持" : "This API type is not available yet");
      return;
    }
    if (selectedModel.enable_auth && !selectedKeyOption?.key) {
      setNotice(isZh ? "该模型要求填写 API Key" : "This model requires an API key");
      return;
    }
    if (!draft.trim() && attachments.length === 0) return;
    if (attachments.length > 0 && !supportsImageAttachments(apiType)) {
      setNotice(
        isZh
          ? "该 API 类型的网关入口暂不支持图片附件，请移除图片或切换 API 类型"
          : "This API type does not accept image attachments on the gateway; remove them or switch types",
      );
      return;
    }

    const userMessage: ChatMessage = {
      id: newId("user"),
      role: "user",
      text: draft,
      attachments,
    };
    const assistantId = newId("assistant");
    const outbound = [...history, userMessage];
    const snapshotParams = params;

    setHistory([
      ...outbound,
      {
        id: assistantId,
        role: "assistant",
        text: "",
        attachments: [],
        pending: true,
      },
    ]);
    setDraft("");
    setAttachments([]);
    setNotice(null);
    setRequestCount((prev) => prev + 1);

    const controller = new AbortController();
    abortRef.current = controller;
    setSending(true);

    const requestedAt = Date.now();
    const startedAt = performance.now();
    let firstTokenMs: number | undefined;
    let streamed = "";
    let streamedThinking = "";
    let usage: PendingUsage;

    try {
      const activeApiType = selectedApiType.id;
      const payload = buildPayload(activeApiType, {
        model: selectedModel.name,
        history: outbound,
        params: snapshotParams,
      });
      const response = await sendChatRequest({
        apiKey: selectedKeyOption?.key,
        path: buildRequestPath(activeApiType, selectedModel.name),
        payload,
        signal: controller.signal,
      });

      if (!response.ok) {
        const failure = await readErrorResponse(response);
        const log = await findRequestLog(selectedModel.name, requestedAt);
        patchMessage(assistantId, (message) => ({
          ...message,
          pending: false,
          logId: log?.id ?? null,
          error: {
            status: failure.status,
            message: failure.message,
            detail: failure.detail,
            logId: log?.id ?? null,
          },
        }));
        setExpandedErrorId(assistantId);
        return;
      }

      await consumeStream(activeApiType, response, (delta) => {
        if (delta.thinking) {
          streamedThinking += delta.thinking;
          const currentThinking = streamedThinking;
          patchMessage(assistantId, (message) => ({ ...message, thinking: currentThinking }));
        }
        if (delta.text) {
          if (firstTokenMs === undefined) {
            firstTokenMs = performance.now() - startedAt;
          }
          streamed += delta.text;
          const current = streamed;
          patchMessage(assistantId, (message) => ({ ...message, text: current }));
        }
        if (delta.usage) usage = delta.usage;
      });

      const latencyMs = Math.round(performance.now() - startedAt);
      const log = await findRequestLog(selectedModel.name, requestedAt);
      const meta: MessageMeta = {
        latencyMs,
        firstTokenMs:
          firstTokenMs !== undefined ? Math.round(firstTokenMs) : (log?.stream_first_chunk_ms ?? undefined),
        inputTokens: usage?.promptTokens ?? log?.input_tokens ?? undefined,
        outputTokens: usage?.completionTokens ?? log?.output_tokens ?? undefined,
      };
      meta.tps = computeMetaTps(meta) ?? undefined;

      patchMessage(assistantId, (message) => ({
        ...message,
        pending: false,
        text: streamed || (isZh ? "（空响应）" : "(empty response)"),
        meta,
        logId: log?.id ?? null,
        error: undefined,
      }));
    } catch (error) {
      const aborted = error instanceof DOMException && error.name === "AbortError";
      const latencyMs = Math.round(performance.now() - startedAt);
      if (aborted) {
        patchMessage(assistantId, (message) => ({
          ...message,
          pending: false,
          text: streamed || (isZh ? "已停止生成" : "Generation stopped"),
          meta: { latencyMs },
        }));
      } else {
        const log = await findRequestLog(selectedModel.name, requestedAt).catch(() => null);
        patchMessage(assistantId, (message) => ({
          ...message,
          pending: false,
          logId: log?.id ?? null,
          error: {
            message: isZh ? "请求失败" : "Request failed",
            detail: error instanceof Error ? error.message : String(error),
            logId: log?.id ?? null,
          },
        }));
        setExpandedErrorId(assistantId);
      }
    } finally {
      abortRef.current = null;
      setSending(false);
    }
  }, [
    apiType,
    attachments,
    draft,
    history,
    isZh,
    params,
    patchMessage,
    selectedApiType,
    selectedKeyOption,
    selectedModel,
    sending,
  ]);

  const openLog = useCallback(
    (message: ChatMessage) => {
      const logId = message.logId ?? message.error?.logId ?? null;
      if (logId) {
        navigate(`/logs?log=${encodeURIComponent(logId)}`);
        return;
      }
      const target = selectedModel?.name ?? "";
      navigate(target ? `/logs?model=${encodeURIComponent(target)}` : "/logs");
    },
    [navigate, selectedModel],
  );

  const clearConversation = useCallback(() => {
    abortRef.current?.abort();
    setHistory([]);
    setRequestCount(0);
    setNotice(null);
    setExpandedErrorId(null);
  }, []);

  const canSend = Boolean(selectedModel) && selectedApiType.supported && !sending;
  const hasModels = models.length > 0;

  return (
    <div
      className={`model-test-scope grid grid-cols-1 gap-4 ${
        configOpen ? "xl:grid-cols-[minmax(0,1fr)_minmax(300px,340px)]" : "xl:grid-cols-1"
      }`}
    >
      {configOpen ? (
        <aside className="glass order-2 flex h-[min(720px,calc(100vh-260px))] min-h-[520px] flex-col overflow-hidden rounded-2xl">
          <div className="model-test-divider flex items-center justify-between border-b px-4 py-3">
            <h3 className="text-sm font-semibold">
              {isZh ? "请求配置" : "Configuration"}
            </h3>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              title={isZh ? "重置参数" : "Reset parameters"}
              onClick={() => {
                setConfigDraft((prev) => ({ ...prev, params: DEFAULT_PARAMS }));
                setNotice(null);
              }}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
            <div className="space-y-2">
              <p className="ml-1 text-xs leading-none font-normal text-slate-900">
                {isZh ? "API 类型" : "API type"}
              </p>
              <Combobox
                className="bg-white"
                value={draftApiType.id}
                onValueChange={(value) =>
                  setConfigDraft((prev) => ({ ...prev, apiType: value as ApiTypeId }))
                }
                options={API_TYPES.map((option) => ({
                  value: option.id,
                  label: option.supported
                    ? option.name
                    : `${option.name} · ${isZh ? "即将支持" : "coming soon"}`,
                }))}
                placeholder={isZh ? "API 类型" : "API type"}
                searchPlaceholder={isZh ? "搜索 API 类型..." : "Search API types..."}
                emptyText={isZh ? "暂无可选类型" : "No options"}
              />
              <p className="model-test-muted ml-1 text-xs">
                <code>{draftApiType.path}</code>
                {!draftApiType.supported ? (
                  <span className="ml-1 text-amber-600">{isZh ? "（即将支持）" : "(coming soon)"}</span>
                ) : null}
              </p>
            </div>

            <div className="space-y-2">
              <p className="ml-1 text-xs leading-none font-normal text-slate-900">
                {isZh ? "模型" : "Model"}
              </p>
              <Combobox
                className="bg-white"
                value={draftModel?.name ?? ""}
                onValueChange={(value) =>
                  setConfigDraft((prev) => ({ ...prev, modelName: value }))
                }
                options={models.map((option) => ({ value: option.name, label: option.name }))}
                placeholder={
                  hasModels
                    ? isZh
                      ? "选择模型"
                      : "Select model"
                    : isZh
                      ? "请先创建模型"
                      : "Create a model first"
                }
                searchPlaceholder={isZh ? "搜索模型..." : "Search models..."}
                emptyText={isZh ? "没有匹配的模型" : "No matching model"}
              />
              {!hasModels ? (
                <p className="ml-1 text-xs text-amber-600">
                  {isZh ? "还没有创建任何模型路由" : "No model routes yet"}
                </p>
              ) : draftModel?.enable_auth ? (
                <p className="model-test-muted ml-1 text-xs">
                  {isZh ? "该模型开启了鉴权，需选择 API Key" : "Auth enabled, an API key is required"}
                </p>
              ) : null}
            </div>

            <div className="space-y-2">
              <p className="ml-1 text-xs leading-none font-normal text-slate-900">
                {isZh ? "API Key" : "API key"}
              </p>
              <Combobox
                className="bg-white"
                value={configDraft.apiKeyId}
                onValueChange={(value) =>
                  setConfigDraft((prev) => ({ ...prev, apiKeyId: value }))
                }
                options={keyOptions.map((option) => ({ value: option.id, label: option.name }))}
                placeholder={isZh ? "选择 API Key" : "Select API key"}
                searchPlaceholder={isZh ? "搜索 API Key..." : "Search API keys..."}
                emptyText={isZh ? "暂无可选 API Key" : "No API keys available"}
              />
            </div>

            <div className="model-test-muted pt-1 text-xs font-medium uppercase tracking-wide">
              {isZh ? "生成参数" : "Parameters"}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <p className="ml-1 text-xs leading-none font-normal text-slate-900">
                  temperature
                </p>
                <Input
                  className="bg-white"
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  value={configDraft.params.temperature}
                  onChange={(event) => updateDraftParams({ temperature: Number(event.target.value) })}
                />
              </div>
              <div className="space-y-2">
                <p className="ml-1 text-xs leading-none font-normal text-slate-900">top_p</p>
                <Input
                  className="bg-white"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={configDraft.params.topP}
                  onChange={(event) => updateDraftParams({ topP: Number(event.target.value) })}
                />
              </div>
              <div className="space-y-2">
                <p className="ml-1 text-xs leading-none font-normal text-slate-900">
                  {configDraft.apiType === "responses"
                    ? "max_output_tokens"
                    : configDraft.apiType === "gemini-generate-content"
                      ? "maxOutputTokens"
                      : "max_tokens"}
                </p>
                <Input
                  className="bg-white"
                  type="text"
                  placeholder="8k / 1m / 8192"
                  value={configDraft.params.maxTokens}
                  onChange={(event) => updateDraftParams({ maxTokens: event.target.value })}
                />
              </div>
              <div className="space-y-2">
                <p className="ml-1 text-xs leading-none font-normal text-slate-900">top_k</p>
                <Input
                  className="bg-white"
                  type="number"
                  step="1"
                  min="1"
                  max="64"
                  value={configDraft.params.topK}
                  onChange={(event) => updateDraftParams({ topK: Number(event.target.value) })}
                />
              </div>
            </div>
          </div>

          <div className="model-test-divider flex items-center justify-end gap-2 border-t px-4 py-3">
            <Button type="button" onClick={applyConfig}>
              {isZh ? "应用" : "Apply"}
            </Button>
            <Button type="button" variant="outline" onClick={closeConfig}>
              {isZh ? "取消" : "Cancel"}
            </Button>
          </div>
        </aside>
      ) : null}

      <section className="glass order-1 flex h-[min(720px,calc(100vh-260px))] min-h-[520px] flex-col overflow-hidden rounded-2xl">
        <header className="model-test-divider flex items-start justify-between gap-3 border-b px-4 py-3">
          <div>
            <div className="text-sm font-semibold">
              {isZh ? "对话调试" : "Conversation"}
            </div>
            <p className="model-test-muted mt-0.5 text-xs">
              {isZh
                ? "请求经网关代理转发，自动进入请求日志"
                : "Requests go through the gateway proxy and are logged"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">
              {`${isZh ? "请求" : "Requests"}: ${requestCount}`}
            </Badge>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={clearConversation}
              disabled={history.length === 0}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {isZh ? "清空" : "Clear"}
            </Button>
          </div>
        </header>

        <div ref={messagesRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
          {history.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <p className="text-sm font-medium">
                {isZh ? "发送一条消息开始测试" : "Send a message to start testing"}
              </p>
              <p className="model-test-muted mt-1 max-w-sm text-xs">
                {isZh
                  ? "支持图片附件；失败可直接跳转到对应请求日志"
                  : "Image attachments supported; failures link to the request log"}
              </p>
            </div>
          ) : (
            history.map((message) => (
              <article
                key={message.id}
                className={
                  message.role === "user"
                    ? "flex justify-end"
                    : "flex justify-start"
                }
              >
                <div
                  className={
                    message.role === "user"
                      ? "model-test-bubble model-test-bubble-user max-w-[85%] px-3.5 py-2.5"
                      : "model-test-bubble max-w-[88%] px-3.5 py-2.5"
                  }
                >
                  {message.attachments.length > 0 ? (
                    <div className="mb-2 flex flex-wrap gap-1.5">
                      {message.attachments.map((attachment) => (
                        <img
                          key={attachment.id}
                          src={attachment.dataUrl}
                          alt={attachment.name}
                          className="model-test-thumb h-16 w-16 rounded-lg object-cover"
                        />
                      ))}
                    </div>
                  ) : null}

                  {message.thinking ? (
                    <ThinkingBlock text={message.thinking} pending={message.pending} isZh={isZh} />
                  ) : null}

                  {message.pending && !message.text && !message.thinking ? (
                    <div className="flex items-center gap-1 py-1">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400" />
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 [animation-delay:120ms]" />
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 [animation-delay:240ms]" />
                    </div>
                  ) : null}

                  {!message.pending && message.error ? (
                    <div
                      className="rounded-xl border px-3 py-2"
                      style={{
                        borderColor: "rgba(239,68,68,0.32)",
                        background: "rgba(239,68,68,0.07)",
                      }}
                    >
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-red-600">
                        <AlertTriangle className="h-3.5 w-3.5" />
                        <span>
                          {message.error.status
                            ? `${isZh ? "请求失败" : "Request failed"} · HTTP ${message.error.status}`
                            : isZh
                              ? "请求失败"
                              : "Request failed"}
                        </span>
                      </div>
                      <p className="mt-1 text-xs">{message.error.message}</p>
                      {message.error.detail ? (
                        <>
                          <button
                            type="button"
                            className="model-test-muted mt-1.5 inline-flex items-center gap-1 text-xs hover:opacity-70"
                            onClick={() =>
                              setExpandedErrorId((prev) => (prev === message.id ? null : message.id))
                            }
                          >
                            {isZh ? "错误详情" : "Details"}
                            <ChevronDown
                              className={
                                expandedErrorId === message.id
                                  ? "h-3 w-3 rotate-180 transition-transform"
                                  : "h-3 w-3 transition-transform"
                              }
                            />
                          </button>
                          {expandedErrorId === message.id ? (
                            <pre className="mt-1 max-h-48 overflow-auto rounded-lg bg-slate-900/90 p-2 text-[11px] leading-relaxed text-slate-100">
                              {message.error.detail}
                            </pre>
                          ) : null}
                        </>
                      ) : null}
                      <button
                        type="button"
                        className="mt-1.5 inline-flex items-center gap-1 text-xs font-medium underline-offset-2 hover:underline"
                        onClick={() => openLog(message)}
                      >
                        <ExternalLink className="h-3 w-3" />
                        {isZh ? "查看日志" : "View log"}
                      </button>
                    </div>
                  ) : null}

                  {!message.error && message.text ? (
                    <div className="break-words text-sm leading-relaxed">
                      <MarkdownView text={message.text} />
                    </div>
                  ) : null}

                  {message.meta ? (
                    <div className="model-test-muted mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
                      {message.meta.firstTokenMs !== undefined ? (
                        <span>{`${isZh ? "首字" : "TTFT"} ${formatDuration(message.meta.firstTokenMs)}`}</span>
                      ) : null}
                      {message.meta.latencyMs !== undefined ? (
                        <span>{`${isZh ? "耗时" : "Latency"} ${formatDuration(message.meta.latencyMs)}`}</span>
                      ) : null}
                      {message.meta.inputTokens !== undefined || message.meta.outputTokens !== undefined ? (
                        <span>
                          {`tokens ${formatTokenCount(message.meta.inputTokens)} / ${formatTokenCount(
                            message.meta.outputTokens,
                          )}`}
                        </span>
                      ) : null}
                      {message.meta.tps !== undefined ? <span>{formatTps(message.meta.tps)}</span> : null}
                      {!message.error && message.logId ? (
                        <button
                          type="button"
                          className="inline-flex items-center gap-1 font-medium underline-offset-2 hover:underline"
                          onClick={() => openLog(message)}
                        >
                          <ExternalLink className="h-3 w-3" />
                          {isZh ? "查看日志" : "View log"}
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </article>
            ))
          )}
        </div>

        {notice ? (
          <div className="model-test-notice mx-4 mb-2 px-3 py-2 text-xs">
            {notice}
          </div>
        ) : null}

        <div
          className="model-test-divider border-t px-4 py-3"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            void addFiles(event.dataTransfer.files);
          }}
        >
          {attachments.length > 0 ? (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((attachment) => (
                <div key={attachment.id} className="relative">
                  <img
                    src={attachment.dataUrl}
                    alt={attachment.name}
                    className="model-test-thumb h-16 w-16 rounded-lg object-cover"
                  />
                  <button
                    type="button"
                    aria-label={`remove ${attachment.name}`}
                    className="model-test-thumb-close absolute -right-1.5 -top-1.5 inline-flex h-5 w-5 items-center justify-center rounded-full shadow-sm hover:opacity-80"
                    onClick={() =>
                      setAttachments((prev) => prev.filter((item) => item.id !== attachment.id))
                    }
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          ) : null}

          <div className="model-test-row relative flex flex-col">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={(event) => {
                void addFiles(event.target.files);
                event.target.value = "";
              }}
            />
            <textarea
              rows={1}
              className="max-h-32 min-h-11 w-full resize-none rounded-xl bg-transparent px-2 pb-1 pt-2.5 text-sm outline-none placeholder:text-slate-400"
              placeholder={isZh ? "输入测试内容，Enter 发送，Shift+Enter 换行" : "Type a message. Enter to send, Shift+Enter for newline"}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onPaste={(event) => {
                const files = Array.from(event.clipboardData.files ?? []);
                if (files.length > 0) {
                  event.preventDefault();
                  void addFiles(files);
                }
              }}
              onKeyDown={(event) => {
                if (event.key !== "Enter" || event.shiftKey) return;
                if (event.nativeEvent.isComposing) return;
                event.preventDefault();
                void handleSend();
              }}
            />
            <div className="flex items-center justify-between pb-1.5">
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-lg"
                  title={isZh ? "添加图片" : "Attach image"}
                  aria-label={isZh ? "添加图片" : "Attach image"}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Paperclip className="h-4 w-4" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-lg"
                  title={isZh ? "请求配置" : "Request settings"}
                  aria-label={isZh ? "请求配置" : "Request settings"}
                  onClick={() => {
                    if (configOpen) {
                      closeConfig();
                    } else {
                      openConfig();
                    }
                  }}
                >
                  <Settings className="h-4 w-4" />
                </Button>
              </div>
              {sending ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-lg"
                  title={isZh ? "停止生成" : "Stop"}
                  aria-label={isZh ? "停止生成" : "Stop"}
                  onClick={handleStop}
                >
                  <Square className="h-4 w-4" />
                </Button>
              ) : (
                <Button
                  type="button"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-lg"
                  onClick={() => void handleSend()}
                  disabled={!canSend || (!draft.trim() && attachments.length === 0)}
                  title={isZh ? "发送" : "Send"}
                  aria-label={isZh ? "发送" : "Send"}
                >
                  <ArrowUp className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
