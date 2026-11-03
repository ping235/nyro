import type { ApiKey, Model } from "@/lib/types";

/**
 * 模型在指定 API Key 下是否可见。
 *
 * 与后端 `GET /v1/models`（crates/nyro-core/src/proxy/handler.rs::models_list）保持一致：
 * - 未开启鉴权（`enable_auth = false`）的模型对所有调用方可见；
 * - 开启鉴权的模型必须被该 Key 显式绑定（`api_keys.model_ids`）；
 * - 未选择 Key 时，仅未开启鉴权的模型可用。
 */
export function isModelAccessibleWithKey(model: Model, key: ApiKey | null | undefined): boolean {
  if (!model.enable_auth) return true;
  if (!key) return false;
  return key.model_ids.includes(model.id);
}

/** 按所选 API Key 过滤模型列表，用于模型选择下拉。 */
export function filterModelsByApiKey(models: Model[], key: ApiKey | null | undefined): Model[] {
  return models.filter((model) => isModelAccessibleWithKey(model, key));
}

/** 反向过滤：给定模型路由，挑出绑定了该路由的 API Key。 */
export function filterApiKeysByModel(
  apiKeys: ApiKey[],
  model: Model | null | undefined,
): ApiKey[] {
  if (!model) return [];
  return apiKeys.filter((key) => key.model_ids.includes(model.id));
}
