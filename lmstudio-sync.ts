import type { Plugin } from "@opencode-ai/plugin"

const LMS_OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
const LMS_LMSAPI_BASE_URL = "http://127.0.0.1:1234/api/v1"
const TIMEOUT_MS   = 6000

// LM Studio /api/v1/models response shape (native)
interface NativeLMSModel {
  key?:             string
  type?:            string
  publisher?:       string
  display_name?:    string
  architecture?:    string | null
  quantization?:    { name?: string; bits_per_weight?: number } | string | null
  params_string?:   string | null
  size_bytes?:      number
  loaded_instances?: number[]
  max_context_length?: number
  format?:          string
  capabilities?:    { vision?: boolean; trained_for_tool_use?: boolean; reasoning?: { allowed_options: string[]; default: string } | boolean }
  description?:     string
}

// LM Studio /v1/models response shape (OpenAI-compatible)
interface OpenAILMSModel {
  id:   string
  type: string
  object: string
  publisher?: string
  capabilities?: {
    vision?:              boolean
    trained_for_tool_use?: boolean
    reasoning?:           { allowed_options: string[]; default: string } | boolean
  }
  max_context_length?: number
}

interface LMSModelsResponse {
  models?: NativeLMSModel[]   // native /api/v1/models
  data?: OpenAILMSModel[]     // OpenAI-compatible /v1/models
}

// Opencode provider config shape (what the config hook receives)
interface ModelEntry {
  name?: string
  contextLength?: number
  reasoning?: boolean
  vision?: boolean
  [key: string]: unknown
}

interface ProviderConfig {
  npm?: string;
  name?: string;
  options?: { 
    isLMStudio?: boolean;
    baseURL?: string;
    apiKey?: string;
    headers?: Record<string, string>;
    [key: string]: unknown; 
  }
  models?: Record<string, ModelEntry>;
  [key: string]: unknown;
}

interface OpenCodeConfig {
  provider?: Record<string, ProviderConfig>
  [key: string]: unknown
}

// Slugify a raw model ID into a human-readable display name.
// "qwen/qwen3.6-35b-a3b-4bit" → "Qwen3.6 35B A3B 4bit"
function toDisplayName(modelId: string): string {
  const bare = modelId.includes("/") ? modelId.split("/").pop()! : modelId
  return bare
    .replace(/[-_.]/g, " ")
    .replace(/\b(\d+)b\b/gi, "$1B")
    .replace(/\b(a\d+b)\b/gi, (m) => m.toUpperCase())
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim()
}

// ── Helpers for rich metadata parsing & friendly names ───────────────────

const ARCH_FAMILIES: Array<[RegExp, string]> = [
  [/^qwen/i, "Qwen"],
  [/^(codellama|llama)/i, "Llama"],
  [/^(mixtral|ministral|mistral|devstral)/i, "Mistral"],
  [/^deepseek/i, "DeepSeek"],
  [/^gemma/i, "Gemma"],
  [/^phi/i, "Phi"],
  [/^gpt[-_]?oss/i, "GPT-OSS"],
  [/^gpt/i, "GPT"],
  [/^(falcon)/i, "Falcon"],
  [/^stablelm/i, "StableLM"],
  [/^starcoder/i, "Starcoder"],
  [/^(command|cohere)/i, "Command R"],
  [/^mpt/i, "MPT"],
  [/^rwkv/i, "RWKV"],
  [/^jais/i, "Jais"],
  [/^granite/i, "Granite"],
  [/^olmo/i, "OLMo"],
  [/^baichuan/i, "Baichuan"],
  [/^xverse/i, "XVERSE"],
  [/^yi\b/i, "Yi"],
]

function architectureFamily(arch?: string | null): string | undefined {
  if (!arch) return undefined
  for (const [re, label] of ARCH_FAMILIES) if (re.test(arch)) return label
  const head = arch.toLowerCase().split(/[^a-z0-9]+/)[0].replace(/\d+$/g, "")
  return head ? head[0].toUpperCase() + head.slice(1) : undefined
}

function normalizeQuant(raw: string): string {
  if (!raw) return ""
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, "")
}

function quantFromName(text: string): string | undefined {
  const tokens = text.split(/[\s_\-./]+/).filter(Boolean)
  for (let i = 0; i < tokens.length; i++) {
    if (!/^q\d{1,2}$/i.test(tokens[i])) continue
    let q = tokens[i].toUpperCase()
    for (let j = i + 1; j < tokens.length && j <= i + 4; j++) {
      const t = tokens[j]
      if (/^[a-z0-9]{1,2}$/i.test(t)) q += t.toUpperCase()
      else break
    }
    return q
  }
  const m = text.match(/(?:^|[/@.\-_])(q\d{1,2}(?:[_\-][a-z0-9]+){0,3})/i)
  if (m) return normalizeQuant(m[1])
  return undefined
}

function quantForModel(m: NativeLMSModel): string {
  const q = m.quantization
  if (typeof q === "string" && q.trim()) return normalizeQuant(q.trim())
  if (q && typeof q === "object" && q.name) return normalizeQuant(q.name)
  // fallback: parse from display_name or key
  const nameSrc = m.display_name || m.key || ""
  return quantFromName(nameSrc) ?? ""
}

function paramsFromModel(m: NativeLMSModel): string {
  if (m.params_string) {
    // strip trailing .0 before B/M: "4.0B" → "4B", keep "1.5B"/"7.1B"
    return m.params_string.replace(/\.0(?=[BM])\b/gi, "")
  }
  const src = (m.display_name || m.key || "").toLowerCase()
  const match = src.match(/\d+(?:\.\d+)?[bm](?:-?\s*a\d+[bm])?/)
  return match ? match[0].toUpperCase().replace(/\.0(?=[BM])/, "") : ""
}

function sizeInGB(sizeBytes: number): string {
  if (!sizeBytes) return ""
  const g = (sizeBytes / 1073741824).toFixed(1)
  return `${g}GB`
}

function isUncensored(text: string): boolean {
  return /uncens|ablit|no[- ]?censor/i.test(text)
}

// Build a friendly display name from model metadata.
// Format order (optional slots omitted when unknown):
//   [Uncensored] [ParamsNum] [Vision] [Quant] [SizeGB] [Architecture] [Maintainer] [ModelName]
function buildFriendlyName(m: NativeLMSModel): string {
  const raw = m.display_name || m.key || ""
  const tokens = raw.split(/[\s_\-]+/).filter(Boolean)

  // slots to fill (each is a tuple of [label, shouldInclude])
  const slots: Array<{ label: string; include: boolean }> = []

  // 1. Uncensored
  if (isUncensored(m.key || "")) {
    slots.push({ label: "Uncensored", include: true })
  }

  // 2. Params
  const params = paramsFromModel(m)
  if (params) {
    slots.push({ label: params, include: true })
  }

  // 3. Vision
  if (m.capabilities?.vision === true) {
    slots.push({ label: "Vision", include: true })
  }

  // 4. Quantization
  const quant = quantForModel(m)
  if (quant) {
    slots.push({ label: quant, include: true })
  }

  // 5. Size
  const size = sizeInGB(m.size_bytes ?? 0)
  if (size) {
    slots.push({ label: size, include: true })
  }

  // 6. Architecture family
  const arch = architectureFamily(m.architecture)
  if (arch) {
    slots.push({ label: arch, include: true })
  }

  // 7. Maintainer / publisher
  const pub = m.publisher || ""
  if (pub) {
    slots.push({ label: pub, include: true })
  }

  // Build removal set from filled slots so we strip matching tokens from display_name
  const normToken = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "")
  const removeTokens = new Set<string>()
  for (const { label, include } of slots) {
    if (!include) continue
    // For uncensored — strip known tokens from display name
    if (label === "Uncensored") {
      ["uncensored", "abliterated"].forEach(t => removeTokens.add(normToken(t)))
    } else if (label === "Vision") {
      removeTokens.add(normToken("vision"))
    } else if (label === params) {
      // split params on non-alnum to get pieces: "35B-A3B" → ["35b","a3b"]
      params.replace(/\.0(?=[BM])\b/gi, "").split(/[^a-z0-9]+/).forEach(t => { if (t) removeTokens.add(normToken(t)) })
    } else if (label === quant) {
      // split raw quant on non-alnum: "Q6_K_P" → ["q6","k","p"]
      const rawQuant = m.quantization && typeof m.quantization === "object" ? (m.quantization as any).name || "" : String(m.quantization || "")
      if (!rawQuant) {
        // parsed from name — we already have the tokens in slots, extract original pieces
        const qMatch = raw.toLowerCase().match(/q\d{1,2}[_\- ]?[a-z0-9 ]{0,6}/i)
        if (qMatch) qMatch[0].replace(/[^a-z0-9]+/g, " ").split(/\s+/).forEach(t => { if (/^[a-z0-9]{1,4}$/i.test(t)) removeTokens.add(normToken(t)) })
      } else {
        rawQuant.replace(/[_\-]/g, " ").split(/\s+/).forEach(t => { if (t) removeTokens.add(normToken(t)) })
      }
    } else if (label === size) {
      // strip "29.3gb" etc from display name if present
      const numPart = label.replace(/GB$/i, "")
      removeTokens.add(normToken(numPart + "gb"))
    } else if (label === arch) {
      removeTokens.add(normToken(arch.toLowerCase()))
    } else {
      // maintainer — strip publisher token from display name
      removeTokens.add(normToken(label))
    }
  }

  // Strip matching tokens from raw to get the "Model Name" slot
  const kept = tokens.filter(t => !removeTokens.has(normToken(t)))
  let modelName = kept.join(" ")

  // If model name is empty after stripping, fall back to a cleaned version of display_name or key
  if (!modelName) {
    const bare = raw.includes("/") ? raw.split("/").pop()! : raw
    modelName = bare.replace(/[-_.]/g, " ").replace(/\b(\d+)b\b/gi, "$1B").trim() || raw
  }

  // Assemble final name in order, only including slots that have content
  const parts: string[] = []
  if (slots[0]?.include && slots[0].label === "Uncensored") parts.push("Uncensored")
  if (params) parts.push(params)
  if (m.capabilities?.vision === true) parts.push("Vision")
  if (quant) parts.push(quant)
  if (size) parts.push(size)
  if (arch) parts.push(arch)
  if (pub) parts.push(pub)
  parts.push(modelName)

  return parts.join(" ")
}

export const LMStudioSyncPlugin: Plugin = async ({ client }) => {
  return {
    config: async (config: OpenCodeConfig) => {
      // ── 1. Find all providers marked with options.isLMStudio ─────────────────
      if (!config.provider) return

      let totalAdded = 0
      let totalScanned = 0

      for (const [providerKey, provider] of Object.entries(config.provider)) {
        if ((provider as ProviderConfig)?.options.isLMStudio !== true) continue

        const p = provider as ProviderConfig
        totalScanned++

        // ── a. Derive base URLs from provider config or fall back to defaults
        let rawBase: string
        if (p.options?.baseURL) {
          rawBase = p.options.baseURL
        } else {
          // check if this looks like a native API endpoint → use OpenAI default as chat base
          const apiMatch = LMS_LMSAPI_BASE_URL.match(/^(https?:\/\/[^/]+)(\/api\/v1)?$/)
          rawBase = apiMatch ? `${apiMatch[1]}/v1` : LMS_OPENAI_BASE_URL
        }

        let origin: string
        try {
          const u = new URL(rawBase)
          origin = u.origin
        } catch {
          await client.app.log({ body: { service: "lmstudio-sync", level: "warn", message: `Provider "${providerKey}" baseURL "${rawBase}" is not a valid URL — skipping` } })
          continue
        }

        const openaiBase = (() => {
          try {
            const u = new URL(rawBase)
            const path = u.pathname.replace(/\/+$/, "")
            if (path === "" || path === "/api/v1") return `${u.origin}/v1`
            if (!path.endsWith("/v1")) return rawBase.replace(/\/+$/, "")
            return rawBase.replace(/\/+$/, "")
          } catch {
            return rawBase
          }
        })()

        // ── b. Fetch native metadata for rich info (display_name, quant, architecture, etc.)
        const bearerAuth = p.options?.apiKey || process.env.LMSTUDIO_API_KEY
        
        const authHeader = bearerAuth ? { "Authorization": `Bearer ${bearerAuth}` } : {}

        let nativeModels: NativeLMSModel[] | undefined
        try {
          const metaUrl = `${origin}/api/v1/models`
          const res = await fetch(metaUrl, { headers: authHeader, signal: AbortSignal.timeout(TIMEOUT_MS) })
          if (res.ok) {
            const json = await res.json() as LMSModelsResponse
            nativeModels = json.models || []
          }
        } catch { /* will fall through to OpenAI-compatible listing */ }

        // ── c. Fetch OpenAI-compatible model list for IDs + capabilities
        let openaiModels: OpenAILMSModel[] | undefined
        try {
          const openaiUrl = `${openaiBase}/models`
          const res = await fetch(openaiUrl, { headers: authHeader, signal: AbortSignal.timeout(TIMEOUT_MS) })
          if (res.ok) {
            const json = await res.json() as LMSModelsResponse
            openaiModels = json.data || []
          }
        } catch { /* no models synced for this provider */ }

        // ── d. Merge native metadata onto OpenAI entries by matching ID ↔ key
        const nativeMap = new Map<string, NativeLMSModel>()
        if (nativeModels) {
          for (const m of nativeModels) {
            if (m.key) nativeMap.set(m.key.toLowerCase(), m)
          }
        }

        // ── e. Ensure provider config exists and set baseURL to OpenAI-compatible endpoint
        p.npm ??= "@ai-sdk/openai-compatible"
        p.name ??= `LM Studio (${providerKey})`
        p.options.baseURL = openaiBase
        p.models ??= {}

        // ── f. Merge discovered models (never overwrite manually set entries)
        let added = 0
        if (openaiModels) {
          for (const m of openaiModels) {
            const id = m.id
            if (!id || p.models[id]) continue

            // Only include non-embedding models as chat-capable
            if (m.type === "embedding") continue

            // Find matching native metadata by ID ↔ key match
            const native = nativeMap.get(id.toLowerCase())

            const entry: ModelEntry = { name: buildFriendlyName(native ?? m as unknown as NativeLMSModel) }

            if (m.max_context_length) entry.contextLength = m.max_context_length
            if (m.capabilities?.vision === true) entry.vision = true

            const r = m.capabilities?.reasoning
            if (r === true || (typeof r === "object" && r !== null)) {
              entry.reasoning = true
            }

            p.models[id] = entry
            added++
          }
        } else if (nativeModels) {
          // Only native API available — use key as ID, build names from native metadata
          for (const m of nativeModels) {
            const id = m.key || ""
            if (!id || p.models[id]) continue
            if (m.type === "embedding") continue

            const entry: ModelEntry = { name: buildFriendlyName(m) }
            if (m.max_context_length) entry.contextLength = m.max_context_length
            if (m.capabilities?.vision === true) entry.vision = true

            p.models[id] = entry
            added++
          }
        }

        totalAdded += added

        await client.app.log({ body: { service: "lmstudio-sync", level: added > 0 ? "info" : "warn", message: `Provider "${providerKey}" synced ${added} model(s) from LM Studio at ${origin}` } })
      }

      if (totalScanned === 0) {
        await client.app.log({ body: { service: "lmstudio-sync", level: "info", message: "No providers with isLMStudio=true found in config — nothing to sync" } })
      } else {
        await client.app.log({ body: { service: "lmstudio-sync", level: "info", message: `Total: synced ${totalAdded} model(s) across ${totalScanned} LM Studio provider(s)` } })
      }
    },
  }
}

export default LMStudioSyncPlugin
