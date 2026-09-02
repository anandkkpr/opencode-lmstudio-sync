// Test buildFriendlyName against real remote models
import { readFileSync } from "fs";

function architectureFamily(arch: string | null): string | undefined {
  if (!arch) return undefined;
  const ARCH_FAMILIES: Array<[RegExp, string]> = [
    [/^qwen/i, "Qwen"],
    [/^(codellama|llama)/i, "Llama"],
    [/^(mixtral|ministral|mistral|devstral)/i, "Mistral"],
    [/^deepseek/i, "DeepSeek"],
    [/^gemma/i, "Gemma"],
    [/^phi/i, "Phi"],
    [/^gpt[-_]?oss/i, "GPT-OSS"],
    [/^gpt/i, "GPT"],
  ];
  for (const [re, label] of ARCH_FAMILIES) if (re.test(arch)) return label;
  const head = arch.toLowerCase().split(/[^a-z0-9]+/)[0].replace(/\d+$/g, "");
  return head ? head[0].toUpperCase() + head.slice(1) : undefined;
}

function normalizeQuant(raw: string): string {
  if (!raw) return "";
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, "");
}

function quantFromName(text: string): string | undefined {
  const tokens = text.split(/[\s_\-./]+/).filter(Boolean);
  for (let i = 0; i < tokens.length; i++) {
    if (!/^q\d{1,2}$/i.test(tokens[i])) continue;
    let q = tokens[i].toUpperCase();
    for (let j = i + 1; j < tokens.length && j <= i + 4; j++) {
      const t = tokens[j];
      if (/^[a-z0-9]{1,2}$/i.test(t)) q += t.toUpperCase(); else break;
    }
    return q;
  }
  const m = text.match(/(?:^|[/@.\-_])(q\d{1,2}(?:[_\-][a-z0-9]+){0,3})/i);
  if (m) return normalizeQuant(m[1]);
  return undefined;
}

function quantForModel(m: { quantization: unknown }): string {
  const q = m.quantization as any;
  if (typeof q === "string" && q.trim()) return normalizeQuant(q.trim());
  if (q && typeof q === "object" && q.name) return normalizeQuant(q.name);
  return quantFromName((m as any).display_name || (m as any).key || "") ?? "";
}

function paramsFromModel(m: { display_name?: string; key?: string; params_string?: string }): string {
  if (m.params_string) return m.params_string.replace(/\.0(?=[BM])\b/gi, "");
  const src = (m.display_name || m.key || "").toLowerCase();
  const match = src.match(/\d+(?:\.\d+)?[bm](?:-?\s*a\d+[bm])?/);
  return match ? match[0].toUpperCase().replace(/\.0(?=[BM])/, "") : "";
}

function sizeInGB(b: number): string {
  if (!b) return "";
  const g = (b / 1073741824).toFixed(1);
  return `${g}GB`;
}

function isUncensored(t: string): boolean {
  return /uncens|ablit|no[- ]?censor/i.test(t);
}

interface NativeModel {
  key?: string;
  display_name?: string;
  architecture?: string | null;
  quantization?: unknown;
  params_string?: string;
  size_bytes?: number;
  capabilities?: { vision?: boolean };
  publisher?: string;
}

function buildFriendlyName(m: NativeModel): string {
  const raw = m.display_name || m.key || "";
  const tokens = raw.split(/[\s_\-]+/).filter(Boolean);
  const slots: Array<{ label: string; include: boolean }> = [];

  if (isUncensored(m.key || "")) slots.push({ label: "Uncensored", include: true });

  const params = paramsFromModel(m);
  if (params) slots.push({ label: params, include: true });

  if ((m.capabilities?.vision === true)) slots.push({ label: "Vision", include: true });

  const quant = quantForModel(m);
  if (quant) slots.push({ label: quant, include: true });

  const size = sizeInGB(m.size_bytes ?? 0);
  if (size) slots.push({ label: size, include: true });

  const arch = architectureFamily(m.architecture);
  if (arch) slots.push({ label: arch, include: true });

  const pub = m.publisher || "";
  if (pub) slots.push({ label: pub, include: true });

  const normToken = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const removeTokens = new Set<string>();

  for (const { label, include } of slots) {
    if (!include) continue;
    if (label === "Uncensored") {
      ["uncensored", "abliterated"].forEach(t => removeTokens.add(normToken(t)));
    } else if (label === "Vision") {
      removeTokens.add(normToken("vision"));
    } else if (label === params) {
      params.replace(/\.0(?=[BM])\b/gi, "").split(/[^a-z0-9]+/).forEach(t => { if (t) removeTokens.add(normToken(t)); });
    } else if (label === quant) {
      const rawQuant = m.quantization && typeof m.quantization === "object" ? (m.quantization as any).name || "" : String(m.quantization || "");
      if (rawQuant) {
        rawQuant.replace(/[_\-]/g, " ").split(/\s+/).forEach(t => { if (t) removeTokens.add(normToken(t)); });
      } else {
        const qMatch = raw.toLowerCase().match(/q\d{1,2}[_\- ]?[a-z0-9 ]{0,6}/i);
        if (qMatch) qMatch[0].replace(/[^a-z0-9]+/g, " ").split(/\s+/).forEach(t => { if (/^[a-z0-9]{1,4}$/i.test(t)) removeTokens.add(normToken(t)); });
      }
    } else if (label === size) {
      const np = label.replace(/GB$/i, "");
      removeTokens.add(normToken(np + "gb"));
    } else if (label === arch) {
      removeTokens.add(normToken(arch.toLowerCase()));
    } else {
      removeTokens.add(normToken(label));
    }
  }

  const kept = tokens.filter(t => !removeTokens.has(normToken(t)));
  let modelName = kept.join(" ");
  if (!modelName) {
    const bare = raw.includes("/") ? raw.split("/").pop()! : raw;
    modelName = bare.replace(/[-_.]/g, " ").replace(/\b(\d+)b\b/gi, "$1B").trim() || raw;
  }

  const parts: string[] = [];
  if (slots[0]?.include && slots[0].label === "Uncensored") parts.push("Uncensored");
  if (params) parts.push(params);
  if ((m.capabilities?.vision === true)) parts.push("Vision");
  if (quant) parts.push(quant);
  if (size) parts.push(size);
  if (arch) parts.push(arch);
  if (pub) parts.push(pub);
  parts.push(modelName);

  return parts.join(" ");
}

// Fetch models from remote
const BASE = "http://anand-asus-x870e-lan0.apskagharbw:51234";
const apiKey = process.env.LMSTUDIO_API_KEY;

console.log(`Fetching models from ${BASE}/api/v1/models ...`);

try {
  const res = await fetch(`${BASE}/api/v1/models`, { headers: { Authorization: `Bearer ${apiKey}` } });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const json = await res.json();
  const models: NativeModel[] = json.models || [];

  console.log(`\nFound ${models.length} models.\n`);

  for (const m of models.sort((a, b) => ((a.display_name||"").localeCompare(b.display_name||"")))) {
    const friendly = buildFriendlyName(m);
    console.log(`KEY:      ${m.key}`);
    console.log(`DISPLAY:  ${m.display_name || "(none)"}`);
    console.log(`ARCH:     ${m.architecture || "(unknown)"}`);
    console.log(`QUANT:    ${JSON.stringify(m.quantization)}`);
    console.log(`PARAMS:   ${m.params_string || "(none)"}`);
    const sz = m.size_bytes ? `${(m.size_bytes / 1073741824).toFixed(1)}GB` : "(unknown)";
    console.log(`SIZE:     ${sz}`);
    console.log(`VISION:   ${JSON.stringify(m.capabilities?.vision)}`);
    console.log(`PUB:      ${m.publisher || "(none)"}`);
    console.log(`→ FRIENDLY: ${friendly}\n`);
  }
} catch (e: any) {
  console.error("Failed:", e.message);

  // Fallback: try OpenAI-compatible endpoint too
  try {
    const res2 = await fetch(`${BASE}/v1/models`, { headers: { Authorization: `Bearer ${apiKey}` } });
    if (!res2.ok) throw new Error(`HTTP ${res2.status}`);
    const json2 = await res2.json();
    const models2 = (json2 as any).data || [];
    console.log(`\nOpenAI-compatible /v1/models returned ${models2.length} entries.\n`);
    for (const m of models2) {
      console.log(`ID:       ${m.id}`);
      console.log(`TYPE:     ${m.type}`);
      console.log(`CAPAB:    ${JSON.stringify(m.capabilities)}\n`);
    }
  } catch (e2: any) {
    console.error("Also failed /v1/models:", e2.message);
  }
}
