# opencode-lmstudio-sync

An opencode plugin that syncs your LM Studio model list at startup and injects them as chat-capable providers with rich metadata (quantization, params, architecture family, vision support).

## Setup

No build step required — opencode loads TypeScript plugins directly.

### 1. Install the plugin file into opencode's global plugin directory

```bash
mkdir -p ~/.config/opencode/plugins

# Copy:
cp lmstudio-sync.ts ~/.config/opencode/plugins/lmstudio-sync.ts

# Or symlink:
ln -sf "$(pwd)/lmstudio-sync.ts" ~/.config/opencode/plugins/lmstudio-sync.ts
```

### 2. Register the plugin in `~/.config/opencode/opencode.jsonc`

Add `"./plugins/lmstudio-sync.ts"` to your `plugin` array:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "./plugins/lmstudio-sync.ts"
  ],
  ...
}
```

### 3. Mark an LM Studio provider with `isLMStudio: true`

Add the `isLMStudio: true` flag to any provider whose models you want synced. The plugin will query both the native `/api/v1/models` endpoint and the OpenAI-compatible `/v1/models` endpoint, merge the metadata, and populate that provider's model list automatically.

```jsonc
{
  "provider": {
    "lmstudio-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LM Studio Local",
      "options": {
        // THIS!!! ↓
        "isLMStudio": true,
        "baseURL": "http://127.0.0.1:1234/v1",
        "apiKey": "{env:LMSTUDIO_API_KEY}"
      }
    },
    "lmstudio-remote": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LM Studio Remote",
      "options": {
        // THIS!!! ↓
        "isLMStudio": true,
        "baseURL": "https://my-remote-lmstudio-server.local:1234/v1",
        "apiKey": "{env:LMSTUDIO_REMOTE_API_KEY}"
      }
    }
  }
}
```

**Key points:**

- Only providers with `options.isLMStudio: true` are scanned — other providers are left untouched, even if they share the same base URL.
- Manually defined models (entries already in `provider.models`) are never overwritten by auto-discovery.
- Embedding-only models are excluded from the chat-capable model list.

### 4. Restart opencode

The plugin runs during config initialization. Restart your opencode session for changes to take effect.

## How it works

On startup, each `isLMStudio` provider triggers:

1. **Native API fetch** — queries `/api/v1/models` for metadata (`display_name`, `quantization`, `architecture`, `params_string`, `size_bytes`, `capabilities`).
2. **OpenAI-compatible fetch** — queries `/v1/models` for the model IDs and capabilities.
3. **Merge & build friendly names** — combines both sources, generates human-readable display names in the format:

   ```
   [AbliterationType] [ParamsNum] [VisionCapable] [Quant] [SizeGB] [Architecture] [Publisher] [ModelName]
   ```

4. **Inject into provider config** — discovered models are added to `provider.models` (skipping any already manually defined).

## Requirements

- LM Studio API server must be running and reachable at the configured base URL.
- If your LM Studio instance uses authentication, set `options.apiKey` or export environment variable `LMSTUDIO_API_KEY` (which the script looks for by default).
