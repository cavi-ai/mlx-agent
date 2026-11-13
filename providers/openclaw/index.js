import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import {
  buildOpenAICompatibleLiveModels,
  buildOpenAICompatibleProviderCatalog,
} from "openclaw/plugin-sdk/provider-catalog-live-runtime";
import { buildProviderReplayFamilyHooks } from "openclaw/plugin-sdk/provider-model-shared";

const PROVIDER_ID = "mlx-agent";
const LABEL = "MLX Agent";
const HINT = "Authenticated local MLX models";
const DEFAULT_BASE_URL = "http://127.0.0.1:8090/v1";
const API_KEY_ENV_VAR = "MLX_AGENT_API_KEY";

const SETUP = {
  providerId: PROVIDER_ID,
  providerLabel: LABEL,
  defaultBaseUrl: DEFAULT_BASE_URL,
  defaultApiKeyEnvVar: API_KEY_ENV_VAR,
  modelPlaceholder: "model directory or Hub repository name",
};

const loadSetup = () => import("openclaw/plugin-sdk/provider-setup");

// Typed endpoints list their task_kinds and serve embeddings and tasks, not chat.
const isChatRoute = (row) => !Array.isArray(row?.task_kinds);

export default definePluginEntry({
  id: PROVIDER_ID,
  name: "MLX Agent Provider",
  description: "Chat models served by an mlx-agent front door",
  register(api) {
    api.registerProvider({
      ...buildProviderReplayFamilyHooks({
        family: "openai-compatible",
        dropReasoningFromHistory: false,
      }),
      id: PROVIDER_ID,
      label: LABEL,
      envVars: [API_KEY_ENV_VAR],
      auth: [
        {
          id: "custom",
          label: LABEL,
          hint: HINT,
          kind: "custom",
          run: async (ctx) =>
            (await loadSetup()).promptAndConfigureOpenAICompatibleSelfHostedProviderAuth({
              cfg: ctx.config,
              prompter: ctx.prompter,
              ...SETUP,
            }),
          runNonInteractive: async (ctx) =>
            (await loadSetup()).configureOpenAICompatibleSelfHostedProviderNonInteractive({
              ctx,
              ...SETUP,
            }),
        },
      ],
      catalog: {
        order: "late",
        run: (ctx) =>
          buildOpenAICompatibleProviderCatalog({
            ctx,
            providerId: PROVIDER_ID,
            buildProvider: () => ({
              baseUrl: DEFAULT_BASE_URL,
              api: "openai-completions",
              models: [],
            }),
            allowExplicitBaseUrl: true,
            discoveryMode: "strict",
            modelDiscovery: {
              projectRows: (rows, fallback) =>
                buildOpenAICompatibleLiveModels(rows.filter(isChatRoute), fallback),
            },
          }),
      },
      wizard: {
        setup: {
          choiceId: PROVIDER_ID,
          choiceLabel: LABEL,
          choiceHint: HINT,
          groupId: PROVIDER_ID,
          groupLabel: LABEL,
          groupHint: HINT,
          methodId: "custom",
        },
        modelPicker: {
          label: `${LABEL} (custom)`,
          hint: `Enter ${LABEL} URL + API key + model`,
          methodId: "custom",
        },
      },
    });
  },
});
