import { state } from "~/lib/state"
import {
  createChatCompletions,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
} from "~/services/copilot/create-chat-completions"
import {
  createResponses,
  isResponsesOnlyModel,
} from "~/services/copilot/create-responses"

interface ExtendedCapabilities {
  supported_endpoints?: Array<string>
}

function shouldUseResponsesApi(model: string): boolean {
  // Consult model metadata first: if supported_endpoints is present and does
  // NOT include "/chat/completions" (or only lists "/responses"), force responses.
  const modelMeta = state.models?.data.find((m) => m.id === model)
  if (modelMeta) {
    const caps = modelMeta.capabilities as typeof modelMeta.capabilities
      & ExtendedCapabilities
    const endpoints: Array<string> | undefined = caps.supported_endpoints

    if (endpoints && endpoints.length > 0) {
      const supportsChatCompletions = endpoints.includes("/chat/completions")
      if (!supportsChatCompletions) return true
      // If it explicitly lists chat/completions, honour that (regex fallback
      // is skipped in this case).
      return false
    }
  }

  // Fallback: regex heuristic
  return isResponsesOnlyModel(model)
}

export const createCompletion = (
  payload: ChatCompletionsPayload,
): Promise<ChatCompletionResponse | AsyncIterable<{ data: string }>> => {
  if (shouldUseResponsesApi(payload.model)) {
    return createResponses(payload)
  }
  return createChatCompletions(payload) as Promise<
    ChatCompletionResponse | AsyncIterable<{ data: string }>
  >
}
