import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"
import {
  type ChatCompletionChunk,
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
  type Message,
  type ToolCall,
} from "~/services/copilot/create-chat-completions"

// ---------------------------------------------------------------------------
// Responses API request types
// ---------------------------------------------------------------------------

type ResponsesInputSystemMessage = {
  role: "system" | "developer"
  content: string
}

type ResponsesInputUserMessage = {
  role: "user"
  content: Array<ResponsesUserContentPart>
}

type ResponsesInputAssistantMessage = {
  role: "assistant"
  content: Array<{ type: "output_text"; text: string }>
  id?: string
}

type ResponsesUserContentPart =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string }

type ResponsesFunctionCall = {
  type: "function_call"
  call_id: string
  name: string
  arguments: string
}

type ResponsesFunctionCallOutput = {
  type: "function_call_output"
  call_id: string
  output: string
}

type ResponsesInputItem =
  | ResponsesInputSystemMessage
  | ResponsesInputUserMessage
  | ResponsesInputAssistantMessage
  | ResponsesFunctionCall
  | ResponsesFunctionCallOutput

type ResponsesTool = {
  type: "function"
  name: string
  description?: string
  parameters: Record<string, unknown>
  strict?: boolean
}

type ResponsesToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; name: string }

interface ResponsesRequestBody {
  model: string
  input: Array<ResponsesInputItem>
  temperature?: number
  top_p?: number
  max_output_tokens?: number
  tools?: Array<ResponsesTool>
  tool_choice?: ResponsesToolChoice
  stream?: boolean
}

// ---------------------------------------------------------------------------
// Responses API response types
// ---------------------------------------------------------------------------

interface ResponsesOutputTextContent {
  type: "output_text"
  text: string
}

interface ResponsesMessageOutputItem {
  type: "message"
  id: string
  role: "assistant"
  content: Array<ResponsesOutputTextContent>
}

interface ResponsesFunctionCallOutputItem {
  type: "function_call"
  id: string
  call_id: string
  name: string
  arguments: string
}

type ResponsesOutputItem =
  | ResponsesMessageOutputItem
  | ResponsesFunctionCallOutputItem
  | { type: string }

interface ResponsesUsage {
  input_tokens: number
  output_tokens: number
  input_tokens_details?: { cached_tokens?: number }
  output_tokens_details?: { reasoning_tokens?: number }
}

interface ResponsesResponse {
  id: string
  model: string
  created_at: number
  output: Array<ResponsesOutputItem>
  incomplete_details?: { reason: string } | null
  usage: ResponsesUsage
}

// ---------------------------------------------------------------------------
// SSE chunk types for the Responses streaming API
// ---------------------------------------------------------------------------

interface ResponseCreatedChunk {
  type: "response.created"
  response: { id: string; model: string; created_at: number }
}

interface OutputItemAddedChunk {
  type: "response.output_item.added"
  output_index: number
  item: {
    type: string
    id?: string
    call_id?: string
    name?: string
    encrypted_content?: string | null
  }
}

interface OutputItemDoneChunk {
  type: "response.output_item.done"
  output_index: number
  item: {
    type: string
    id?: string
    call_id?: string
    name?: string
    arguments?: string
  }
}

interface TextDeltaChunk {
  type: "response.output_text.delta"
  item_id: string
  delta: string
}

interface FunctionCallDeltaChunk {
  type: "response.function_call_arguments.delta"
  output_index: number
  delta: string
}

interface ResponseCompletedChunk {
  type: "response.completed" | "response.incomplete"
  response: {
    incomplete_details?: { reason: string } | null
    usage: ResponsesUsage
    service_tier?: string | null
  }
}

interface ErrorChunk {
  type: "error"
  code: string
  message: string
}

type ResponsesSSEChunk =
  | ResponseCreatedChunk
  | OutputItemAddedChunk
  | OutputItemDoneChunk
  | TextDeltaChunk
  | FunctionCallDeltaChunk
  | ResponseCompletedChunk
  | ErrorChunk
  | { type: string }

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

export function isResponsesOnlyModel(model: string): boolean {
  const match = /^gpt-(\d+)/.exec(model)
  if (!match) return false
  const n = Number(match[1])
  return n >= 5 && !model.startsWith("gpt-5-mini")
}

// ---------------------------------------------------------------------------
// Reasoning model detection (mirrors opencode logic)
// ---------------------------------------------------------------------------

function isReasoningModel(model: string): boolean {
  if (model.startsWith("gpt-5-chat")) return false
  if (
    model.startsWith("o")
    || model.startsWith("gpt-5")
    || model.startsWith("codex-")
    || model.startsWith("computer-use")
  ) {
    return true
  }
  return false
}

function systemMessageMode(model: string): "system" | "developer" | "remove" {
  if (model.startsWith("o1-mini") || model.startsWith("o1-preview")) {
    return "remove"
  }
  if (isReasoningModel(model)) {
    return "developer"
  }
  return "system"
}

// ---------------------------------------------------------------------------
// finish_reason mapping
// ---------------------------------------------------------------------------

function mapResponsesFinishReason(
  incompleteReason: string | null | undefined,
  hasFunctionCall: boolean,
): ChatCompletionChunk["choices"][0]["finish_reason"] {
  switch (incompleteReason) {
    case undefined:
    case null: {
      return hasFunctionCall ? "tool_calls" : "stop"
    }
    case "max_output_tokens": {
      return "length"
    }
    case "content_filter": {
      return "content_filter"
    }
    default: {
      return hasFunctionCall ? "tool_calls" : "stop"
    }
  }
}

// ---------------------------------------------------------------------------
// Content helpers (avoid nested ternaries)
// ---------------------------------------------------------------------------

function extractTextFromContent(content: Message["content"]): string {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return ""
  return content
    .flatMap((p) => (p.type === "text" ? [p.text] : []))
    .join("\n\n")
}

function buildUserContentParts(
  content: Message["content"],
): Array<ResponsesUserContentPart> {
  const parts: Array<ResponsesUserContentPart> = []
  if (typeof content === "string") {
    parts.push({ type: "input_text", text: content })
    return parts
  }
  if (!Array.isArray(content)) return parts
  for (const part of content) {
    if (part.type === "text") {
      parts.push({ type: "input_text", text: part.text })
    } else {
      // image_url
      parts.push({ type: "input_image", image_url: part.image_url.url })
    }
  }
  return parts
}

// ---------------------------------------------------------------------------
// Per-role message converters (keep complexity per function low)
// ---------------------------------------------------------------------------

function convertSystemOrDeveloperMessage(
  msg: Message,
  mode: "system" | "developer" | "remove",
  input: Array<ResponsesInputItem>,
): void {
  if (mode === "remove") return
  const text = extractTextFromContent(msg.content)
  input.push({ role: mode, content: text })
}

function convertUserMessage(
  msg: Message,
  input: Array<ResponsesInputItem>,
): void {
  input.push({ role: "user", content: buildUserContentParts(msg.content) })
}

function convertAssistantMessage(
  msg: Message,
  input: Array<ResponsesInputItem>,
): void {
  const text = extractTextFromContent(msg.content)
  if (text) {
    input.push({
      role: "assistant",
      content: [{ type: "output_text", text }],
    })
  }
  if (!msg.tool_calls) return
  for (const tc of msg.tool_calls) {
    input.push({
      type: "function_call",
      call_id: tc.id,
      name: tc.function.name,
      arguments: tc.function.arguments,
    })
  }
}

function convertToolMessage(
  msg: Message,
  input: Array<ResponsesInputItem>,
): void {
  const output = extractTextFromContent(msg.content)
  input.push({
    type: "function_call_output",
    call_id: msg.tool_call_id ?? "",
    output,
  })
}

// ---------------------------------------------------------------------------
// Chat-completions payload → Responses API request body
// ---------------------------------------------------------------------------

function buildResponsesInput(
  payload: ChatCompletionsPayload,
): Array<ResponsesInputItem> {
  const input: Array<ResponsesInputItem> = []
  const mode = systemMessageMode(payload.model)

  for (const msg of payload.messages) {
    switch (msg.role) {
      case "system":
      case "developer": {
        convertSystemOrDeveloperMessage(msg, mode, input)
        break
      }
      case "user": {
        convertUserMessage(msg, input)
        break
      }
      case "assistant": {
        convertAssistantMessage(msg, input)
        break
      }
      case "tool": {
        convertToolMessage(msg, input)
        break
      }
      default: {
        break
      }
    }
  }

  return input
}

function buildResponsesTools(
  payload: ChatCompletionsPayload,
): Array<ResponsesTool> | undefined {
  if (!payload.tools || payload.tools.length === 0) return undefined
  return payload.tools.map((t) => ({
    type: "function",
    name: t.function.name,
    description: t.function.description,
    parameters: t.function.parameters,
    strict: false,
  }))
}

function buildResponsesToolChoice(
  payload: ChatCompletionsPayload,
): ResponsesToolChoice | undefined {
  const tc = payload.tool_choice
  if (!tc) return undefined
  if (tc === "auto" || tc === "none" || tc === "required") return tc
  // object: { type: "function"; function: { name: string } }
  return { type: "function", name: tc.function.name }
}

function buildMaxOutputTokens(
  payload: ChatCompletionsPayload,
): number | undefined {
  const v = payload.max_completion_tokens ?? payload.max_tokens
  if (v === null || v === undefined) return undefined
  return v
}

function buildRequestBody(
  payload: ChatCompletionsPayload,
): ResponsesRequestBody {
  const body: ResponsesRequestBody = {
    model: payload.model,
    input: buildResponsesInput(payload),
  }

  const maxOutputTokens = buildMaxOutputTokens(payload)
  if (maxOutputTokens !== undefined) body.max_output_tokens = maxOutputTokens

  if (!isReasoningModel(payload.model)) {
    if (payload.temperature !== null && payload.temperature !== undefined) {
      body.temperature = payload.temperature
    }
    if (payload.top_p !== null && payload.top_p !== undefined) {
      body.top_p = payload.top_p
    }
  }

  const tools = buildResponsesTools(payload)
  if (tools) body.tools = tools

  const toolChoice = buildResponsesToolChoice(payload)
  if (toolChoice !== undefined) body.tool_choice = toolChoice

  if (payload.stream) body.stream = true

  return body
}

// ---------------------------------------------------------------------------
// Non-streaming: parse Responses JSON → ChatCompletionResponse
// ---------------------------------------------------------------------------

function parseResponsesResponse(
  resp: ResponsesResponse,
): ChatCompletionResponse {
  const toolCalls: Array<ToolCall> = []
  let text = ""
  let hasFunctionCall = false

  for (const item of resp.output) {
    if (item.type === "message") {
      const msgItem = item as ResponsesMessageOutputItem
      for (const part of msgItem.content) {
        text += part.text
      }
    } else if (item.type === "function_call") {
      hasFunctionCall = true
      const fnItem = item as ResponsesFunctionCallOutputItem
      toolCalls.push({
        id: fnItem.call_id,
        type: "function",
        function: {
          name: fnItem.name,
          arguments: fnItem.arguments,
        },
      })
    }
  }

  const finishReason: ChatCompletionResponse["choices"][0]["finish_reason"] =
    mapResponsesFinishReason(resp.incomplete_details?.reason, hasFunctionCall)
    ?? "stop"

  return {
    id: resp.id,
    object: "chat.completion",
    created: Math.floor(resp.created_at),
    model: resp.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: text || null,
          ...(toolCalls.length > 0 && { tool_calls: toolCalls }),
        },
        logprobs: null,
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: resp.usage.input_tokens,
      completion_tokens: resp.usage.output_tokens,
      total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
      ...(resp.usage.input_tokens_details?.cached_tokens !== undefined && {
        prompt_tokens_details: {
          cached_tokens: resp.usage.input_tokens_details.cached_tokens,
        },
      }),
    },
  }
}

// ---------------------------------------------------------------------------
// Streaming helpers — split out to keep per-function complexity low
// ---------------------------------------------------------------------------

interface ToolCallSlot {
  toolName: string
  callId: string
  argsAccumulator: string
}

interface StreamState {
  responseId: string
  model: string
  createdAt: number
  ongoingToolCalls: Map<number, ToolCallSlot>
  currentTextId: string | null
  hasFunctionCall: boolean
}

function makeBaseChunk(s: StreamState): Omit<ChatCompletionChunk, "choices"> {
  return {
    id: s.responseId,
    object: "chat.completion.chunk",
    created: s.createdAt,
    model: s.model,
  }
}

function makeChunk(
  s: StreamState,
  delta: ChatCompletionChunk["choices"][0]["delta"],
  finishReason: ChatCompletionChunk["choices"][0]["finish_reason"] = null,
): ChatCompletionChunk {
  return {
    ...makeBaseChunk(s),
    choices: [{ index: 0, delta, finish_reason: finishReason, logprobs: null }],
  }
}

function handleOutputItemAdded(
  chunk: OutputItemAddedChunk,
  s: StreamState,
): ChatCompletionChunk | undefined {
  const item = chunk.item
  if (item.type === "message" && item.id) {
    s.currentTextId = item.id
    return undefined
  }
  if (item.type === "function_call" && item.call_id && item.name) {
    s.ongoingToolCalls.set(chunk.output_index, {
      toolName: item.name,
      callId: item.call_id,
      argsAccumulator: "",
    })
    return makeChunk(s, {
      tool_calls: [
        {
          index: chunk.output_index,
          id: item.call_id,
          type: "function",
          function: { name: item.name, arguments: "" },
        },
      ],
    })
  }
  return undefined
}

function handleTextDelta(
  chunk: TextDeltaChunk,
  s: StreamState,
): ChatCompletionChunk {
  if (!s.currentTextId) s.currentTextId = chunk.item_id
  return makeChunk(s, { content: chunk.delta })
}

function handleFunctionCallDelta(
  chunk: FunctionCallDeltaChunk,
  s: StreamState,
): ChatCompletionChunk | undefined {
  const slot = s.ongoingToolCalls.get(chunk.output_index)
  if (!slot) return undefined
  slot.argsAccumulator += chunk.delta
  return makeChunk(s, {
    tool_calls: [
      { index: chunk.output_index, function: { arguments: chunk.delta } },
    ],
  })
}

function handleOutputItemDone(
  chunk: OutputItemDoneChunk,
  s: StreamState,
): void {
  if (chunk.item.type === "function_call") {
    s.hasFunctionCall = true
    s.ongoingToolCalls.delete(chunk.output_index)
  } else if (chunk.item.type === "message") {
    s.currentTextId = null
  }
}

function handleCompleted(
  chunk: ResponseCompletedChunk,
  s: StreamState,
): ChatCompletionChunk {
  const incompleteReason = chunk.response.incomplete_details?.reason ?? null
  const finishReason = mapResponsesFinishReason(
    incompleteReason,
    s.hasFunctionCall,
  )
  const usage = chunk.response.usage
  return {
    ...makeBaseChunk(s),
    choices: [
      { index: 0, delta: {}, finish_reason: finishReason, logprobs: null },
    ],
    usage: {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.input_tokens + usage.output_tokens,
      ...(usage.input_tokens_details?.cached_tokens !== undefined && {
        prompt_tokens_details: {
          cached_tokens: usage.input_tokens_details.cached_tokens,
        },
      }),
    },
  }
}

// ---------------------------------------------------------------------------
// Streaming: convert Responses SSE events → AsyncIterable<{data:string}>
// ---------------------------------------------------------------------------

async function* streamResponsesAsCompletions(
  sseSource: AsyncIterable<{ data?: string }>,
  model: string,
  responseId: string,
): AsyncIterable<{ data: string }> {
  const s: StreamState = {
    responseId,
    model,
    createdAt: Math.floor(Date.now() / 1000),
    ongoingToolCalls: new Map(),
    currentTextId: null,
    hasFunctionCall: false,
  }

  yield { data: JSON.stringify(makeChunk(s, { role: "assistant" })) }

  for await (const raw of sseSource) {
    if (!raw.data || raw.data === "[DONE]") continue

    let chunk: ResponsesSSEChunk
    try {
      chunk = JSON.parse(raw.data) as ResponsesSSEChunk
    } catch {
      continue
    }

    const out = dispatchSSEChunk(chunk, s)
    if (out) yield { data: JSON.stringify(out) }
  }

  yield { data: "[DONE]" }
}

function dispatchSSEChunk(
  chunk: ResponsesSSEChunk,
  s: StreamState,
): ChatCompletionChunk | undefined {
  switch (chunk.type) {
    case "response.created": {
      return undefined
    }
    case "response.output_item.added": {
      return handleOutputItemAdded(chunk as OutputItemAddedChunk, s)
    }
    case "response.output_text.delta": {
      return handleTextDelta(chunk as TextDeltaChunk, s)
    }
    case "response.function_call_arguments.delta": {
      return handleFunctionCallDelta(chunk as FunctionCallDeltaChunk, s)
    }
    case "response.output_item.done": {
      handleOutputItemDone(chunk as OutputItemDoneChunk, s)
      return undefined
    }
    case "response.completed":
    case "response.incomplete": {
      return handleCompleted(chunk as ResponseCompletedChunk, s)
    }
    case "error": {
      consola.error("Responses API error event:", chunk)
      return undefined
    }
    default: {
      return undefined
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const createResponses = async (
  payload: ChatCompletionsPayload,
): Promise<ChatCompletionResponse | AsyncIterable<{ data: string }>> => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const enableVision = payload.messages.some(
    (x) =>
      typeof x.content !== "string"
      && x.content?.some((p) => p.type === "image_url"),
  )

  const isAgentCall = payload.messages.some((msg) =>
    ["assistant", "tool"].includes(msg.role),
  )

  const headers: Record<string, string> = {
    ...copilotHeaders(state, enableVision),
    "X-Initiator": isAgentCall ? "agent" : "user",
  }

  const body = buildRequestBody(payload)
  consola.debug("Responses API request body:", JSON.stringify(body).slice(-400))

  const response = await fetch(`${copilotBaseUrl(state)}/responses`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    consola.error("Failed to create responses", response)
    throw new HTTPError("Failed to create responses", response)
  }

  if (payload.stream) {
    const responseId =
      response.headers.get("x-request-id")
      ?? response.headers.get("openai-request-id")
      ?? `resp-${Date.now()}`

    const sseIterable = events(response)
    return streamResponsesAsCompletions(sseIterable, payload.model, responseId)
  }

  const json = (await response.json()) as ResponsesResponse
  return parseResponsesResponse(json)
}
