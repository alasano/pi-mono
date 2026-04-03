/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	StreamFn,
} from "./types.js";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

type AssistantAbortState = {
	signal: AbortSignal;
	hardAborted: boolean;
	interruptAborted: boolean;
	cleanup: () => void;
};

type QueueKind = "steering" | "follow-up";

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn);
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

function isRunInterrupted(config: AgentLoopConfig): boolean {
	return config.isInterrupted?.() ?? false;
}

function getQueueCallbacks(config: AgentLoopConfig, kind: QueueKind) {
	if (kind === "steering") {
		return {
			getMessages: config.getSteeringMessages,
			requeueMessages: config.requeueSteeringMessages,
		};
	}
	return {
		getMessages: config.getFollowUpMessages,
		requeueMessages: config.requeueFollowUpMessages,
	};
}

function getInterruptedToolOutcome(config: AgentLoopConfig): ImmediateToolCallOutcome | undefined {
	if (!isRunInterrupted(config)) {
		return undefined;
	}
	return {
		kind: "immediate",
		result: createErrorToolResult("Tool execution was blocked: session interrupted"),
		isError: true,
	};
}

async function pollQueuedMessages(config: AgentLoopConfig, kind: QueueKind): Promise<AgentMessage[]> {
	const { getMessages } = getQueueCallbacks(config, kind);
	if (!getMessages) {
		return [];
	}
	return (await getMessages()) || [];
}

function preserveQueuedMessages(config: AgentLoopConfig, kind: QueueKind, messages: AgentMessage[]): void {
	if (messages.length === 0) {
		return;
	}
	const { requeueMessages } = getQueueCallbacks(config, kind);
	requeueMessages?.(messages);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<void> {
	if (config.interruptSignal !== undefined && config.isInterrupted === undefined) {
		throw new Error(
			"interruptSignal requires isInterrupted to be provided. " +
				"Without isInterrupted, the assistant stream will be interrupted but the loop will not exit after turn_end.",
		);
	}

	let firstTurn = true;
	let pendingMessages: AgentMessage[] = [];
	let pendingMessagesKind: QueueKind | undefined;

	const consumePendingMessages = async () => {
		for (const message of pendingMessages) {
			await emit({ type: "message_start", message });
			await emit({ type: "message_end", message });
			currentContext.messages.push(message);
			newMessages.push(message);
		}
		pendingMessages = [];
		pendingMessagesKind = undefined;
	};

	const requeuePendingMessages = () => {
		if (pendingMessages.length > 0 && pendingMessagesKind !== undefined) {
			preserveQueuedMessages(config, pendingMessagesKind, pendingMessages);
			pendingMessages = [];
			pendingMessagesKind = undefined;
		}
	};

	if (!config.skipInitialSteeringPoll) {
		pendingMessages = await pollQueuedMessages(config, "steering");
		pendingMessagesKind = pendingMessages.length > 0 ? "steering" : undefined;
	}

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			await consumePendingMessages();

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
			newMessages.push(message);

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");

			const toolResults: ToolResultMessage[] = [];
			hasMoreToolCalls = false;
			if (toolCalls.length > 0) {
				const executedToolBatch = await executeToolCalls(currentContext, message, config, signal, emit);
				toolResults.push(...executedToolBatch.messages);
				hasMoreToolCalls = !executedToolBatch.terminate;

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			await emit({ type: "turn_end", message, toolResults });

			if (isRunInterrupted(config)) {
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			pendingMessages = await pollQueuedMessages(config, "steering");
			pendingMessagesKind = pendingMessages.length > 0 ? "steering" : undefined;
			if (pendingMessages.length > 0 && isRunInterrupted(config)) {
				requeuePendingMessages();
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = await pollQueuedMessages(config, "follow-up");
		pendingMessages = followUpMessages;
		pendingMessagesKind = followUpMessages.length > 0 ? "follow-up" : undefined;
		if (followUpMessages.length > 0 && isRunInterrupted(config)) {
			requeuePendingMessages();
			await emit({ type: "agent_end", messages: newMessages });
			return;
		}
		if (followUpMessages.length > 0) {
			continue;
		}

		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

function createAssistantAbortState(
	hardSignal: AbortSignal | undefined,
	interruptSignal: AbortSignal | undefined,
): AssistantAbortState {
	const controller = new AbortController();
	const state: AssistantAbortState = {
		signal: controller.signal,
		hardAborted: false,
		interruptAborted: false,
		cleanup: () => {
			if (hardSignal) {
				hardSignal.removeEventListener("abort", onHardAbort);
			}
			if (interruptSignal) {
				interruptSignal.removeEventListener("abort", onInterruptAbort);
			}
		},
	};

	const abortCombined = () => {
		if (!controller.signal.aborted) {
			controller.abort();
		}
	};

	const onHardAbort = () => {
		state.hardAborted = true;
		abortCombined();
	};

	const onInterruptAbort = () => {
		state.interruptAborted = true;
		abortCombined();
	};

	if (hardSignal?.aborted) {
		onHardAbort();
	} else if (hardSignal) {
		hardSignal.addEventListener("abort", onHardAbort, { once: true });
	}

	if (interruptSignal?.aborted) {
		onInterruptAbort();
	} else if (interruptSignal) {
		interruptSignal.addEventListener("abort", onInterruptAbort, { once: true });
	}

	return state;
}

function normalizeAssistantStopReason(message: AssistantMessage, abortState: AssistantAbortState): AssistantMessage {
	if (message.stopReason !== "aborted") {
		return message;
	}
	if (abortState.hardAborted || !abortState.interruptAborted) {
		return message;
	}
	return {
		...message,
		stopReason: "interrupted",
	};
}

function createInterruptedAssistantMessage(
	config: AgentLoopConfig,
	stopReason: "aborted" | "interrupted",
): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: config.model.api,
		provider: config.model.provider,
		model: config.model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				total: 0,
			},
		},
		stopReason,
		timestamp: Date.now(),
	};
}

async function emitInterruptedAssistantMessage(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	abortState: AssistantAbortState,
): Promise<AssistantMessage> {
	const message = createInterruptedAssistantMessage(config, abortState.hardAborted ? "aborted" : "interrupted");
	context.messages.push(message);
	await emit({ type: "message_start", message: { ...message } });
	await emit({ type: "message_end", message });
	return message;
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<AssistantMessage> {
	const assistantAbortState = createAssistantAbortState(signal, config.interruptSignal);

	try {
		let messages = context.messages;
		if (config.transformContext) {
			messages = await config.transformContext(messages, assistantAbortState.signal);
		}

		if (assistantAbortState.signal.aborted) {
			return emitInterruptedAssistantMessage(context, config, emit, assistantAbortState);
		}

		const llmMessages = await config.convertToLlm(messages);
		if (assistantAbortState.signal.aborted) {
			return emitInterruptedAssistantMessage(context, config, emit, assistantAbortState);
		}

		const llmContext: Context = {
			systemPrompt: context.systemPrompt,
			messages: llmMessages,
			tools: context.tools,
		};

		const streamFunction = streamFn || streamSimple;
		const resolvedApiKey =
			(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;
		if (assistantAbortState.signal.aborted) {
			return emitInterruptedAssistantMessage(context, config, emit, assistantAbortState);
		}

		const response = await streamFunction(config.model, llmContext, {
			...config,
			apiKey: resolvedApiKey,
			signal: assistantAbortState.signal,
		});

		let partialMessage: AssistantMessage | null = null;
		let addedPartial = false;

		for await (const event of response) {
			switch (event.type) {
				case "start":
					partialMessage = event.partial;
					context.messages.push(partialMessage);
					addedPartial = true;
					await emit({ type: "message_start", message: { ...partialMessage } });
					break;

				case "text_start":
				case "text_delta":
				case "text_end":
				case "thinking_start":
				case "thinking_delta":
				case "thinking_end":
				case "toolcall_start":
				case "toolcall_delta":
				case "toolcall_end":
					if (partialMessage) {
						partialMessage = event.partial;
						context.messages[context.messages.length - 1] = partialMessage;
						await emit({
							type: "message_update",
							assistantMessageEvent: event,
							message: { ...partialMessage },
						});
					}
					break;

				case "done":
				case "error": {
					const finalMessage = normalizeAssistantStopReason(await response.result(), assistantAbortState);
					if (addedPartial) {
						context.messages[context.messages.length - 1] = finalMessage;
					} else {
						context.messages.push(finalMessage);
					}
					if (!addedPartial) {
						await emit({ type: "message_start", message: { ...finalMessage } });
					}
					await emit({ type: "message_end", message: finalMessage });
					return finalMessage;
				}
			}
		}

		const finalMessage = normalizeAssistantStopReason(await response.result(), assistantAbortState);
		if (addedPartial) {
			context.messages[context.messages.length - 1] = finalMessage;
		} else {
			context.messages.push(finalMessage);
			await emit({ type: "message_start", message: { ...finalMessage } });
		}
		await emit({ type: "message_end", message: finalMessage });
		return finalMessage;
	} finally {
		assistantAbortState.cleanup();
	}
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const hasSequentialToolCall = toolCalls.some(
		(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
	);
	if (config.toolExecution === "sequential" || hasSequentialToolCall) {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

type ExecutedToolCallBatch = {
	messages: ToolResultMessage[];
	terminate: boolean;
};

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallOutcome[] = [];
	const messages: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		let finalized: FinalizedToolCallOutcome;
		if (preparation.kind === "immediate") {
			finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			};
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				preparation,
				executed,
				config,
				signal,
			);
		}

		await emitToolExecutionEnd(finalized, emit);
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		finalizedCalls.push(finalized);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	};
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
	const finalizedSlots: (FinalizedToolCallOutcome | undefined)[] = new Array(toolCalls.length).fill(undefined);
	const runnableCalls: { prepared: PreparedToolCall; index: number }[] = [];

	for (let i = 0; i < toolCalls.length; i++) {
		const toolCall = toolCalls[i];
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
			} satisfies FinalizedToolCallOutcome;
			await emitToolExecutionEnd(finalized, emit);
			finalizedSlots[i] = finalized;
		} else {
			runnableCalls.push({ prepared: preparation, index: i });
		}
	}

	const finalizedEntries = await Promise.all(
		runnableCalls.map(async (entry) => {
			const executed = await executePreparedToolCall(entry.prepared, signal, emit);
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				entry.prepared,
				executed,
				config,
				signal,
			);
			await emitToolExecutionEnd(finalized, emit);
			return { finalized, index: entry.index };
		}),
	);

	for (const entry of finalizedEntries) {
		finalizedSlots[entry.index] = entry.finalized;
	}

	const orderedFinalizedCalls = finalizedSlots.filter(
		(finalized): finalized is FinalizedToolCallOutcome => finalized !== undefined,
	);
	const messages: ToolResultMessage[] = [];
	for (const finalized of orderedFinalizedCalls) {
		const toolResultMessage = createToolResultMessage(finalized);
		await emitToolResultMessage(toolResultMessage, emit);
		messages.push(toolResultMessage);
	}

	return {
		messages,
		terminate: shouldTerminateToolBatch(orderedFinalizedCalls),
	};
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
};

function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
	return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	const preflightInterrupted = getInterruptedToolOutcome(config);
	if (preflightInterrupted) {
		return preflightInterrupted;
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}

		const interruptedAfterHook = getInterruptedToolOutcome(config);
		if (interruptedAfterHook) {
			return interruptedAfterHook;
		}

		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
	};
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		content: finalized.result.content,
		details: finalized.result.details,
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
