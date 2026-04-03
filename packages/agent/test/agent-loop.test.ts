import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
	type UserMessage,
} from "@mariozechner/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { agentLoop, agentLoopContinue, runAgentLoop, runAgentLoopContinue } from "../src/agent-loop.js";
import type { AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool, StreamFn } from "../src/types.js";

// Mock stream for testing - mimics MockAssistantStream
class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createUserMessage(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

// Simple identity converter for tests - just passes through standard messages
function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
}

describe("agentLoop with AgentMessage", () => {
	it("should emit events with AgentMessage types", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("Hello");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Hi there!" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();

		// Should have user message and assistant message
		expect(messages.length).toBe(2);
		expect(messages[0].role).toBe("user");
		expect(messages[1].role).toBe("assistant");

		// Verify event sequence
		const eventTypes = events.map((e) => e.type);
		expect(eventTypes).toContain("agent_start");
		expect(eventTypes).toContain("turn_start");
		expect(eventTypes).toContain("message_start");
		expect(eventTypes).toContain("message_end");
		expect(eventTypes).toContain("turn_end");
		expect(eventTypes).toContain("agent_end");
	});

	it("should handle custom message types via convertToLlm", async () => {
		// Create a custom message type
		interface CustomNotification {
			role: "notification";
			text: string;
			timestamp: number;
		}

		const notification: CustomNotification = {
			role: "notification",
			text: "This is a notification",
			timestamp: Date.now(),
		};

		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [notification as unknown as AgentMessage], // Custom message in context
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("Hello");

		let convertedMessages: Message[] = [];
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				// Filter out notifications, convert rest
				convertedMessages = messages
					.filter((m) => (m as { role: string }).role !== "notification")
					.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
				return convertedMessages;
			},
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		// The notification should have been filtered out in convertToLlm
		expect(convertedMessages.length).toBe(1); // Only user message
		expect(convertedMessages[0].role).toBe("user");
	});

	it("should apply transformContext before convertToLlm", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [
				createUserMessage("old message 1"),
				createAssistantMessage([{ type: "text", text: "old response 1" }]),
				createUserMessage("old message 2"),
				createAssistantMessage([{ type: "text", text: "old response 2" }]),
			],
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("new message");

		let transformedMessages: AgentMessage[] = [];
		let convertedMessages: Message[] = [];

		const config: AgentLoopConfig = {
			model: createModel(),
			transformContext: async (messages) => {
				// Keep only last 2 messages (prune old ones)
				transformedMessages = messages.slice(-2);
				return transformedMessages;
			},
			convertToLlm: (messages) => {
				convertedMessages = messages.filter(
					(m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult",
				) as Message[];
				return convertedMessages;
			},
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const _ of stream) {
			// consume
		}

		// transformContext should have been called first, keeping only last 2
		expect(transformedMessages.length).toBe(2);
		// Then convertToLlm receives the pruned messages
		expect(convertedMessages.length).toBe(2);
	});

	it("should preserve existing stream option passthrough, including interrupt fields", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		const isInterrupted = () => false;
		const beforeToolCall = async () => undefined;
		const afterToolCall = async () => undefined;
		const getSteeringMessages = async () => [];
		const getFollowUpMessages = async () => [];
		let receivedOptions: Record<string, unknown> | undefined;

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			interruptSignal: interruptController.signal,
			isInterrupted,
			beforeToolCall,
			afterToolCall,
			getSteeringMessages,
			getFollowUpMessages,
			toolExecution: "sequential",
			sessionId: "session-123",
		};

		const streamFn: StreamFn = (_model, _llmContext, options) => {
			receivedOptions = options as Record<string, unknown> | undefined;
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Hi there!" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);
		for await (const _event of stream) {
			// consume
		}

		expect(receivedOptions).toBeDefined();
		expect(receivedOptions).toMatchObject({
			interruptSignal: interruptController.signal,
			isInterrupted,
			beforeToolCall,
			afterToolCall,
			getSteeringMessages,
			getFollowUpMessages,
			toolExecution: "sequential",
			sessionId: "session-123",
		});
	});

	it("should rewrite aborted assistant streams to interrupted when interrupted", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
		};

		const stream = agentLoop([userPrompt], context, config, undefined, (_model, _llmContext, options) => {
			const signal = options?.signal;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const partial = createAssistantMessage([], "aborted");
				mockStream.push({ type: "start", partial });
				queueMicrotask(() => interruptController.abort());
			});
			signal?.addEventListener(
				"abort",
				() => {
					const aborted = createAssistantMessage([{ type: "text", text: "partial" }], "aborted");
					mockStream.push({ type: "error", reason: "aborted", error: aborted });
				},
				{ once: true },
			);
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		const assistant = [...messages].reverse().find((message) => message.role === "assistant");
		if (!assistant || assistant.role !== "assistant") {
			throw new Error("Expected assistant message");
		}
		expect(assistant.stopReason).toBe("interrupted");
	});

	it("should preserve natural assistant completion if the stream finishes before interrupt takes effect", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
		};

		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "done" }], "stop");
				mockStream.push({ type: "done", reason: "stop", message });
				queueMicrotask(() => interruptController.abort());
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		const assistant = [...messages].reverse().find((message) => message.role === "assistant");
		if (!assistant || assistant.role !== "assistant") {
			throw new Error("Expected assistant message");
		}
		expect(assistant.stopReason).toBe("stop");
	});

	it("should synthesize an interrupted assistant message when transformContext is interrupted", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		let streamCalled = false;
		const config: AgentLoopConfig = {
			model: createModel(),
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
			transformContext: async (messages, signal) => {
				interruptController.abort();
				expect(signal?.aborted).toBe(true);
				return messages;
			},
			convertToLlm: identityConverter,
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			streamCalled = true;
			const mockStream = new MockAssistantStream();
			return mockStream;
		});

		for await (const event of stream) {
			events.push(event);
		}

		const assistantEvents = events.filter(
			(event): event is Extract<AgentEvent, { type: "message_start" | "message_end" }> =>
				event.type === "message_start" || event.type === "message_end",
		);
		const assistantMessages = assistantEvents.filter((event) => event.message.role === "assistant");
		const messages = await stream.result();
		const assistant = [...messages].reverse().find((message) => message.role === "assistant");
		if (!assistant || assistant.role !== "assistant") {
			throw new Error("Expected assistant message");
		}

		expect(streamCalled).toBe(false);
		expect(assistantMessages).toHaveLength(2);
		expect(assistantMessages[0].type).toBe("message_start");
		expect(assistantMessages[1].type).toBe("message_end");
		expect(assistant.content).toEqual([]);
		expect(assistant.usage).toEqual(createUsage());
		expect(assistant.stopReason).toBe("interrupted");
	});

	it("should synthesize an interrupted assistant message when api key resolution is interrupted", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		let streamCalled = false;
		const config: AgentLoopConfig = {
			model: createModel(),
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
			convertToLlm: identityConverter,
			getApiKey: async () => {
				interruptController.abort();
				await Promise.resolve();
				return "resolved-api-key";
			},
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			streamCalled = true;
			const mockStream = new MockAssistantStream();
			return mockStream;
		});

		for await (const event of stream) {
			events.push(event);
		}

		const assistantEvents = events.filter(
			(event): event is Extract<AgentEvent, { type: "message_start" | "message_end" }> =>
				event.type === "message_start" || event.type === "message_end",
		);
		const assistantMessages = assistantEvents.filter((event) => event.message.role === "assistant");
		const messages = await stream.result();
		const assistant = [...messages].reverse().find((message) => message.role === "assistant");
		if (!assistant || assistant.role !== "assistant") {
			throw new Error("Expected assistant message");
		}

		expect(streamCalled).toBe(false);
		expect(assistantMessages).toHaveLength(2);
		expect(assistantMessages[0].type).toBe("message_start");
		expect(assistantMessages[1].type).toBe("message_end");
		expect(assistant.content).toEqual([]);
		expect(assistant.usage).toEqual(createUsage());
		expect(assistant.stopReason).toBe("interrupted");
	});

	it("should synthesize an interrupted assistant message when convertToLlm is interrupted", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		let streamCalled = false;
		const config: AgentLoopConfig = {
			model: createModel(),
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
			convertToLlm: (messages) => {
				interruptController.abort();
				return identityConverter(messages);
			},
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			streamCalled = true;
			const mockStream = new MockAssistantStream();
			return mockStream;
		});

		for await (const event of stream) {
			events.push(event);
		}

		const assistantEvents = events.filter(
			(event): event is Extract<AgentEvent, { type: "message_start" | "message_end" }> =>
				event.type === "message_start" || event.type === "message_end",
		);
		const assistantMessages = assistantEvents.filter((event) => event.message.role === "assistant");
		const messages = await stream.result();
		const assistant = [...messages].reverse().find((message) => message.role === "assistant");
		if (!assistant || assistant.role !== "assistant") {
			throw new Error("Expected assistant message");
		}

		expect(streamCalled).toBe(false);
		expect(assistantMessages).toHaveLength(2);
		expect(assistantMessages[0].type).toBe("message_start");
		expect(assistantMessages[1].type).toBe("message_end");
		expect(assistant.content).toEqual([]);
		expect(assistant.usage).toEqual(createUsage());
		expect(assistant.stopReason).toBe("interrupted");
	});

	it("should keep assistant stopReason aborted when abort follows interrupt", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};
		const userPrompt: AgentMessage = createUserMessage("Hello");
		const interruptController = new AbortController();
		const hardAbortController = new AbortController();
		let releaseError: (() => void) | undefined;
		const errorReady = new Promise<void>((resolve) => {
			releaseError = resolve;
		});
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
		};

		const stream = agentLoop(
			[userPrompt],
			context,
			config,
			hardAbortController.signal,
			(_model, _llmContext, options) => {
				const signal = options?.signal;
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					mockStream.push({ type: "start", partial: createAssistantMessage([], "aborted") });
					queueMicrotask(() => interruptController.abort());
				});
				signal?.addEventListener(
					"abort",
					() => {
						void errorReady.then(() => {
							const message = createAssistantMessage([{ type: "text", text: "partial" }], "aborted");
							mockStream.push({ type: "error", reason: "aborted", error: message });
						});
					},
					{ once: true },
				);
				return mockStream;
			},
		);

		await Promise.resolve();
		hardAbortController.abort();
		releaseError?.();

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		const assistant = [...messages].reverse().find((message) => message.role === "assistant");
		if (!assistant || assistant.role !== "assistant") {
			throw new Error("Expected assistant message");
		}
		expect(assistant.stopReason).toBe("aborted");
	});

	it("should handle tool calls and results", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo something");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					// First call: return tool call
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
						"toolUse",
					);
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					// Second call: return final response
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		// Tool should have been executed
		expect(executed).toEqual(["hello"]);

		// Should have tool execution events
		const toolStart = events.find((e) => e.type === "tool_execution_start");
		const toolEnd = events.find((e) => e.type === "tool_execution_end");
		expect(toolStart).toBeDefined();
		expect(toolEnd).toBeDefined();
		if (toolEnd?.type === "tool_execution_end") {
			expect(toolEnd.isError).toBe(false);
		}
	});

	it("should execute mutated beforeToolCall args without revalidation", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: Array<string | number> = [];
		const tool: AgentTool<typeof toolSchema, { value: string | number }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value as string | number);
				return {
					content: [{ type: "text", text: `echoed: ${String(params.value)}` }],
					details: { value: params.value as string | number },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo something");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			beforeToolCall: async ({ args }) => {
				const mutableArgs = args as { value: string | number };
				mutableArgs.value = 123;
				return undefined;
			},
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
						"toolUse",
					);
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);
		for await (const _event of stream) {
			// consume
		}

		expect(executed).toEqual([123]);
	});

	it("should prepare tool arguments for validation", async () => {
		const replaceSchema = Type.Object({ oldText: Type.String(), newText: Type.String() });
		const toolSchema = Type.Object({ edits: Type.Array(replaceSchema) });
		const executed: Array<Array<{ oldText: string; newText: string }>> = [];
		const tool: AgentTool<typeof toolSchema, { count: number }> = {
			name: "edit",
			label: "Edit",
			description: "Edit tool",
			parameters: toolSchema,
			prepareArguments(args) {
				if (!args || typeof args !== "object") {
					return args as { edits: { oldText: string; newText: string }[] };
				}
				const input = args as {
					edits?: Array<{ oldText: string; newText: string }>;
					oldText?: string;
					newText?: string;
				};
				if (typeof input.oldText !== "string" || typeof input.newText !== "string") {
					return args as { edits: { oldText: string; newText: string }[] };
				}
				return {
					edits: [...(input.edits ?? []), { oldText: input.oldText, newText: input.newText }],
				};
			},
			async execute(_toolCallId, params) {
				executed.push(params.edits);
				return {
					content: [{ type: "text", text: `edited ${params.edits.length}` }],
					details: { count: params.edits.length },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("edit something");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{
								type: "toolCall",
								id: "tool-1",
								name: "edit",
								arguments: { oldText: "before", newText: "after" },
							},
						],
						"toolUse",
					);
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);
		for await (const _event of stream) {
			// consume
		}

		expect(executed).toEqual([[{ oldText: "before", newText: "after" }]]);
	});

	it("should emit tool_execution_end in completion order but persist tool results in source order", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseFirst?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolExecutionEndIds = events.flatMap((event) => {
			if (event.type !== "tool_execution_end") {
				return [];
			}
			return [event.toolCallId];
		});
		const toolResultIds = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message.toolCallId];
		});
		const turnToolResultIds = events.flatMap((event) => {
			if (event.type !== "turn_end") {
				return [];
			}
			return event.toolResults.map((toolResult) => toolResult.toolCallId);
		});

		expect(parallelObserved).toBe(true);
		expect(toolExecutionEndIds).toEqual(["tool-2", "tool-1"]);
		expect(toolResultIds).toEqual(["tool-1", "tool-2"]);
		expect(turnToolResultIds).toEqual(["tool-1", "tool-2"]);
	});

	it("should inject queued messages after all tool calls complete", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("start");
		const queuedUserMessage: AgentMessage = createUserMessage("interrupt");

		let queuedDelivered = false;
		let callIndex = 0;
		let sawInterruptInContext = false;

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "sequential",
			getSteeringMessages: async () => {
				// Return steering message after tool execution has started.
				if (executed.length >= 1 && !queuedDelivered) {
					queuedDelivered = true;
					return [queuedUserMessage];
				}
				return [];
			},
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, (_model, ctx, _options) => {
			// Check if interrupt message is in context on second call
			if (callIndex === 1) {
				sawInterruptInContext = ctx.messages.some(
					(m) => m.role === "user" && typeof m.content === "string" && m.content === "interrupt",
				);
			}

			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					// First call: return two tool calls
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					// Second call: return final response
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		for await (const event of stream) {
			events.push(event);
		}

		// Both tools should execute before steering is injected
		expect(executed).toEqual(["first", "second"]);

		const toolEnds = events.filter(
			(e): e is Extract<AgentEvent, { type: "tool_execution_end" }> => e.type === "tool_execution_end",
		);
		expect(toolEnds.length).toBe(2);
		expect(toolEnds[0].isError).toBe(false);
		expect(toolEnds[1].isError).toBe(false);

		// Queued message should appear in events after both tool result messages
		const eventSequence = events.flatMap((event) => {
			if (event.type !== "message_start") return [];
			if (event.message.role === "toolResult") return [`tool:${event.message.toolCallId}`];
			if (event.message.role === "user" && typeof event.message.content === "string") {
				return [event.message.content];
			}
			return [];
		});
		expect(eventSequence).toContain("interrupt");
		expect(eventSequence.indexOf("tool:tool-1")).toBeLessThan(eventSequence.indexOf("interrupt"));
		expect(eventSequence.indexOf("tool:tool-2")).toBeLessThan(eventSequence.indexOf("interrupt"));

		// Interrupt message should be in context when second LLM call is made
		expect(sawInterruptInContext).toBe(true);
	});

	it("should force sequential execution when a tool has executionMode=sequential even with default parallel config", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const slowTool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "slow",
			label: "Slow",
			description: "Slow tool",
			parameters: toolSchema,
			executionMode: "sequential",
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `slow: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [slowTool],
		};

		const userPrompt: AgentMessage = createUserMessage("run both");
		// config is parallel (default), but tool forces sequential
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "slow", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "slow", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseFirst?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// With sequential execution, second tool should NOT start before first finishes
		expect(parallelObserved).toBe(false);

		const toolResultIds = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message.toolCallId];
		});
		expect(toolResultIds).toEqual(["tool-1", "tool-2"]);
	});

	it("should force sequential execution when one of multiple tools has executionMode=sequential", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executionOrder: string[] = [];
		let releaseSlow: (() => void) | undefined;
		const slowDone = new Promise<void>((resolve) => {
			releaseSlow = resolve;
		});

		const slowTool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "slow",
			label: "Slow",
			description: "Slow tool",
			parameters: toolSchema,
			executionMode: "sequential",
			async execute(_toolCallId, params) {
				executionOrder.push(`slow:${params.value}`);
				if (params.value === "a") {
					await slowDone;
				}
				return {
					content: [{ type: "text", text: `slow: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const fastTool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "fast",
			label: "Fast",
			description: "Fast tool",
			parameters: toolSchema,
			// no executionMode = defaults to parallel
			async execute(_toolCallId, params) {
				executionOrder.push(`fast:${params.value}`);
				return {
					content: [{ type: "text", text: `fast: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [slowTool, fastTool],
		};

		const userPrompt: AgentMessage = createUserMessage("run both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			// parallel by default, but slowTool forces sequential
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "slow", arguments: { value: "a" } },
							{ type: "toolCall", id: "tool-2", name: "fast", arguments: { value: "b" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseSlow?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// Fast tool should NOT run before slow tool finishes
		expect(executionOrder[0]).toBe("slow:a");
		expect(executionOrder).toContain("fast:b");
	});

	it("should allow parallel execution when all tools have executionMode=parallel", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			executionMode: "parallel",
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseFirst?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// With executionMode=parallel, second tool should start before first finishes
		expect(parallelObserved).toBe(true);
	});

	it("should stop after a tool batch when every tool result sets terminate=true", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
					terminate: true,
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage(
					[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
					"toolUse",
				);
				mockStream.push({ type: "done", reason: "toolUse", message });
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(llmCalls).toBe(1);
		expect(messages.map((message) => message.role)).toEqual(["user", "assistant", "toolResult"]);
		expect(events.filter((event) => event.type === "turn_end")).toHaveLength(1);
	});

	it("should continue after parallel tool calls when not all tool results terminate", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
					terminate: params.value === "first",
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
		};

		let callIndex = 0;
		const stream = agentLoop([createUserMessage("echo both")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		expect(callIndex).toBe(2);
		expect(messages.map((message) => message.role)).toEqual([
			"user",
			"assistant",
			"toolResult",
			"toolResult",
			"assistant",
		]);
	});

	it("should allow afterToolCall to mark a tool batch as terminating", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolCall: async () => ({ terminate: true }),
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage(
					[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
					"toolUse",
				);
				mockStream.push({ type: "done", reason: "toolUse", message });
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		expect(llmCalls).toBe(1);
	});
	it("should propagate isInterrupted errors during tool preflight", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let executeCalled = false;
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executeCalled = true;
				return {
					content: [{ type: "text", text: "unexpected" }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [createUserMessage("run tool")],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => {
				throw new Error("interrupt check failed");
			},
		};

		await expect(
			runAgentLoopContinue(
				context,
				config,
				async () => undefined,
				undefined,
				() => {
					const mockStream = new MockAssistantStream();
					queueMicrotask(() => {
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } }],
								"toolUse",
							),
						});
					});
					return mockStream;
				},
			),
		).rejects.toThrow("interrupt check failed");
		expect(executeCalled).toBe(false);
	});

	it("should block remaining sequential tool calls after interruption", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let interrupted = false;
		let firstExecuted = false;
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				firstExecuted = true;
				interrupted = true;
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "sequential",
			isInterrupted: () => interrupted,
		};

		const stream = agentLoop([createUserMessage("run tools")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				mockStream.push({
					type: "done",
					reason: "toolUse",
					message: createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					),
				});
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolResults = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message];
		});
		const blockedResult = toolResults.find((message) => message.toolCallId === "tool-2");
		if (!blockedResult) {
			throw new Error("Expected blocked tool result");
		}

		expect(firstExecuted).toBe(true);
		expect(executed).toEqual(["first"]);
		expect(toolResults.map((message) => ({ id: message.toolCallId, isError: message.isError }))).toEqual([
			{ id: "tool-1", isError: false },
			{ id: "tool-2", isError: true },
		]);
		expect(blockedResult.content).toEqual([
			{ type: "text", text: "Tool execution was blocked: session interrupted" },
		]);
		const agentEnd = events.find(
			(event): event is Extract<AgentEvent, { type: "agent_end" }> => event.type === "agent_end",
		);
		expect(
			agentEnd?.messages.some((message) => message.role === "assistant" && message.stopReason === "toolUse"),
		).toBe(true);
	});

	it("should block parallel tool calls that have not finished preflight when interrupted", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let interrupted = false;
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
			isInterrupted: () => interrupted,
			beforeToolCall: async ({ toolCall }) => {
				if (toolCall.id === "tool-2") {
					interrupted = true;
				}
				return undefined;
			},
		};

		const stream = agentLoop([createUserMessage("run tools")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				mockStream.push({
					type: "done",
					reason: "toolUse",
					message: createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
							{ type: "toolCall", id: "tool-3", name: "echo", arguments: { value: "third" } },
						],
						"toolUse",
					),
				});
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolEnds = events.filter(
			(event): event is Extract<AgentEvent, { type: "tool_execution_end" }> => event.type === "tool_execution_end",
		);
		const blockedEnds = toolEnds.filter((event) => event.isError);
		const blockedMessages = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult" || !event.message.isError) {
				return [];
			}
			return [event.message];
		});

		expect(executed).toEqual(["first"]);
		// Events fire in execution order (blocked tools emit during preflight, executed tools after)
		expect(
			toolEnds
				.map((event) => ({ id: event.toolCallId, isError: event.isError }))
				.sort((left, right) => left.id.localeCompare(right.id)),
		).toEqual([
			{ id: "tool-1", isError: false },
			{ id: "tool-2", isError: true },
			{ id: "tool-3", isError: true },
		]);
		// The returned tool results array (used for context and turn_end) must be in source order
		const turnEnd = events.find(
			(event): event is Extract<AgentEvent, { type: "turn_end" }> => event.type === "turn_end",
		);
		if (!turnEnd || turnEnd.type !== "turn_end") throw new Error("Expected turn_end");
		const turnEndToolResults = turnEnd.toolResults;
		expect(turnEndToolResults.map((r) => r.toolCallId)).toEqual(["tool-1", "tool-2", "tool-3"]);
		expect(blockedEnds).toHaveLength(2);
		expect(blockedMessages.map((message) => message.content[0])).toEqual([
			{ type: "text", text: "Tool execution was blocked: session interrupted" },
			{ type: "text", text: "Tool execution was blocked: session interrupted" },
		]);
	});

	it("should block a tool call if interruption happens during beforeToolCall", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let interrupted = false;
		let executeCalled = false;
		let releaseHook: (() => void) | undefined;
		const hookReady = new Promise<void>((resolve) => {
			releaseHook = resolve;
		});
		let resolveHookStarted: (() => void) | undefined;
		const hookStarted = new Promise<void>((resolve) => {
			resolveHookStarted = resolve;
		});
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executeCalled = true;
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			beforeToolCall: async () => {
				resolveHookStarted?.();
				await hookReady;
				return undefined;
			},
		};

		const stream = agentLoop([createUserMessage("run tool")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				mockStream.push({
					type: "done",
					reason: "toolUse",
					message: createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } }],
						"toolUse",
					),
				});
			});
			return mockStream;
		});

		await hookStarted;
		interrupted = true;
		releaseHook?.();

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolEnd = events.find(
			(event): event is Extract<AgentEvent, { type: "tool_execution_end" }> => event.type === "tool_execution_end",
		);
		if (!toolEnd) {
			throw new Error("Expected tool execution end event");
		}

		expect(executeCalled).toBe(false);
		expect(toolEnd.isError).toBe(true);
		expect(toolEnd.result.content).toEqual([
			{ type: "text", text: "Tool execution was blocked: session interrupted" },
		]);
	});

	it("should propagate interrupt predicate failures after beforeToolCall", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo",
			parameters: toolSchema,
			async execute() {
				throw new Error("tool should not execute");
			},
		};
		let interruptChecks = 0;
		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			beforeToolCall: async () => undefined,
			isInterrupted: () => {
				interruptChecks++;
				if (interruptChecks >= 3) {
					throw new Error("interrupt predicate failed");
				}
				return false;
			},
		};

		await expect(
			runAgentLoop(
				[createUserMessage("run tool")],
				context,
				config,
				async () => undefined,
				undefined,
				() => {
					const stream = new MockAssistantStream();
					queueMicrotask(() => {
						stream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[
									{
										type: "toolCall",
										id: "tool-1",
										name: "echo",
										arguments: { value: "hello" },
									},
								],
								"toolUse",
							),
						});
					});
					return stream;
				},
			),
		).rejects.toThrow("interrupt predicate failed");
	});

	it("should allow all parallel tools to finish when interruption happens after dispatch", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let interrupted = false;
		const started: string[] = [];
		let releaseTools: (() => void) | undefined;
		const toolsCanFinish = new Promise<void>((resolve) => {
			releaseTools = resolve;
		});
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				started.push(params.value);
				if (started.length === 2) {
					interrupted = true;
					releaseTools?.();
				}
				await toolsCanFinish;
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
			isInterrupted: () => interrupted,
		};

		const stream = agentLoop([createUserMessage("run tools")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				mockStream.push({
					type: "done",
					reason: "toolUse",
					message: createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					),
				});
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolEnds = events.filter(
			(event): event is Extract<AgentEvent, { type: "tool_execution_end" }> => event.type === "tool_execution_end",
		);
		expect(started.sort()).toEqual(["first", "second"]);
		expect(toolEnds.map((event) => event.isError)).toEqual([false, false]);
	});

	it("should exit after turn_end when interrupt fires after all tools complete but before next turn", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let interrupted = false;
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		let streamCallCount = 0;
		let steeringPollCount = 0;
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "sequential",
			isInterrupted: () => interrupted,
			afterToolCall: async ({ toolCall }) => {
				if (toolCall.id === "tool-2") {
					interrupted = true;
				}
				return undefined;
			},
			getSteeringMessages: async () => {
				steeringPollCount++;
				return [];
			},
			getFollowUpMessages: async () => [],
		};

		const events: AgentEvent[] = [];
		await runAgentLoop(
			[createUserMessage("run tools")],
			{ systemPrompt: "", messages: [], tools: [tool] },
			config,
			async (event) => {
				events.push(event);
			},
			undefined,
			() => {
				streamCallCount++;
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					mockStream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(
							[
								{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
								{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
							],
							"toolUse",
						),
					});
				});
				return mockStream;
			},
		);

		expect(executed).toEqual(["first", "second"]);
		expect(streamCallCount).toBe(1);

		const eventTypes = events.map((event) => event.type);
		const turnEndIndex = eventTypes.lastIndexOf("turn_end");
		const agentEndIndex = eventTypes.lastIndexOf("agent_end");
		expect(turnEndIndex).toBeGreaterThan(-1);
		expect(agentEndIndex).toBeGreaterThan(turnEndIndex);

		const toolResults = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message];
		});
		expect(toolResults).toHaveLength(2);
		expect(toolResults.every((result) => !result.isError)).toBe(true);
		// Initial steering poll happens before the first turn, but the post-turn steering poll
		// should be skipped because the interrupt check fires immediately after turn_end.
		expect(steeringPollCount).toBe(1);
	});

	it.each([
		{
			name: "prompt callers",
			run: async (
				config: AgentLoopConfig,
				emittedEvents: AgentEvent[],
				captureContext: (messages: Message[]) => void,
			) =>
				runAgentLoop(
					[createUserMessage("retry me")],
					{ systemPrompt: "", messages: [], tools: [] },
					config,
					async (event) => {
						emittedEvents.push(event);
					},
					undefined,
					(_model, llmAgentContext) => {
						captureContext(llmAgentContext.messages);
						const stream = new MockAssistantStream();
						queueMicrotask(() => {
							stream.push({
								type: "done",
								reason: "stop",
								message: createAssistantMessage([{ type: "text", text: "done" }]),
							});
						});
						return stream;
					},
				),
			expectedContext: [
				{ role: "user", content: "retry me" },
				{ role: "user", content: "queued steer" },
			],
			expectedRoles: ["user", "user", "assistant"],
		},
		{
			name: "continue callers",
			run: async (
				config: AgentLoopConfig,
				emittedEvents: AgentEvent[],
				captureContext: (messages: Message[]) => void,
			) =>
				runAgentLoopContinue(
					{ systemPrompt: "", messages: [createUserMessage("retry me")], tools: [] },
					config,
					async (event) => {
						emittedEvents.push(event);
					},
					undefined,
					(_model, llmAgentContext) => {
						captureContext(llmAgentContext.messages);
						const stream = new MockAssistantStream();
						queueMicrotask(() => {
							stream.push({
								type: "done",
								reason: "stop",
								message: createAssistantMessage([{ type: "text", text: "done" }]),
							});
						});
						return stream;
					},
				),
			expectedContext: [
				{ role: "user", content: "retry me" },
				{ role: "user", content: "queued steer" },
			],
			expectedRoles: ["user", "assistant"],
		},
	])(
		"should inject queued steering before the first assistant turn for $name",
		async ({ run, expectedContext, expectedRoles, name }) => {
			const emittedEvents: AgentEvent[] = [];
			let steeringPolled = 0;
			let llmContext: Message[] | undefined;
			const config: AgentLoopConfig = {
				model: createModel(),
				convertToLlm: identityConverter,
				getSteeringMessages: async () => {
					steeringPolled++;
					return steeringPolled === 1 ? [createUserMessage("queued steer")] : [];
				},
			};

			const messages = await run(config, emittedEvents, (messages) => {
				llmContext = messages;
			});

			expect(steeringPolled).toBe(2);
			expect(llmContext).toMatchObject(expectedContext);
			expect(messages.map((message) => message.role)).toEqual(expectedRoles);
			expect(messages).toContainEqual(expect.objectContaining({ role: "user", content: "queued steer" }));
			if (name === "prompt callers") {
				expect(messages[0]).toMatchObject({ role: "user", content: "retry me" });
			} else {
				expect(messages[0]).toMatchObject({ role: "user", content: "queued steer" });
			}
			expect(emittedEvents.map((event) => event.type)).toContain("message_start");
		},
	);

	it("should propagate isInterrupted errors instead of treating them as interruption state", async () => {
		const context: AgentContext = {
			systemPrompt: "",
			messages: [createUserMessage("retry me")],
			tools: [],
		};
		let interruptChecks = 0;
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => {
				interruptChecks++;
				if (interruptChecks === 1) {
					throw new Error("interrupt predicate failed");
				}
				return false;
			},
		};

		await expect(
			runAgentLoopContinue(
				context,
				config,
				async () => undefined,
				undefined,
				() => {
					const stream = new MockAssistantStream();
					queueMicrotask(() => {
						stream.push({
							type: "done",
							reason: "stop",
							message: createAssistantMessage([{ type: "text", text: "done" }], "stop"),
						});
					});
					return stream;
				},
			),
		).rejects.toThrow("interrupt predicate failed");
	});

	it("should preserve queued steering and follow-up messages after interruption", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("steer later")];
		const followUpQueue: AgentMessage[] = [createUserMessage("follow up later")];
		let interrupted = false;
		let steeringPolls = 0;
		let followUpPolls = 0;

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				steeringPolls++;
				if (steeringPolls === 1) {
					return [];
				}
				return steeringQueue.splice(0, steeringQueue.length);
			},
			getFollowUpMessages: async () => {
				followUpPolls++;
				return followUpQueue.splice(0, followUpQueue.length);
			},
		};

		const stream = agentLoop([createUserMessage("interrupt me")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				interrupted = true;
				mockStream.push({
					type: "done",
					reason: "stop",
					message: createAssistantMessage([{ type: "text", text: "partial" }], "stop"),
				});
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		expect(steeringPolls).toBe(1);
		expect(followUpPolls).toBe(0);
		expect(steeringQueue).toHaveLength(1);
		expect(followUpQueue).toHaveLength(1);
	});

	it("should settle the first turn before ending when already interrupted", async () => {
		const interruptController = new AbortController();
		interruptController.abort();
		let streamCalled = false;
		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			interruptSignal: interruptController.signal,
			isInterrupted: () => interruptController.signal.aborted,
		};

		const events: AgentEvent[] = [];
		const messages = await runAgentLoop(
			[createUserMessage("interrupt me")],
			context,
			config,
			async (event) => {
				events.push(event);
			},
			undefined,
			() => {
				streamCalled = true;
				throw new Error("stream should not be called");
			},
		);

		const assistant = messages.find((message): message is AssistantMessage => message.role === "assistant");
		expect(streamCalled).toBe(false);
		expect(assistant?.stopReason).toBe("interrupted");
		expect(events.map((event) => event.type)).toEqual([
			"agent_start",
			"turn_start",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"turn_end",
			"agent_end",
		]);
		const turnEndIndex = events.findIndex((event) => event.type === "turn_end");
		const agentEndIndex = events.findIndex((event) => event.type === "agent_end");
		expect(turnEndIndex).toBeLessThan(agentEndIndex);
	});

	it("should emit turn_end before agent_end when interrupted after the first assistant turn", async () => {
		let interrupted = false;
		let steeringPolls = 0;
		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				steeringPolls++;
				return [];
			},
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([createUserMessage("interrupt me")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				interrupted = true;
				mockStream.push({
					type: "done",
					reason: "stop",
					message: createAssistantMessage([{ type: "text", text: "partial" }], "stop"),
				});
			});
			return mockStream;
		});

		for await (const event of stream) {
			events.push(event);
		}

		const eventTypes = events.map((event) => event.type);
		expect(eventTypes).toEqual([
			"agent_start",
			"turn_start",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"turn_end",
			"agent_end",
		]);
		expect(steeringPolls).toBe(1);
	});

	it("should stop before processing steering messages returned after interruption", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("steer later")];
		let interrupted = false;
		let steeringPolls = 0;
		let assistantTurns = 0;

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				steeringPolls++;
				if (steeringPolls === 1) {
					return [];
				}
				interrupted = true;
				return steeringQueue.splice(0, steeringQueue.length);
			},
			requeueSteeringMessages: (messages) => steeringQueue.unshift(...messages),
		};

		const stream = agentLoop([createUserMessage("interrupt me")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				assistantTurns++;
				mockStream.push({
					type: "done",
					reason: "stop",
					message: createAssistantMessage([{ type: "text", text: `turn ${assistantTurns}` }], "stop"),
				});
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		expect(assistantTurns).toBe(1);
		expect(steeringPolls).toBe(2);
		expect(steeringQueue).toHaveLength(1);
	});

	it("should requeue pending steering messages and end before the next turn when interrupted", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("queued steer")];
		let interrupted = false;
		let assistantTurns = 0;
		let steeringPolls = 0;
		const events: AgentEvent[] = [];

		const context: AgentContext = {
			systemPrompt: "",
			messages: [createUserMessage("retry me")],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				steeringPolls++;
				if (steeringPolls === 1) {
					return [];
				}
				interrupted = true;
				return steeringQueue.splice(0, steeringQueue.length);
			},
			requeueSteeringMessages: (messages) => steeringQueue.unshift(...messages),
		};

		const messages = await runAgentLoopContinue(
			context,
			config,
			async (event) => {
				events.push(event);
			},
			undefined,
			() => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					assistantTurns++;
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: `turn ${assistantTurns}` }], "stop"),
					});
				});
				return stream;
			},
		);

		expect(assistantTurns).toBe(1);
		expect(steeringPolls).toBe(2);
		expect(steeringQueue).toEqual([expect.objectContaining({ role: "user", content: "queued steer" })]);
		expect(messages).toHaveLength(1);
		expect(messages[0]).toMatchObject({ role: "assistant", stopReason: "stop" });
		expect(events.map((event) => event.type)).toEqual([
			"agent_start",
			"turn_start",
			"message_start",
			"message_end",
			"turn_end",
			"agent_end",
		]);
	});

	it("should continue through the first turn when interruption is already set during the initial steering poll", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("queued steer")];
		let interrupted = false;
		let assistantTurns = 0;
		const events: AgentEvent[] = [];

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				interrupted = true;
				return steeringQueue.splice(0, steeringQueue.length);
			},
			requeueSteeringMessages: (messages) => steeringQueue.unshift(...messages),
		};

		const messages = await runAgentLoop(
			[createUserMessage("interrupt me")],
			context,
			config,
			async (event) => {
				events.push(event);
			},
			undefined,
			() => {
				assistantTurns++;
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "done" }], "stop"),
					});
				});
				return stream;
			},
		);

		expect(assistantTurns).toBe(1);
		expect(steeringQueue).toEqual([]);
		expect(messages).toEqual([
			expect.objectContaining({ role: "user", content: "interrupt me" }),
			expect.objectContaining({ role: "user", content: "queued steer" }),
			expect.objectContaining({ role: "assistant", stopReason: "stop" }),
		]);
		expect(events.map((event) => event.type)).toEqual([
			"agent_start",
			"turn_start",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"turn_end",
			"agent_end",
		]);
	});

	it("should requeue steering messages if interruption lands during the poll before the next turn", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("queued steer")];
		let interrupted = false;
		let steeringPolls = 0;
		let assistantTurns = 0;

		const context: AgentContext = {
			systemPrompt: "",
			messages: [createUserMessage("retry me")],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				steeringPolls++;
				if (steeringPolls === 1) {
					return [];
				}
				interrupted = true;
				return steeringQueue.splice(0, steeringQueue.length);
			},
			requeueSteeringMessages: (messages) => steeringQueue.unshift(...messages),
		};

		const messages = await runAgentLoopContinue(
			context,
			config,
			async () => undefined,
			undefined,
			() => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					assistantTurns++;
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: `turn ${assistantTurns}` }], "stop"),
					});
				});
				return stream;
			},
		);

		expect(assistantTurns).toBe(1);
		expect(steeringPolls).toBe(2);
		expect(steeringQueue).toEqual([expect.objectContaining({ role: "user", content: "queued steer" })]);
		expect(messages).toHaveLength(1);
		expect(messages[0]).toMatchObject({ role: "assistant", stopReason: "stop" });
	});

	it("should stop before processing follow-up messages returned after interruption", async () => {
		const followUpQueue: AgentMessage[] = [createUserMessage("follow up later")];
		let interrupted = false;
		let releaseFollowUpPoll: (() => void) | undefined;
		const followUpPollCanFinish = new Promise<void>((resolve) => {
			releaseFollowUpPoll = resolve;
		});
		let resolveFollowUpPollStarted: (() => void) | undefined;
		const followUpPollStarted = new Promise<void>((resolve) => {
			resolveFollowUpPollStarted = resolve;
		});
		let followUpPolls = 0;
		let assistantTurns = 0;

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => [],
			getFollowUpMessages: async () => {
				followUpPolls++;
				resolveFollowUpPollStarted?.();
				await followUpPollCanFinish;
				return followUpQueue.splice(0, followUpQueue.length);
			},
			requeueFollowUpMessages: (messages) => followUpQueue.unshift(...messages),
		};

		const stream = agentLoop([createUserMessage("interrupt me")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				assistantTurns++;
				mockStream.push({
					type: "done",
					reason: "stop",
					message: createAssistantMessage([{ type: "text", text: `turn ${assistantTurns}` }], "stop"),
				});
			});
			return mockStream;
		});

		await followUpPollStarted;
		interrupted = true;
		releaseFollowUpPoll?.();

		for await (const _event of stream) {
			// consume
		}

		expect(assistantTurns).toBe(1);
		expect(followUpPolls).toBe(1);
		expect(followUpQueue).toHaveLength(1);
	});

	it("should finish consuming pending steering messages before ending on interruption", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("queued steer 1"), createUserMessage("queued steer 2")];
		let interrupted = false;
		let steeringPolls = 0;
		const events: AgentEvent[] = [];
		let assistantTurns = 0;

		const context: AgentContext = {
			systemPrompt: "",
			messages: [createUserMessage("retry me")],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				steeringPolls++;
				if (steeringPolls === 1) {
					return [];
				}
				return steeringQueue.splice(0, steeringQueue.length);
			},
			requeueSteeringMessages: (messages) => steeringQueue.unshift(...messages),
		};

		const messages = await runAgentLoopContinue(
			context,
			config,
			async (event) => {
				events.push(event);
				if (
					event.type === "message_end" &&
					event.message.role === "user" &&
					event.message.content === "queued steer 1"
				) {
					interrupted = true;
				}
			},
			undefined,
			() => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					assistantTurns++;
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: `turn ${assistantTurns}` }], "stop"),
					});
				});
				return stream;
			},
		);

		expect(assistantTurns).toBe(2);
		expect(steeringPolls).toBe(2);
		expect(steeringQueue).toEqual([]);
		expect(messages).toEqual([
			expect.objectContaining({ role: "assistant", stopReason: "stop" }),
			expect.objectContaining({ role: "user", content: "queued steer 1" }),
			expect.objectContaining({ role: "user", content: "queued steer 2" }),
			expect.objectContaining({ role: "assistant", stopReason: "stop" }),
		]);
		expect(events.map((event) => event.type)).toEqual([
			"agent_start",
			"turn_start",
			"message_start",
			"message_end",
			"turn_end",
			"turn_start",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"turn_end",
			"agent_end",
		]);
	});

	it("should requeue polled steering without fabricating an aborted assistant message", async () => {
		const steeringQueue: AgentMessage[] = [createUserMessage("queued steer")];
		let interrupted = false;
		let releaseSteeringPoll: (() => void) | undefined;
		const steeringPollCanFinish = new Promise<void>((resolve) => {
			releaseSteeringPoll = resolve;
		});
		let resolveSteeringPollStarted: (() => void) | undefined;
		const steeringPollStarted = new Promise<void>((resolve) => {
			resolveSteeringPollStarted = resolve;
		});
		const abortController = new AbortController();
		const assistantMessages: AssistantMessage[] = [];

		const context: AgentContext = {
			systemPrompt: "",
			messages: [createUserMessage("retry me")],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			skipInitialSteeringPoll: true,
			isInterrupted: () => interrupted,
			getSteeringMessages: async () => {
				resolveSteeringPollStarted?.();
				await steeringPollCanFinish;
				return steeringQueue.splice(0, steeringQueue.length);
			},
			requeueSteeringMessages: (queuedMessages) => steeringQueue.unshift(...queuedMessages),
		};

		const stream = agentLoopContinue(context, config, abortController.signal, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				mockStream.push({
					type: "done",
					reason: "stop",
					message: createAssistantMessage([{ type: "text", text: "done" }], "stop"),
				});
			});
			return mockStream;
		});
		await steeringPollStarted;
		interrupted = true;
		abortController.abort();
		releaseSteeringPoll?.();

		for await (const event of stream) {
			if (event.type === "message_end" && event.message.role === "assistant") {
				assistantMessages.push(event.message);
			}
		}

		expect(assistantMessages).toEqual([expect.objectContaining({ role: "assistant", stopReason: "stop" })]);
		expect(steeringQueue).toEqual([expect.objectContaining({ role: "user", content: "queued steer" })]);
	});
});

describe("agentLoopContinue with AgentMessage", () => {
	it("should throw when context has no messages", () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		expect(() => agentLoopContinue(context, config)).toThrow("Cannot continue: no messages in context");
	});

	it("should continue from existing context without emitting user message events", async () => {
		const userMessage: AgentMessage = createUserMessage("Hello");

		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [userMessage],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoopContinue(context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();

		// Should only return the new assistant message (not the existing user message)
		expect(messages.length).toBe(1);
		expect(messages[0].role).toBe("assistant");

		// Should NOT have user message events (that's the key difference from agentLoop)
		const messageEndEvents = events.filter((e) => e.type === "message_end");
		expect(messageEndEvents.length).toBe(1);
		expect((messageEndEvents[0] as any).message.role).toBe("assistant");
	});

	it("should allow custom message types as last message (caller responsibility)", async () => {
		// Custom message that will be converted to user message by convertToLlm
		interface CustomMessage {
			role: "custom";
			text: string;
			timestamp: number;
		}

		const customMessage: CustomMessage = {
			role: "custom",
			text: "Hook content",
			timestamp: Date.now(),
		};

		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [customMessage as unknown as AgentMessage],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				// Convert custom to user message
				return messages
					.map((m) => {
						if ((m as any).role === "custom") {
							return {
								role: "user" as const,
								content: (m as any).text,
								timestamp: m.timestamp,
							};
						}
						return m;
					})
					.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
			},
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response to custom message" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		// Should not throw - the custom message will be converted to user message
		const stream = agentLoopContinue(context, config, undefined, streamFn);

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(messages.length).toBe(1);
		expect(messages[0].role).toBe("assistant");
	});
});
