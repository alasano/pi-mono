import type { AssistantMessage } from "@mariozechner/pi-ai";
import type { Container, TUI } from "@mariozechner/pi-tui";
import type { ToolDefinition, ToolGroupDefinition } from "../../core/extensions/index.js";
import { ToolExecutionComponent } from "./components/tool-execution.js";
import { ToolGroupComponent } from "./components/tool-group.js";

type AssistantContent = Exclude<AssistantMessage["content"], string>;
type AssistantContentBlock = AssistantContent[number];
type PendingToolComponent = ToolExecutionComponent | ToolGroupComponent;

export type PendingToolResult = {
	content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
	details?: unknown;
	isError: boolean;
};

export type ToolGroupState = {
	component: ToolGroupComponent;
	definition: ToolGroupDefinition;
};

export type ToolGroupControllerState = {
	pendingTools: Map<string, PendingToolComponent>;
	openGroup: ToolGroupState | null;
	lastProcessedContentIndex: number;
	emptyTextBlockIndices: Set<number>;
};

type ReplayAssistantMessage = {
	content: AssistantContent;
	stopReason?: string;
	errorMessage?: string;
};

export type ToolGroupControllerDependencies = {
	ui: TUI;
	chatContainer: Container;
	getShowImages: () => boolean;
	getToolOutputExpanded: () => boolean;
	getCwd: () => string;
	getRegisteredToolDefinition: (toolName: string) => ToolDefinition | undefined;
	findMatchingGroupDefinition: (toolName: string, args: unknown) => ToolGroupDefinition | undefined;
};

export class ToolGroupController {
	constructor(
		private readonly deps: ToolGroupControllerDependencies,
		private readonly state: ToolGroupControllerState,
	) {}

	get pendingTools(): Map<string, PendingToolComponent> {
		return this.state.pendingTools;
	}

	get openGroup(): ToolGroupState | null {
		return this.state.openGroup;
	}

	getToolGroupScope(definition: ToolGroupDefinition): "message" | "toolRun" {
		return definition.lifecycle?.scope ?? "message";
	}

	closeOpenGroup(): void {
		if (!this.state.openGroup) return;
		this.state.openGroup.component.close();
		this.state.openGroup = null;
	}

	closeMessageScopedOpenGroup(): void {
		if (this.state.openGroup && this.getToolGroupScope(this.state.openGroup.definition) === "message") {
			this.closeOpenGroup();
		}
	}

	tryResolveOpenToolRunGroupForContinuation(
		toolName: string,
		args: unknown,
		context: string,
	): ToolGroupState | undefined {
		const group = this.state.openGroup;
		if (!group) {
			return undefined;
		}

		if (this.getToolGroupScope(group.definition) !== "toolRun") {
			return undefined;
		}

		try {
			if (group.definition.match(toolName, args)) {
				return group;
			}
		} catch (error) {
			console.warn(`Tool group '${group.definition.name}' match() threw during ${context}:`, error);
		}

		this.closeOpenGroup();
		return undefined;
	}

	resetStreamingContentIndex(): void {
		this.state.lastProcessedContentIndex = 0;
		this.state.emptyTextBlockIndices.clear();
	}

	resetStreamingToolTracking(): void {
		this.closeOpenGroup();
		this.resetStreamingContentIndex();
	}

	isGroupBoundary(block: AssistantContentBlock): boolean {
		return block.type === "thinking" || (block.type === "text" && !!block.text?.trim());
	}

	private createToolExecutionComponent(toolName: string, toolCallId: string, args: unknown): ToolExecutionComponent {
		const component = new ToolExecutionComponent(
			toolName,
			toolCallId,
			args,
			{
				showImages: this.deps.getShowImages(),
			},
			this.deps.getRegisteredToolDefinition(toolName),
			this.deps.ui,
			this.deps.getCwd(),
		);
		component.setExpanded(this.deps.getToolOutputExpanded());
		this.deps.chatContainer.addChild(component);
		return component;
	}

	private createOrExtendGroup(
		toolName: string,
		toolCallId: string,
		args: unknown,
		definition: ToolGroupDefinition,
	): ToolGroupComponent {
		if (this.state.openGroup?.definition === definition) {
			this.state.openGroup.component.addMember(toolCallId, toolName, args);
			return this.state.openGroup.component;
		}

		this.closeOpenGroup();

		const groupComponent = new ToolGroupComponent(this.deps.ui, definition, {
			showImages: this.deps.getShowImages(),
		});
		groupComponent.setExpanded(this.deps.getToolOutputExpanded());
		groupComponent.addMember(toolCallId, toolName, args);
		this.deps.chatContainer.addChild(groupComponent);
		this.state.openGroup = { component: groupComponent, definition };
		return groupComponent;
	}

	routeToolCallToPendingComponent(
		toolName: string,
		toolCallId: string,
		args: unknown,
		context: string = "continuation",
	): PendingToolComponent {
		const existingComponent = this.state.pendingTools.get(toolCallId);
		if (existingComponent) {
			return existingComponent;
		}

		const continuedGroup = this.tryResolveOpenToolRunGroupForContinuation(toolName, args, context);
		if (continuedGroup) {
			continuedGroup.component.addMember(toolCallId, toolName, args);
			this.state.pendingTools.set(toolCallId, continuedGroup.component);
			return continuedGroup.component;
		}

		const groupDefinition = this.deps.findMatchingGroupDefinition(toolName, args);
		if (!groupDefinition) {
			this.closeOpenGroup();
		}

		const component = groupDefinition
			? this.createOrExtendGroup(toolName, toolCallId, args, groupDefinition)
			: this.createToolExecutionComponent(toolName, toolCallId, args);
		this.state.pendingTools.set(toolCallId, component);
		return component;
	}

	updatePendingToolArgs(toolCallId: string, args: unknown): void {
		const component = this.state.pendingTools.get(toolCallId);
		if (!component) return;
		if (component instanceof ToolGroupComponent) {
			component.updateMemberArgs(toolCallId, args);
		} else {
			component.updateArgs(args);
		}
	}

	markPendingToolExecutionStarted(toolCallId: string): void {
		const component = this.state.pendingTools.get(toolCallId);
		if (!component) return;
		if (component instanceof ToolGroupComponent) {
			component.markMemberExecutionStarted(toolCallId);
		} else {
			component.markExecutionStarted();
		}
	}

	markPendingToolArgsComplete(toolCallId: string): void {
		const component = this.state.pendingTools.get(toolCallId);
		if (!component) return;
		if (component instanceof ToolGroupComponent) {
			component.setMemberArgsComplete(toolCallId);
		} else {
			component.setArgsComplete();
		}
	}

	updatePendingToolResult(toolCallId: string, result: PendingToolResult, isPartial: boolean): void {
		const component = this.state.pendingTools.get(toolCallId);
		if (!component) return;
		if (component instanceof ToolGroupComponent) {
			component.updateMemberResult(toolCallId, result, isPartial);
		} else {
			component.updateResult(result, isPartial);
		}
	}

	private partitionPendingToolTargets(): {
		individual: Array<{ toolCallId: string; component: ToolExecutionComponent }>;
		grouped: Map<ToolGroupComponent, string[]>;
	} {
		const individual: Array<{ toolCallId: string; component: ToolExecutionComponent }> = [];
		const grouped = new Map<ToolGroupComponent, string[]>();

		for (const [toolCallId, component] of this.state.pendingTools.entries()) {
			if (component instanceof ToolGroupComponent) {
				const ids = grouped.get(component) ?? [];
				ids.push(toolCallId);
				grouped.set(component, ids);
				continue;
			}
			individual.push({ toolCallId, component });
		}

		return { individual, grouped };
	}

	setPendingToolsErrorResult(result: PendingToolResult): void {
		const batches = this.partitionPendingToolTargets();
		for (const { component } of batches.individual) {
			component.updateResult(result);
		}
		for (const [groupComponent, toolCallIds] of batches.grouped) {
			groupComponent.batchUpdate(() => {
				for (const toolCallId of toolCallIds) {
					groupComponent.updateMemberResult(toolCallId, result, false);
				}
			});
		}
	}

	injectPendingToolErrors(getErrorMessage: (toolCallId: string) => string): void {
		const batches = this.partitionPendingToolTargets();
		for (const { toolCallId, component } of batches.individual) {
			component.updateResult({
				content: [{ type: "text", text: getErrorMessage(toolCallId) }],
				isError: true,
			});
		}
		for (const [groupComponent, toolCallIds] of batches.grouped) {
			groupComponent.batchUpdate(() => {
				for (const toolCallId of toolCallIds) {
					groupComponent.updateMemberResult(
						toolCallId,
						{ content: [{ type: "text", text: getErrorMessage(toolCallId) }], isError: true },
						false,
					);
				}
			});
		}
	}

	finalizePendingToolArgs(): void {
		const batches = this.partitionPendingToolTargets();
		for (const { component } of batches.individual) {
			component.setArgsComplete();
		}
		for (const [groupComponent, toolCallIds] of batches.grouped) {
			groupComponent.batchUpdate(() => {
				for (const toolCallId of toolCallIds) {
					groupComponent.setMemberArgsComplete(toolCallId);
				}
			});
		}
	}

	reconcileProcessedStreamingBlock(block: AssistantContentBlock, index: number): void {
		if (block.type === "toolCall") {
			this.updatePendingToolArgs(block.id, block.arguments);
		} else if (
			this.state.openGroup &&
			this.getToolGroupScope(this.state.openGroup.definition) === "toolRun" &&
			block.type === "text" &&
			this.state.emptyTextBlockIndices.has(index) &&
			block.text?.trim()
		) {
			this.state.emptyTextBlockIndices.delete(index);
			this.closeOpenGroup();
		}
	}

	processNewStreamingBlock(block: AssistantContentBlock, index: number): void {
		if (block.type === "toolCall") {
			if (this.state.pendingTools.has(block.id)) {
				this.updatePendingToolArgs(block.id, block.arguments);
			} else {
				this.routeToolCallToPendingComponent(block.name, block.id, block.arguments, "continuation");
			}
			return;
		}

		if (block.type === "text" && !block.text?.trim()) {
			this.state.emptyTextBlockIndices.add(index);
		}

		if (this.isGroupBoundary(block)) {
			this.closeOpenGroup();
		}
	}

	onAssistantMessageStart(): void {
		this.resetStreamingContentIndex();
		this.closeMessageScopedOpenGroup();
	}

	onStreamingContentUpdate(content: AssistantContent): void {
		const processedCount = Math.min(this.state.lastProcessedContentIndex, content.length);
		for (let i = 0; i < processedCount; i++) {
			this.reconcileProcessedStreamingBlock(content[i], i);
		}
		for (let i = processedCount; i < content.length; i++) {
			this.processNewStreamingBlock(content[i], i);
		}
		this.state.lastProcessedContentIndex = content.length;
	}

	onAssistantMessageEnd(options: { stopReason?: string; errorMessage?: string }): void {
		if (options.stopReason === "aborted" || options.stopReason === "error") {
			const errorResult: PendingToolResult = {
				content: [{ type: "text", text: options.errorMessage || "Error" }],
				isError: true,
			};
			this.setPendingToolsErrorResult(errorResult);
			this.clearPendingToolsAndGroup();
			return;
		}

		this.finalizePendingToolArgs();
		const openScope = this.state.openGroup ? this.getToolGroupScope(this.state.openGroup.definition) : "message";
		if (options.stopReason === "toolUse" && openScope === "toolRun") {
			this.resetStreamingContentIndex();
		} else {
			this.resetStreamingToolTracking();
		}
	}

	onToolExecutionStart(toolCallId: string, toolName: string, args: unknown): void {
		this.routeToolCallToPendingComponent(toolName, toolCallId, args, "continuation");
		this.markPendingToolExecutionStarted(toolCallId);
	}

	onToolExecutionUpdate(toolCallId: string, result: PendingToolResult): boolean {
		if (!this.state.pendingTools.has(toolCallId)) {
			return false;
		}
		this.updatePendingToolResult(toolCallId, result, true);
		return true;
	}

	onToolExecutionEnd(toolCallId: string, result: PendingToolResult): boolean {
		if (!this.state.pendingTools.has(toolCallId)) {
			return false;
		}
		this.updatePendingToolResult(toolCallId, result, false);
		this.state.pendingTools.delete(toolCallId);
		return true;
	}

	onReplayAssistantMessage(
		message: ReplayAssistantMessage,
		retryAttempt: number,
		pendingToolErrors: Map<string, string>,
	): void {
		const isAbortedOrError = message.stopReason === "aborted" || message.stopReason === "error";
		let errorMessage: string | undefined;
		if (isAbortedOrError) {
			if (message.stopReason === "aborted") {
				errorMessage =
					retryAttempt > 0
						? `Aborted after ${retryAttempt} retry attempt${retryAttempt > 1 ? "s" : ""}`
						: "Operation aborted";
			} else {
				errorMessage = message.errorMessage || "Error";
			}
		}

		for (const content of message.content) {
			if (content.type === "toolCall") {
				this.routeToolCallToPendingComponent(content.name, content.id, content.arguments, "replay continuation");
				this.markPendingToolArgsComplete(content.id);
				this.markPendingToolExecutionStarted(content.id);
				if (errorMessage) {
					pendingToolErrors.set(content.id, errorMessage);
				}
				continue;
			}

			if (this.isGroupBoundary(content)) {
				this.closeOpenGroup();
			}
		}

		if (this.state.openGroup) {
			const replayScope = this.getToolGroupScope(this.state.openGroup.definition);
			if (!(replayScope === "toolRun" && message.stopReason === "toolUse")) {
				this.closeOpenGroup();
			}
		}
	}

	onReplayToolResult(toolCallId: string, result: PendingToolResult): boolean {
		if (!this.state.pendingTools.has(toolCallId)) {
			return false;
		}

		this.updatePendingToolResult(toolCallId, result, false);
		this.state.pendingTools.delete(toolCallId);
		return true;
	}

	clearPendingToolsAndGroup(): void {
		this.resetStreamingToolTracking();
		this.state.pendingTools.clear();
	}
}
