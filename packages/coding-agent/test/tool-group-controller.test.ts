import { Container, type Terminal, Text, TUI } from "@mariozechner/pi-tui";
import { beforeAll, describe, expect, it, vi } from "vitest";
import type { ToolGroupDefinition } from "../src/core/extensions/types.js";
import { ToolExecutionComponent } from "../src/modes/interactive/components/tool-execution.js";
import { ToolGroupComponent } from "../src/modes/interactive/components/tool-group.js";
import { initTheme } from "../src/modes/interactive/theme/theme.js";
import { ToolGroupController, type ToolGroupControllerState } from "../src/modes/interactive/tool-group-controller.js";

class FakeTerminal implements Terminal {
	columns = 80;
	rows = 24;
	kittyProtocolActive = true;
	writes: string[] = [];
	start(): void {}
	stop(): void {}
	async drainInput(): Promise<void> {}
	write(data: string): void {
		this.writes.push(data);
	}
	moveBy(_lines: number): void {}
	hideCursor(): void {}
	showCursor(): void {}
	clearLine(): void {}
	clearFromCursor(): void {}
	clearScreen(): void {}
	setTitle(_title: string): void {}
}

function createGroupDef(
	name: string,
	matchFn: (toolName: string) => boolean,
	lifecycle?: ToolGroupDefinition["lifecycle"],
): ToolGroupDefinition {
	return {
		name,
		match: (toolName) => matchFn(toolName),
		render: (members) => new Text(`${name}: ${members.length}`, 0, 0),
		...(lifecycle !== undefined ? { lifecycle } : {}),
	};
}

function createController(definitions: ToolGroupDefinition[] = []) {
	const ui = new TUI(new FakeTerminal());
	const chatContainer = new Container();
	const state: ToolGroupControllerState = {
		pendingTools: new Map(),
		openGroup: null,
		lastProcessedContentIndex: 0,
		emptyTextBlockIndices: new Set(),
	};

	const controller = new ToolGroupController(
		{
			ui,
			chatContainer,
			getShowImages: () => true,
			getToolOutputExpanded: () => false,
			getCwd: () => process.cwd(),
			getRegisteredToolDefinition: () => undefined,
			findMatchingGroupDefinition: (toolName, args) => {
				for (const definition of definitions) {
					try {
						if (definition.match(toolName, args)) {
							return definition;
						}
					} catch {
						// Ignore in tests; controller logs and closes for continuation errors.
					}
				}
				return undefined;
			},
		},
		state,
	);

	return { controller, state, chatContainer };
}

describe("ToolGroupController", () => {
	beforeAll(() => {
		initTheme("dark");
	});

	it("returns matching open toolRun group and closes it when continuation no longer matches", () => {
		const readGroup = createGroupDef("read-group", (toolName) => toolName === "read", { scope: "toolRun" });
		const { controller, state } = createController([readGroup]);

		controller.routeToolCallToPendingComponent("read", "tc1", { path: "a.ts" });
		expect(state.openGroup).not.toBeNull();

		const continuedGroup = controller.tryResolveOpenToolRunGroupForContinuation("read", { path: "b.ts" }, "test");
		expect(continuedGroup).toBe(state.openGroup);

		const nonMatching = controller.tryResolveOpenToolRunGroupForContinuation("bash", { command: "ls" }, "test");
		expect(nonMatching).toBeUndefined();
		expect(state.openGroup).toBeNull();
	});

	it("routes matching toolRun calls into the same group component", () => {
		const readGroup = createGroupDef("read-group", (toolName) => toolName === "read", { scope: "toolRun" });
		const { controller, state } = createController([readGroup]);

		controller.routeToolCallToPendingComponent("read", "tc1", { path: "a.ts" });
		const firstGroup = state.openGroup?.component;
		expect(firstGroup).toBeInstanceOf(ToolGroupComponent);

		controller.routeToolCallToPendingComponent("read", "tc2", { path: "b.ts" });
		expect(state.pendingTools.get("tc1")).toBe(firstGroup);
		expect(state.pendingTools.get("tc2")).toBe(firstGroup);
		expect(state.openGroup?.component).toBe(firstGroup);
	});

	it("reconciles processed empty text blocks by closing toolRun groups when text becomes meaningful", () => {
		const readGroup = createGroupDef("read-group", (toolName) => toolName === "read", { scope: "toolRun" });
		const { controller, state } = createController([readGroup]);

		controller.routeToolCallToPendingComponent("read", "tc1", { path: "a.ts" });
		expect(state.openGroup).not.toBeNull();

		controller.processNewStreamingBlock({ type: "text", text: "" }, 1);
		expect(state.openGroup).not.toBeNull();

		controller.reconcileProcessedStreamingBlock({ type: "text", text: "Now speaking" }, 1);
		expect(state.openGroup).toBeNull();
	});

	it("finalizes pending args for grouped and ungrouped tool components", () => {
		const readGroup = createGroupDef("read-group", (toolName) => toolName === "read");
		const { controller, state } = createController([readGroup]);

		controller.routeToolCallToPendingComponent("read", "tc1", { path: "a.ts" });
		controller.routeToolCallToPendingComponent("bash", "tc2", { command: "ls" });

		const grouped = state.pendingTools.get("tc1");
		const individual = state.pendingTools.get("tc2");
		expect(grouped).toBeInstanceOf(ToolGroupComponent);
		expect(individual).toBeInstanceOf(ToolExecutionComponent);

		const groupedSpy = vi.spyOn(grouped as ToolGroupComponent, "setMemberArgsComplete");
		const individualSpy = vi.spyOn(individual as ToolExecutionComponent, "setArgsComplete");

		controller.finalizePendingToolArgs();

		expect(groupedSpy).toHaveBeenCalledWith("tc1");
		expect(individualSpy).toHaveBeenCalledTimes(1);
	});

	it("shares replay routing rules and collects per-call replay errors", () => {
		const readGroup = createGroupDef("read-group", (toolName) => toolName === "read", { scope: "toolRun" });
		const { controller, state } = createController([readGroup]);
		const pendingToolErrors = new Map<string, string>();

		controller.onReplayAssistantMessage(
			{
				content: [{ type: "toolCall", id: "tc1", name: "read", arguments: { path: "a.ts" } }],
				stopReason: "toolUse",
			},
			0,
			pendingToolErrors,
		);

		const firstGroup = state.openGroup?.component;
		expect(firstGroup).toBeInstanceOf(ToolGroupComponent);

		controller.onReplayAssistantMessage(
			{
				content: [
					{ type: "toolCall", id: "tc2", name: "read", arguments: { path: "b.ts" } },
					{ type: "text", text: "analysis" },
				],
				stopReason: "toolUse",
			},
			0,
			pendingToolErrors,
		);

		expect(state.pendingTools.get("tc2")).toBe(firstGroup);
		expect(state.openGroup).toBeNull();

		controller.onReplayAssistantMessage(
			{
				content: [{ type: "toolCall", id: "tc3", name: "read", arguments: { path: "c.ts" } }],
				stopReason: "aborted",
			},
			0,
			pendingToolErrors,
		);
		expect(pendingToolErrors.get("tc3")).toBe("Operation aborted");
	});
});
