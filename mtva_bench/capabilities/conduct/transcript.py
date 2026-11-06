"""The call as the judge reads it: agent turns numbered from 1, tool calls and replies inline."""

from mtva_bench.simulate.call import OPENING_TURN
from mtva_bench.simulate.types import Conversation

ENDINGS = {
    "terminal_tool": "The call ended when the agent called {tool}.",
    "caller_done": "The caller was done and the simulation stopped; the agent had not hung up.",
    "max_turns": "The call reached the turn limit without ending.",
    "context_limit": "The call stopped because it no longer fit in the agent's context window.",
}


def transcript(conversation: Conversation) -> tuple[str, int]:
    """The call as numbered text, and how many agent turns it has. Agent [n] is the agent's
    n-th turn: what it said and the tool calls it made before the caller spoke again."""
    lines: list[str] = []
    names: dict[str, str] = {}
    turn = 0
    in_turn = False
    messages = conversation.messages[1:]
    if messages and messages[0].role == "user" and messages[0].content == OPENING_TURN:
        messages = messages[1:]
    for message in messages:
        if message.role == "user":
            lines.append(f"Caller: {message.content or ''}")
            in_turn = False
        elif message.role == "assistant":
            if not in_turn:
                turn += 1
                lines.append(f"Agent [{turn}]:")
                in_turn = True
            if message.content:
                lines.append(f"  says: {message.content.strip()}")
            for call in message.tool_calls:
                names[call.id] = call.name
                lines.append(f"  → {call.name} {call.arguments or '{}'}")
        elif message.role == "tool":
            name = names.get(message.tool_call_id or "", "tool")
            lines.append(f"  ← {name}: {message.content or ''}")
    for index, line in enumerate(lines):
        if line.startswith("Agent [") and (
            index + 1 == len(lines) or not lines[index + 1].startswith("  ")
        ):
            lines[index] = line + " (said nothing)"
    return "\n".join(lines), turn


def ending(conversation: Conversation) -> str:
    terminal = [event.name for event in conversation.tool_events if event.terminal]
    return ENDINGS[conversation.end_reason].format(tool=terminal[-1] if terminal else "a tool")
