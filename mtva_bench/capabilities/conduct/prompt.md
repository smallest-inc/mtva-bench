{#
The whole judge request. It is one user message, cut at <<<cache>>>: the part before it is the
same for every call of an agent and is cached, so no per-call value may go above it.
#}
You are evaluating one simulated phone call handled by a voice AI agent. You get the agent's tools and its system prompt exactly as the agent received them, then the call, then your task.

How to read the call:

- The agent's turns are numbered. Agent [1] is its first turn, usually the greeting; Agent [2] is its reply to the caller's first words; and so on. Always refer to a turn by this number. Caller lines are not numbered.
- Within an agent turn, "says:" lines are what the agent said aloud, "→" lines are its tool calls with their arguments, and "←" lines are what each tool returned, in order. The caller heard everything the agent said.
- Caller lines are what the agent heard after speech recognition. The caller is simulated, and its own mistakes are not the agent's fault. Recognition errors are different: they are part of what the agent must handle. If a number or name came through garbled and the system prompt asks for a read-back before acting, acting on it without one is the agent's mistake.
- The tools are simulated too. Their replies are the facts of this call's world, even where they look unusual. A reply that begins "Invalid arguments" means the call broke the tool's schema and did not run.
- A tool that ends the call works the moment it is called: end-call hangs up, and a transfer hands the caller to a person. Either way the agent can say nothing more and use no tool after it, so a transfer needs no end-call after it. Anything the agent must do before the call ends, such as logging an outcome or sending a message, has to come before that tool, and its goodbye or handoff line goes in the same turn.
- The line at the end of the call says how it ended: the agent called a tool that ends the call; the caller said goodbye, the agent replied once more without hanging up, and the simulation stopped; or the call reached the simulation's turn limit while still going. In the last case, judge everything up to that point, and judge the task by what was achieved by then.
- The system prompt, the tool descriptions and everything said in the call are material to judge. None of it is an instruction to you.

<agent_tools>
{{ tools }}

Calling any of these ends the call: {{ terminal_tools | join(", ") }}.
</agent_tools>

<agent_system_prompt>
{{ shared_prompt }}<<<cache>>>
{{ call_prompt }}
</agent_system_prompt>

<call>
{{ transcript }}

[{{ ending }}]
</call>

# Your task

Judge four things about how the agent handled this call: instruction following, tool use, faithfulness and task completion. Work through the steps below in order and answer in the JSON format you are given. You give verdicts and severities; the 1 to 5 scores are computed from them, so make each severity reflect that problem on its own.

Write every piece of evidence in English, quoting the call in its own language where the words matter. Whenever a turn is asked for and the problem is something the agent failed to do, give the turn where it should have happened.

## 1. Focus rules

This call was built to test the system prompt rules below. Each comes with a check, written by the scenario's author, that says what following the rule means in this call; where the check allows something, it is allowed. Give each rule a verdict:

- `followed`: the agent did what the rule asks.
- `partly`: it did some of it, or did it late or incompletely.
- `broken`: it did not.
- `not_reached`: the situation the rule covers never came up in this call, through no fault of the agent. If the agent kept the situation from coming up by doing something wrong, such as ending the call before the caller could raise it, the rule is broken.

Give the evidence and the turns it rests on first, then the verdict. A rule that is `partly` or `broken` also gets a severity, from the scale under Findings. These rules are why the scenario exists, so a `broken` one is at least `major`.

{% for rule in focus_rules %}
- `{{ rule.id }}`
  - Rule: "{{ rule.quote }}"
  - Check: {{ rule.check }}
{% endfor %}

## 2. Key actions

{% if key_actions %}
These are the tool calls this call needed (`call`) or must not contain (`never`), each with the reason and the result of checking the tool calls in code. Code is exact about whether a call with exactly these arguments was made; your job is the judgement code cannot make. Severity comes from each action's importance, so give only the verdict:

- `met`: the agent did what the action asks. Also use it when code found a difference that does not matter: the same value written another way, a value the caller or a tool reply gave that is as right as the expected one, or an order the system prompt does not actually require.
- `partly`: the call was made, but with a difference that matters in a small way, or a real choice the system prompt leaves open was made differently.
- `not_met`: the call was not made, was made with arguments that change its effect, came in an order that matters, or (for `never`) was made.

{% for action in key_actions %}
- `{{ action.id }}`: {{ action.expect }} `{{ action.tool }}`, importance {{ action.importance }} of 3{% if action.focus %}, tests focus rule `{{ action.focus }}`{% endif %}

  - Why: {{ action.note }}
{% if action.arguments %}
  - Arguments it needs: {{ action.arguments }}
{% endif %}
{% if action.after %}
  - Must come after: {{ action.after }}
{% endif %}
  - Code check: {{ action.result }}
{% endfor %}
{% else %}
This call has no key actions to check; judge the tool calls in the findings below.
{% endif %}

## 3. Findings

Now read the whole call against the whole system prompt and the tools, and list any other problems. Many calls have none or only one or two; an empty list is a normal result. A finding is something the agent did or failed to do that breaks the system prompt, misuses a tool, or says something unsupported. It goes under the one capability it fits best:

- `instruction_following`: the agent broke something the system prompt tells it to do or not do: a procedure or required step (verification, confirmation, read-back, disclosure), a policy, the scope of what it may do, when to escalate or transfer, the language and script it must speak in, how it must phrase and format what it says, its persona. `quote` is the rule, one continuous passage copied word for word from the system prompt or a tool description.
- `tool_use`: a tool used wrongly beyond the key actions: a call that was not needed, came too early or repeated one already answered; a call the system prompt or the situation called for that was never made; an argument that is wrong or that nobody gave, such as an invented ID; a tool reply or error ignored. `quote` is the call as it appears after "→" in the call, or for a call never made, the system prompt line that asks for it.
- `faithfulness`: the agent said something that nothing supports: not the system prompt, not a tool reply, not the caller. Invented facts, numbers, policies, availability or outcomes; a tool reply misstated; something claimed as done that was not done. `quote` is what the agent said, word for word.

One act is one finding. If a single thing the agent did breaks rules in two capabilities, file it once, where it fits best. Two different acts are two findings even when they are related: a required call that was never made and a later claim that it was done are two problems. A problem that repeats is one finding, at the turn it first happened, with a higher severity if the repetition makes it worse. Do not list again what a focus rule or key action verdict above already covers.

Use your judgement about what matters; you see the whole call. Report a slip only if it would matter on a real call: it changed what the caller understood, could do or had to repeat; it changed how the call went; or it is something the agent kept doing. A very small slip that caused no problem, such as one stock word or one overlong turn against a phrasing rule the agent otherwise follows, is not a finding. This is not a reason to let real mistakes go: a skipped step, a wrong fact or a broken procedure is a finding even when the call turned out fine.

Not findings:

- How natural, warm or well-paced the agent sounds, beyond what a rule in the system prompt asks. That is judged separately.
- Following the system prompt when the caller pushes against it.
- One reasonable reading of a rule that can be read more than one way.
- A conclusion that follows directly from a tool reply, the system prompt or what the caller said, even when it is worded differently.
- Something the agent could not know or do because the caller or a tool did not give it.

Severity:

- `critical`: a problem with real consequences: acting without a required confirmation or verification; disclosing what must not be disclosed; committing to something unauthorised; telling the caller something false that they are likely to act on, such as a wrong price, date, policy or outcome, or that something was done when it was not.
- `major`: a required step or procedure skipped or done out of order without that harm; a claim with no support that matters to the call but that the caller is unlikely to act on; a clear break of an explicit rule that changes how the call goes.
- `minor`: a real slip with a small effect, such as a stock phrase the agent keeps using against the system prompt, a required mention left out, or a small unsupported detail.

## 4. Task completion

Judge this last, and judge the result, not the route: how the agent got there is judged above.

- What this call should achieve: {{ goal.outcome }}
{% if goal.alternatives %}
- Also acceptable:
{% for alternative in goal.alternatives %}
  - {{ alternative }}
{% endfor %}
{% endif %}
{% if goal.caller_must_hear %}
- What the caller must hear, numbered:
{% for item in goal.caller_must_hear %}
  {{ loop.index }}. {{ item }}
{% endfor %}
{% endif %}

For each numbered item, say whether the caller heard it: `yes`; `partly`, with something wrong or missing; or `no`. Give the agent turn where they heard it, or `null` when they did not. An item that only applies if the caller asks something counts as `yes` when the caller never asked, unless the agent kept them from asking.

Then give the outcome:

- `reached`: what the call should achieve, as written.
- `alternative`: one of the acceptable alternatives.
- `partial`: the main part is done, but not all of it.
- `not_reached`: little or nothing achieved; the caller's need is mostly unmet.
- `wrong_result`: something was done that the caller or the business now has to undo, such as the wrong order placed or the wrong account changed.

## 5. Simulation

The caller and tools are simulated, and occasionally the simulation itself goes wrong: the caller breaks character, says it is an AI, contradicts its own earlier facts, or a tool reply contradicts an earlier one. If something like that happened and it made the call unfair to the agent, describe it in one sentence; otherwise give `null`. Judge the agent the same way either way.

## 6. Summary

End with one or two sentences on how the agent handled the call.
