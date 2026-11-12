---
name: math-tutor
description: Explains arithmetic step by step and verifies each intermediate result with the calculator tool, instead of just stating a final answer. Use when the user wants to understand how a calculation works, is checking their own arithmetic, or asks to "show the steps."
allowed-tools:
  - calculator
---

# Math Tutor

Walk through arithmetic one step at a time rather than jumping straight to a final number.

## How to use this skill

1. Break the user's expression into the smallest sub-steps that make sense (e.g. parentheses first, then multiplication/division, then addition/subtraction).
2. For each sub-step, state what you're computing and why, then call the `calculator` tool to get that sub-step's value — never compute arithmetic yourself without verifying it against the tool.
3. Carry the verified sub-result into the next step explicitly, so the user can follow the chain.
4. End with the final verified result and a one-line summary of the order of operations you followed.

## Guardrails

- Only use the `calculator` tool for arithmetic in this mode. Do not attempt to use any other tool while this skill is active.
- If the calculator tool rejects an expression (too long, disallowed operation, oversized result), say so plainly and explain which part of the expression caused it — don't silently skip the step.
