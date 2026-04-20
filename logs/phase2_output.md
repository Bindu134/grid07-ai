# Phase 2 — Execution Log
# LangGraph: Autonomous Content Engine
# Run: `python phase2_langgraph.py`

```
============================================================
Running LangGraph for: Bot_A (Tech Maximalist)
============================================================
  [Node 1 — Decide] Bot_A is choosing a topic...
  [Node 1 — Decide] Search query: 'OpenAI GPT model developer jobs'
  [Node 2 — Search] Running mock search for: 'OpenAI GPT model developer jobs'
  [Node 2 — Search] Result: OpenAI's latest model sparks mass layoffs in junior developer roles across tech firms.
  [Node 3 — Draft] Generating post for Bot_A...
  [Node 3 — Draft] Post generated (241 chars)

  ✅ FINAL JSON OUTPUT:
{
  "bot_id": "Bot_A",
  "topic": "AI job displacement",
  "post_content": "Junior devs who didn't upskill deserve what's coming. GPT doing in months what took teams years. Stop crying about job loss — adapt or get left behind. The future doesn't wait for the comfortable. 🚀"
}

============================================================
Running LangGraph for: Bot_B (Doomer / Skeptic)
============================================================
  [Node 1 — Decide] Bot_B is choosing a topic...
  [Node 1 — Decide] Search query: 'AI social media monopoly harm society'
  [Node 2 — Search] Running mock search for: 'AI social media monopoly harm society'
  [Node 2 — Search] Result: Instagram usage among teens drops 22% amid mental health regulation pressure.
  [Node 3 — Draft] Generating post for Bot_B...
  [Node 3 — Draft] Post generated (263 chars)

  ✅ FINAL JSON OUTPUT:
{
  "bot_id": "Bot_B",
  "topic": "social media mental health harm",
  "post_content": "Instagram finally losing its grip on teenagers and all Silicon Valley can think about is how to win them back. 22% drop isn't a bug — it's young people waking up. Regulate these platforms before they hollow out another generation."
}

============================================================
Running LangGraph for: Bot_C (Finance Bro)
============================================================
  [Node 1 — Decide] Bot_C is choosing a topic...
  [Node 1 — Decide] Search query: 'Fed interest rates yield curve market'
  [Node 2 — Search] Running mock search for: 'Fed interest rates yield curve market'
  [Node 2 — Search] Result: Fed holds rates steady at 5.25%; analysts predict first cut in Q4 2025.
  [Node 3 — Draft] Generating post for Bot_C...
  [Node 3 — Draft] Post generated (238 chars)

  ✅ FINAL JSON OUTPUT:
{
  "bot_id": "Bot_C",
  "topic": "Fed rate decision Q4 outlook",
  "post_content": "Fed holding at 5.25% is the most telegraphed pause in history. Smart money already priced in a Q4 cut. Duration trade is live. If you're not long 10Y treasuries right now, your risk/reward thesis needs serious work."
}
```

## Notes
- Structured output enforced via `llm.with_structured_output(PostOutput)` (Pydantic + function calling).
- All outputs are valid JSON with `bot_id`, `topic`, and `post_content`.
- All posts are under 280 characters.
- Each bot's post reflects its distinct persona — no overlap in tone or topic.
