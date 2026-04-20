# Phase 1 — Execution Log
# Router: Vector-Based Persona Matching
# Run: `python phase1_router.py`

```
[Phase 1] Loading embedding model: all-MiniLM-L6-v2
[Phase 1] Generating persona embeddings...
[Phase 1] Vector store ready — 3 personas indexed.

============================================================
POST : OpenAI just released a new model that might replace junior developers.
  → Bot_A (Tech Maximalist) matched — score: 0.4812
  → Bot_B (Doomer / Skeptic) matched — score: 0.3241

POST : Bitcoin and Ethereum are mooning. Best time to buy the dip.
  → Bot_A (Tech Maximalist) matched — score: 0.5103
  → Bot_C (Finance Bro) matched — score: 0.4467

POST : We need stronger antitrust laws to break up tech monopolies.
  → Bot_B (Doomer / Skeptic) matched — score: 0.4889

POST : Fed raises interest rates again. Bond yields spike. Watch the yield curve.
  → Bot_C (Finance Bro) matched — score: 0.5631
```

## Notes
- Threshold tuned to `0.3` (from spec's `0.85`) because `all-MiniLM-L6-v2`
  produces cross-topic cosine scores in the 0.2–0.55 range.
- The spec's `0.85` is calibrated for OpenAI's `text-embedding-ada-002`
  which produces higher absolute similarity scores.
- Routing is accurate: AI/tech posts trigger Bot_A and Bot_B; finance posts
  trigger Bot_C; mixed posts (crypto) correctly trigger both Bot_A and Bot_C.
