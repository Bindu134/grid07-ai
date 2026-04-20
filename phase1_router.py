# phase1_router.py
# Phase 1: Vector-Based Persona Matching (The Router)
#
# Builds an in-memory FAISS vector store of bot persona embeddings.
# When a new post arrives, it is embedded and compared against each
# persona using cosine similarity. Only bots above the threshold are returned.
#
# Run: python phase1_router.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from personas import PERSONAS
from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# 1. Load embedding model (runs locally, no API key needed)
# ---------------------------------------------------------------------------
print(f"[Phase 1] Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# 2. Build in-memory FAISS vector store from persona descriptions
#    IndexFlatIP = Inner Product index.
#    When vectors are L2-normalised, inner product == cosine similarity.
# ---------------------------------------------------------------------------
persona_ids   = list(PERSONAS.keys())                              # ["Bot_A", "Bot_B", "Bot_C"]
persona_texts = [PERSONAS[pid]["description"] for pid in persona_ids]

print("[Phase 1] Generating persona embeddings...")
persona_embeddings = embedder.encode(
    persona_texts,
    normalize_embeddings=True,   # Required for cosine similarity via inner product
    show_progress_bar=False,
).astype(np.float32)

dimension = persona_embeddings.shape[1]                           # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatIP(dimension)
index.add(persona_embeddings)

print(f"[Phase 1] Vector store ready — {index.ntotal} personas indexed.\n")


# ---------------------------------------------------------------------------
# 3. Router function (exact signature from assignment spec)
# ---------------------------------------------------------------------------
def route_post_to_bots(post_content: str, threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """
    Embeds a post and returns all bots whose persona cosine similarity
    exceeds the given threshold.

    NOTE on threshold: The spec suggests 0.85, but all-MiniLM-L6-v2 produces
    cross-topic cosine scores in the 0.2–0.55 range. Default is tuned to 0.3
    for realistic routing with this embedding model.

    Args:
        post_content: The text of the incoming post.
        threshold:    Minimum cosine similarity score for a bot to be matched.

    Returns:
        List of dicts sorted by score desc: [{"bot_id": ..., "score": ...}]
    """
    # Embed and normalise the incoming post
    post_vec = embedder.encode(
        [post_content],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Query the FAISS index for all persona scores
    scores, indices = index.search(post_vec, len(persona_ids))

    # Filter by threshold and build result list
    matched = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            matched.append({
                "bot_id":      persona_ids[idx],
                "persona_name": PERSONAS[persona_ids[idx]]["name"],
                "score":       round(float(score), 4),
            })

    # Sort by descending similarity score
    matched.sort(key=lambda x: -x["score"])
    return matched


# ---------------------------------------------------------------------------
# 4. Demo — run with the example post from the assignment spec
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEST_POSTS = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin and Ethereum are mooning. Best time to buy the dip.",
        "We need stronger antitrust laws to break up tech monopolies.",
        "Fed raises interest rates again. Bond yields spike. Watch the yield curve.",
    ]

    for post in TEST_POSTS:
        print(f"POST : {post}")
        results = route_post_to_bots(post)

        if results:
            for r in results:
                print(f"  → {r['bot_id']} ({r['persona_name']}) matched — score: {r['score']}")
        else:
            print("  → No bots matched above threshold.")
        print()
