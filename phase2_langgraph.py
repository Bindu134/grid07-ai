# phase2_langgraph.py
# Phase 2: The Autonomous Content Engine (LangGraph)
#
# Builds a 3-node LangGraph StateGraph:
#   Node 1 (decide)  → LLM picks a topic and formats a search query
#   Node 2 (search)  → Executes mock_searxng_search tool
#   Node 3 (draft)   → LLM drafts a 280-char opinionated post as JSON
#
# Structured output is guaranteed via Pydantic + .with_structured_output()
#
# Run: python phase2_langgraph.py

import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from personas import PERSONAS
from tools import mock_searxng_search
from config import GROQ_API_KEY, GROQ_MODEL


# ---------------------------------------------------------------------------
# 1. Initialise the LLM (Groq — free tier, fast inference)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.8,   # Higher temp for opinionated, varied posts
)


# ---------------------------------------------------------------------------
# 2. Structured output schema (guarantees exact JSON format from spec)
# ---------------------------------------------------------------------------
class PostOutput(BaseModel):
    """Strict JSON schema for the generated post. Enforced via function calling."""
    bot_id:       str = Field(description="The ID of the bot generating this post (e.g. Bot_A)")
    topic:        str = Field(description="The topic or theme of the post in 3-5 words")
    post_content: str = Field(description="The opinionated post content, max 280 characters")


# ---------------------------------------------------------------------------
# 3. LangGraph State — passed between nodes and mutated at each step
# ---------------------------------------------------------------------------
class BotState(TypedDict):
    bot_id:         str   # Which bot is running
    persona:        str   # Full persona description
    search_query:   str   # Query decided by Node 1
    search_result:  str   # Headlines returned by Node 2
    final_post:     dict  # Structured JSON output from Node 3


# ---------------------------------------------------------------------------
# 4. Node 1 — Decide: LLM picks a topic and formats a search query
# ---------------------------------------------------------------------------
def decide_search_node(state: BotState) -> dict:
    """
    The bot 'wakes up' and decides what it wants to post about today.
    The LLM returns a short, focused search query based on the persona.
    """
    print(f"  [Node 1 — Decide] {state['bot_id']} is choosing a topic...")

    messages = [
        SystemMessage(content=(
            f"You are a social media bot with this persona:\n{state['persona']}\n\n"
            "Decide what topic you want to post about today. "
            "Return ONLY a short search query (4-7 words, no punctuation). "
            "Do not explain. Just the query."
        )),
        HumanMessage(content="What do you want to post about today? Give me your search query."),
    ]

    response = llm.invoke(messages)
    search_query = response.content.strip().strip('"')
    print(f"  [Node 1 — Decide] Search query: '{search_query}'")

    return {"search_query": search_query}


# ---------------------------------------------------------------------------
# 5. Node 2 — Search: Executes the mock search tool
# ---------------------------------------------------------------------------
def web_search_node(state: BotState) -> dict:
    """
    Calls mock_searxng_search with the query from Node 1.
    Returns hardcoded headlines as real-world context for the post.
    """
    print(f"  [Node 2 — Search] Running mock search for: '{state['search_query']}'")

    # Invoke the @tool function directly (tool.invoke strips the ToolCall wrapper)
    result = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  [Node 2 — Search] Result: {result}")

    return {"search_result": result}


# ---------------------------------------------------------------------------
# 6. Node 3 — Draft: LLM generates a structured 280-char post
# ---------------------------------------------------------------------------
def draft_post_node(state: BotState) -> dict:
    """
    Uses the persona (system prompt) + search result (context) to draft a post.
    Structured output via Pydantic guarantees valid JSON every time.
    """
    print(f"  [Node 3 — Draft] Generating post for {state['bot_id']}...")

    # Bind structured output schema to the LLM (uses function calling internally)
    structured_llm = llm.with_structured_output(PostOutput)

    messages = [
        SystemMessage(content=(
            f"You are a social media bot. Your persona:\n{state['persona']}\n\n"
            "You MUST stay in character. Be opinionated, direct, and provocative. "
            "Your post must be under 280 characters."
        )),
        HumanMessage(content=(
            f"Today's news context: {state['search_result']}\n\n"
            f"Write a highly opinionated post about this. "
            f"Your bot_id is '{state['bot_id']}'. "
            f"Return the structured output with bot_id, topic, and post_content."
        )),
    ]

    output: PostOutput = structured_llm.invoke(messages)

    # Enforce 280-char limit (truncate if needed, preserve meaning)
    post = output.post_content
    if len(post) > 280:
        post = post[:277] + "..."

    result = {
        "bot_id":       output.bot_id,
        "topic":        output.topic,
        "post_content": post,
    }

    print(f"  [Node 3 — Draft] Post generated ({len(post)} chars)")
    return {"final_post": result}


# ---------------------------------------------------------------------------
# 7. Build and compile the LangGraph state machine
# ---------------------------------------------------------------------------
def build_graph() -> any:
    """Assembles the 3-node linear state graph and returns a compiled app."""
    graph = StateGraph(BotState)

    # Register nodes
    graph.add_node("decide", decide_search_node)
    graph.add_node("search", web_search_node)
    graph.add_node("draft",  draft_post_node)

    # Wire edges: decide → search → draft → END
    graph.set_entry_point("decide")
    graph.add_edge("decide", "search")
    graph.add_edge("search", "draft")
    graph.add_edge("draft",  END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 8. Demo — run all 3 bots through the graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()

    for bot_id, persona_data in PERSONAS.items():
        print(f"\n{'='*60}")
        print(f"Running LangGraph for: {bot_id} ({persona_data['name']})")
        print(f"{'='*60}")

        initial_state: BotState = {
            "bot_id":        bot_id,
            "persona":       persona_data["description"],
            "search_query":  "",
            "search_result": "",
            "final_post":    {},
        }

        final_state = app.invoke(initial_state)

        print(f"\n  ✅ FINAL JSON OUTPUT:")
        print(json.dumps(final_state["final_post"], indent=2))
