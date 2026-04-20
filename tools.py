# tools.py
# Mock search tool simulating a SearXNG news API.
# Returns hardcoded recent headlines based on keywords in the query.
# The @tool decorator makes this compatible with LangChain/LangGraph tool calling.

from langchain_core.tools import tool


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a web search by returning hardcoded recent news headlines.
    Matches keywords in the query to return relevant results.

    Args:
        query: A natural language search query string.

    Returns:
        A string containing mock news headlines relevant to the query.
    """
    query_lower = query.lower()

    # Keyword → headline mapping (ordered by specificity)
    keyword_headlines = {
        "crypto":     "Bitcoin hits new all-time high amid regulatory ETF approvals; altcoins surge 40%.",
        "bitcoin":    "Bitcoin hits new all-time high amid regulatory ETF approvals; altcoins surge 40%.",
        "ai":         "OpenAI's latest model sparks mass layoffs in junior developer roles across tech firms.",
        "openai":     "OpenAI's latest model sparks mass layoffs in junior developer roles across tech firms.",
        "elon":       "Elon Musk's xAI raises $6B; Grok 3 claims to outperform GPT-4 on benchmarks.",
        "space":      "SpaceX Starship completes first full orbital test flight, landing both stages.",
        "regulation": "EU AI Act enforcement begins Q3 2025; fines up to 3% of global annual turnover.",
        "privacy":    "Meta faces $1.3B GDPR fine as EU regulators tighten data transfer rules.",
        "monopoly":   "DOJ antitrust case against Google advances; breakup of ad business on the table.",
        "market":     "Fed signals rate pause; S&P 500 hits record high on cooling inflation data.",
        "interest":   "Fed holds rates steady at 5.25%; analysts predict first cut in Q4 2025.",
        "trading":    "Algorithmic trading now accounts for 73% of US equity volume, SEC report finds.",
        "stock":      "S&P 500 hits record high; tech sector leads gains with 12% quarterly return.",
        "ev":         "Tesla battery degradation lawsuits rise in California; average loss 8% over 5 years.",
        "electric":   "EV adoption hits 18% of new car sales globally; charging infrastructure lags.",
        "climate":    "UN report: 2024 was hottest year on record; carbon emissions still rising.",
        "billionaire":"Wealth of top 10 billionaires doubled since 2020 while median wages stagnated.",
        "social":     "Instagram usage among teens drops 22% amid mental health regulation pressure.",
    }

    # Return first matching headline
    for keyword, headline in keyword_headlines.items():
        if keyword in query_lower:
            return headline

    # Fallback
    return "No major headlines found. Markets remain cautious amid geopolitical uncertainty."
