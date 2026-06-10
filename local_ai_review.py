import ollama


def review_stock_ranking(ranking_text: str) -> str:
    prompt = f"""
You are reviewing a swing-trading stock ranking system.

Here is the ranked list:

{ranking_text}

Analyze:
1. Which stocks look strongest technically.
2. Which stocks look riskiest.
3. What earnings/news/geopolitical risks should be checked.
4. Whether the ranking logic seems reasonable.
5. A cautious final summary.

Do not give guaranteed predictions.
Do not tell me to buy or sell.
Focus on risk-aware analysis.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    sample_ranking = """
    1. NVDA - Score: 92 - Strong momentum, above 20DMA and 50DMA
    2. AMD - Score: 84 - Improving trend, high volatility
    3. MU - Score: 78 - Strong recent move, but memory-cycle risk
    4. AAPL - Score: 70 - Stable but weaker momentum
    """

    analysis = review_stock_ranking(sample_ranking)
    print(analysis)
