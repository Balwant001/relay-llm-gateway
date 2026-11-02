"""Model registry and prices in USD per 1M tokens (input, output).
Prices are approximate: check each provider's pricing page and edit before relying on the numbers."""

MODELS = {
    "mock-fast": {"provider": "mock", "in": 0.10, "out": 0.30},
    "mock-smart": {"provider": "mock", "in": 1.00, "out": 3.00},
    "mock-flaky": {"provider": "mock", "in": 0.10, "out": 0.30},  # always fails: demonstrates failover
    "claude-sonnet-4-6": {"provider": "anthropic", "in": 3.00, "out": 15.00},
    "gpt-4o-mini": {"provider": "openai", "in": 0.15, "out": 0.60},
    "llama-3.1-8b-instant": {"provider": "groq", "in": 0.05, "out": 0.08},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    m = MODELS[model]
    return (input_tokens * m["in"] + output_tokens * m["out"]) / 1_000_000
