"""Locust load test: locust -f scripts/locustfile.py --host http://localhost:8000
Set GATEWAY_KEY to a key created in the dashboard. Raise the key's rate limit first (for example 100000/min)."""
import os
import random

from locust import HttpUser, between, task

HEADERS = {"Authorization": f"Bearer {os.getenv('GATEWAY_KEY', '')}"}
MODEL = os.getenv("GATEWAY_MODEL", "mock-smart")
PROMPTS = ["What is the capital of France?", "What is 12 * 7?", "What is the capital of India?"]


class GatewayUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task(3)
    def repeated_prompt(self):  # should be served from the cache
        self.client.post("/v1/chat", headers=HEADERS, name="/v1/chat (repeated)",
                         json={"model": MODEL, "messages": [{"role": "user", "content": random.choice(PROMPTS)}]})

    @task(1)
    def unique_prompt(self):  # always a cache miss
        a, b = random.randint(1, 9999), random.randint(1, 9999)
        self.client.post("/v1/chat", headers=HEADERS, name="/v1/chat (unique)",
                         json={"model": MODEL, "messages": [{"role": "user", "content": f"What is {a} + {b}?"}]})
