#!/usr/bin/env python3
"""
Playwright E2E Scenario Runner
 - Starts the backend server on a test port (8010)
 - Drives the elegant manuscript UI
 - Runs 50+ end-to-end scenarios
 - Verifies expected behavior per scenario (normal vs spiritual)
 - Prints a JSON summary with pass/fail counts and examples
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

TEST_HOST = "127.0.0.1"
TEST_PORT = int(os.getenv("E2E_PORT", "8010"))
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


@dataclass
class Scenario:
    name: str
    message: str
    expectation: str  # "normal" or "spiritual"
    min_quotes: int = 0


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    reason: str = ""


def build_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []

    # Normal conversation (no quotes expected)
    normal_msgs = [
        "hello",
        "hi",
        "hey there",
        "how are you",
        "what's up",
        "nice to meet you",
        "how's your day going",
        "tell me something fun",
        "do you like music",
        "what do you think about the weather",
        "i went for a walk",
        "i cooked dinner",
        "i'm working on a project",
        "i love coding",
        "tell me about yourself",
        "good morning",
        "good evening",
        "good night",
        "thanks",
        "see you later",
    ]
    for i, msg in enumerate(normal_msgs, 1):
        scenarios.append(Scenario(name=f"normal_{i}", message=msg, expectation="normal"))

    # Emotional support (by spec: normal empathy, no quotes unless explicitly asked)
    emotional_msgs = [
        "i feel sad",
        "i feel anxious",
        "i'm struggling",
        "i'm stressed",
        "i feel lonely",
        "i'm afraid",
        "i'm confused",
        "life is hard",
        "i feel lost",
        "i need comfort",
    ]
    for i, msg in enumerate(emotional_msgs, 1):
        scenarios.append(Scenario(name=f"emotional_{i}", message=msg, expectation="normal", min_quotes=0))

    # Emotional + explicit request for quotes (spiritual expected)
    emotional_with_quotes = [
        "i feel sad, can you share a hidden words quote",
        "i'm anxious, please give me two quotes",
        "i'm struggling, what do the hidden words say",
        "i feel lonely, share a quote",
        "i'm afraid, spiritual guidance please",
    ]
    for i, msg in enumerate(emotional_with_quotes, 1):
        min_q = 2 if "two" in msg else 1
        scenarios.append(Scenario(name=f"emotional_spiritual_{i}", message=msg, expectation="spiritual", min_quotes=min_q))

    # Explicit quote/guidance requests (quotes expected)
    quote_msgs = [
        "share a hidden words quote",
        "quote about love",
        "quote about patience",
        "what do the hidden words say about justice",
        "give me a quote",
        "could you share a spiritual quote",
        "please share two quotes about love",
        "give me three quotes about patience",
        "list two quotes about joy",
        "i need guidance from the hidden words",
        "show me a quotation",
        "find a quote about hope",
        "retrieve a quote about mercy",
        "get a quote about forgiveness",
        "any quotation about unity",
        "any quote about peace",
        "share a quote from the hidden words",
        "what does it say about detachment",
        "teachings about love",
        "spiritual guidance on adversity",
    ]
    for i, msg in enumerate(quote_msgs, 1):
        # Try to infer requested count
        min_q = 1
        if any(tok in msg for tok in ["two", "2"]):
            min_q = 2
        if any(tok in msg for tok in ["three", "3"]):
            min_q = 3
        scenarios.append(Scenario(name=f"quote_{i}", message=msg, expectation="spiritual", min_quotes=min_q))

    # Add some prefixed polite variants to exceed 50 total
    prefixes = ["hey ", "hi ", "please ", "could you ", "can you "]
    extras = [
        "share a quote about love",
        "give me two quotes about patience",
        "what do the hidden words say about kindness",
        "i feel anxious about tomorrow",
        "just saying hello",
    ]
    for pre in prefixes:
        for text in extras:
            # Spiritual only if explicitly asking for quote or mentioning Hidden Words
            exp = "spiritual" if ("quote" in text or "hidden words" in text) else "normal"
            min_q = 2 if ("two" in text and exp == "spiritual") else (1 if exp == "spiritual" else 0)
            scenarios.append(Scenario(name=f"mix_{pre.strip()}_{text[:12].replace(' ', '_')}", message=pre + text, expectation=exp, min_quotes=min_q))

    return scenarios


def start_server() -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--app-dir",
        "backend",
        "--host",
        TEST_HOST,
        "--port",
        str(TEST_PORT),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def wait_for_health(timeout_seconds: int = 40) -> bool:
    import http.client
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(TEST_HOST, TEST_PORT, timeout=2)
            conn.request("GET", "/api/health")
            resp = conn.getresponse()
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


async def run_playwright_scenarios(scenarios: List[Scenario], headless: bool = True) -> Dict[str, Any]:
    from playwright.async_api import async_playwright

    results: List[ScenarioResult] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()

        # Navigate
        # Use the elegant manuscript UI (root) which has .reveal-button
        await page.goto(BASE_URL, timeout=30000)
        await page.wait_for_selector(".reveal-button", timeout=15000)
        await page.wait_for_selector("#messageInput", timeout=15000)

        async def send_and_assert(scene: Scenario) -> ScenarioResult:
            try:
                # Fill and click
                await page.fill("#messageInput", scene.message)
                await page.click(".reveal-button")

                # Wait for an agent message to appear
                await page.wait_for_selector(".message.agent, .message.agent .message-content", timeout=35000)
                # Give UI a moment to format quotes
                await page.wait_for_timeout(300)

                # Inspect only the LAST agent message
                last_agent = page.locator(".message.agent").last
                quote_blocks = await last_agent.locator(".hidden-word-quote").count()
                agent_text = await last_agent.inner_text()
                agent_html = await last_agent.inner_html()
                agent_text_l = agent_text.lower().strip()

                if scene.expectation == "normal":
                    # Expect no special quote blocks and avoid obvious spiritual keywords
                    if quote_blocks > 0:
                        return ScenarioResult(scene.name, False, reason=f"Unexpected quote blocks: {quote_blocks}")
                    forbidden = ["hidden words", "o son of", "o friend", "quotation", "spiritual"]
                    if any(tok in agent_text_l for tok in forbidden):
                        return ScenarioResult(scene.name, False, reason="Response contains spiritual content")
                    # Non-empty plain response
                    if not agent_text_l:
                        return ScenarioResult(scene.name, False, reason="Empty response")
                    return ScenarioResult(scene.name, True)

                # Spiritual expectation
                # Consider success if we have either special quote blocks or clearly quoted text
                # Accept standard (") or curly quotes (“ ”), or explicit Hidden Words markers
                curly_left = agent_text.count('“')
                curly_right = agent_text.count('”')
                std_quotes = agent_text.count('"')
                hw_markers = any(k in agent_text_l for k in ["hidden words", "o son of", "o friend", "o children"]) 
                # fallback: any quote cards in the chat area
                dom_quote = ("hidden-word-quote" in agent_html) or (await page.locator(".hidden-word-quote").count() > 0)
                has_quotes = (quote_blocks >= max(1, scene.min_quotes)) or ((std_quotes + min(curly_left, curly_right)) >= 2) or hw_markers or dom_quote
                if not has_quotes:
                    return ScenarioResult(scene.name, False, reason=f"No quotes detected (blocks={quote_blocks}, ascii={std_quotes}, curlyL={curly_left}, curlyR={curly_right}, domQuote={dom_quote})")
                return ScenarioResult(scene.name, True)

            except Exception as e:
                return ScenarioResult(scene.name, False, reason=f"Exception: {e}")

        # Run sequentially; reload page between scenarios to isolate state
        for s in scenarios:
            results.append(await send_and_assert(s))
            await page.goto(BASE_URL, timeout=30000)
            await page.wait_for_selector(".reveal-button", timeout=15000)
            await page.wait_for_selector("#messageInput", timeout=15000)

        await context.close()
        await browser.close()

    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    sample_failures = [asdict(r) for r in results if not r.ok][:5]
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "failures": sample_failures,
        "url": BASE_URL,
    }


def main() -> int:
    headless = True
    scenarios = build_scenarios()
    proc: Optional[subprocess.Popen] = None
    try:
        # Start server
        proc = start_server()
        if not wait_for_health(50):
            print(json.dumps({
                "ok": False,
                "error": "Server did not become healthy",
                "url": BASE_URL,
            }))
            return 1

        # Run scenarios
        summary = asyncio.get_event_loop().run_until_complete(
            run_playwright_scenarios(scenarios, headless=headless)
        )
        summary["ok"] = summary["failed"] == 0
        print(json.dumps(summary))
        return 0 if summary["ok"] else 2
    finally:
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.kill(proc.pid, signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())


