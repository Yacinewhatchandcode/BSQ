#!/usr/bin/env python3
"""
Autonomous E2E Runner
- Starts the backend server on a test port
- Waits for health
- Uses Playwright to drive the UI and send a message
- Verifies a non-empty agent response is rendered
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import http.client


TEST_HOST = "127.0.0.1"
TEST_PORT = int(os.getenv("E2E_PORT", "8010"))
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


@dataclass
class E2EResult:
    ok: bool
    page_loaded: bool
    websocket_connected: bool
    response_rendered: bool
    details: str
    url: str = BASE_URL


def wait_for_health(timeout_seconds: int = 30) -> bool:
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


async def run_playwright(headless: bool = True) -> E2EResult:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        page_loaded = False
        websocket_connected = False
        response_rendered = False
        details = ""

        try:
            await page.goto(BASE_URL, timeout=30000)
            # Core UI elements
            await page.wait_for_selector(".reveal-button", timeout=10000)
            await page.wait_for_selector("#messageInput", timeout=10000)
            page_loaded = True

            # Observe WebSocket events (best-effort)
            ws_events = []
            page.on("websocket", lambda ws: ws_events.append("open"))

            # Send a message
            await page.fill("#messageInput", "what is love")
            await page.click(".reveal-button")

            # Wait for typing indicator to hide or for an agent message to appear
            try:
                await page.wait_for_function(
                    "document.getElementById('typingIndicator') ? document.getElementById('typingIndicator').style.display === 'none' : true",
                    timeout=30000,
                )
            except Exception:
                # ignore; continue to check message presence
                pass

            # Expect any agent message
            try:
                await page.wait_for_selector(".message.agent .message-content, .message.agent", timeout=30000)
                # Basic non-empty content check
                content = await page.locator(".message.agent").last.inner_text()
                response_rendered = bool(content.strip())
            except Exception as e:
                details = f"No agent message: {e}"

            websocket_connected = len(ws_events) > 0

        except Exception as e:
            details = f"E2E error: {e}"
        finally:
            await context.close()
            await browser.close()

        ok = page_loaded and response_rendered
        return E2EResult(
            ok=ok,
            page_loaded=page_loaded,
            websocket_connected=websocket_connected,
            response_rendered=response_rendered,
            details=details,
        )


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
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


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run E2E UI test against local server")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    args = parser.parse_args()

    proc: Optional[subprocess.Popen] = None
    try:
        proc = start_server()
        if not wait_for_health(45):
            print(json.dumps({
                "ok": False,
                "error": "Server did not become healthy",
                "url": BASE_URL,
            }))
            return 1

        result = asyncio.get_event_loop().run_until_complete(run_playwright(headless=args.headless))
        print(json.dumps(asdict(result)))
        return 0 if result.ok else 2
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


