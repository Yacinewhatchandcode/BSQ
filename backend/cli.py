#!/usr/bin/env python3
"""
Cutting-edge CLI workflow (August 2025 feel):
- Web search (lite) for quick discovery
- Start server & open health
- Run E2E & variation tests
- Rapid UX/UI design via BahaiUXDesignerAgent

Usage:
  python backend/cli.py                # interactive menu
  python backend/cli.py search "query"
  python backend/cli.py serve --host 127.0.0.1 --port 8000
  python backend/cli.py e2e
  python backend/cli.py variations
  python backend/cli.py design "Design a landing hero for ..."
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from typing import List, Tuple

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def ddg_lite_search(query: str, num: int = 5) -> List[Tuple[str, str]]:
    url = "https://lite.duckduckgo.com/lite/"
    try:
        resp = requests.post(url, data={"q": query}, timeout=10)
        resp.raise_for_status()
        html = resp.text
        # Parse links from lite page: pattern <a rel="nofollow" href="LINK">TITLE</a>
        matches = re.findall(r'<a[^>]*href="(http[s]?://[^"]+)"[^>]*>(.*?)</a>', html, flags=re.I)
        results = []
        for href, title in matches:
            # Skip duckduckgo internal links
            if "duckduckgo.com" in href:
                continue
            clean = re.sub(r"<[^>]+>", "", title)
            results.append((clean, href))
            if len(results) >= num:
                break
        return results
    except Exception as e:
        return [("Search error", str(e))]


def action_search(query: str):
    console.rule("Web Search")
    results = ddg_lite_search(query)
    table = Table(show_lines=False)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("URL", style="magenta")
    for i, (title, url) in enumerate(results, 1):
        table.add_row(str(i), title, url)
    console.print(table)


def action_serve(host: str, port: int):
    console.rule("Start Server")
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", host, "--port", str(port)]
    console.print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def action_e2e():
    console.rule("E2E Test")
    subprocess.run([sys.executable, "backend/e2e_runner.py", "--headless"], check=False)


def action_variations():
    console.rule("160 Variations")
    subprocess.run([sys.executable, "backend/variation_tester.py"], check=False)


def action_design(prompt: str):
    console.rule("UX/UI Designer")
    from bahai_ux_designer_agent import BahaiUXDesignerAgent

    agent = BahaiUXDesignerAgent()
    res = agent.design_interface(prompt)
    concept = res.get("design_concept", "")
    console.print(Panel.fit(textwrap.shorten(concept, 2000), title="Design Concept"))
    impl = res.get("implementation", {})
    css = impl.get("css", "")
    html = impl.get("html", "")
    console.print(Panel.fit(css[:1200] + ("..." if len(css) > 1200 else ""), title="CSS (preview)"))
    console.print(Panel.fit(html[:1200] + ("..." if len(html) > 1200 else ""), title="HTML (preview)"))


def interactive_menu():
    console.print(Panel.fit("Spiritual Quest – Workflow CLI (Aug 2025)", subtitle="Use arrows not required – just number keys"))
    items = [
        "Web Search",
        "Start Server",
        "Run E2E",
        "Run 160 Variations",
        "UX/UI Design",
        "Exit",
    ]
    for idx, label in enumerate(items, 1):
        console.print(f"[cyan]{idx}[/cyan]. {label}")
    choice = console.input("\nSelect [1-6]: ").strip()
    if choice == "1":
        q = console.input("Query: ").strip()
        action_search(q)
    elif choice == "2":
        action_serve("127.0.0.1", 8000)
    elif choice == "3":
        action_e2e()
    elif choice == "4":
        action_variations()
    elif choice == "5":
        p = console.input("Design brief: ").strip()
        action_design(p)
    else:
        console.print("Bye.")


def main():
    parser = argparse.ArgumentParser(description="Workflow CLI")
    sub = parser.add_subparsers(dest="cmd")

    sp = sub.add_parser("search"); sp.add_argument("query")
    sp = sub.add_parser("serve"); sp.add_argument("--host", default="127.0.0.1"); sp.add_argument("--port", type=int, default=8000)
    sub.add_parser("e2e")
    sub.add_parser("variations")
    sp = sub.add_parser("design"); sp.add_argument("prompt")

    args = parser.parse_args()
    if args.cmd == "search":
        action_search(args.query)
    elif args.cmd == "serve":
        action_serve(args.host, args.port)
    elif args.cmd == "e2e":
        action_e2e()
    elif args.cmd == "variations":
        action_variations()
    elif args.cmd == "design":
        action_design(args.prompt)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()


