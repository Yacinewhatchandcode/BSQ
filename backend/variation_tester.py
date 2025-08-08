#!/usr/bin/env python3
"""
Variation Tester: send 120+ question variations to /api/chat and report success rate
"""

import itertools
import time
import requests

BASE_URL = "http://localhost:8000"

def generate_variations():
    intents = [
        "quote about love",
        "share a hidden words quote",
        "what is love",
        "spiritual guidance",
        "hello",
        "how are you",
        "i feel sad",
        "i feel anxious",
        "tell me about patience",
        "tell me about justice",
    ]
    modifiers = [
        "please",
        "now",
        "briefly",
        "in one line",
        "give me two",
        "three please",
        "from bahá'u'lláh",
        "with attribution",
        "no attribution",
        "arabic if possible",
        "",
    ]
    prefixes = [
        "",
        "hey ",
        "hi ",
        "can you ",
        "could you ",
        "please ",
    ]
    suffixes = [
        "?",
        ".",
        "!",
        "",
    ]
    variations = []
    for pre, intent, mod, suf in itertools.product(prefixes, intents, modifiers, suffixes):
        text = f"{pre}{intent} {mod}{suf}".strip()
        # limit total size
        if len(variations) >= 160:
            break
        variations.append(text)
    return variations

def test_variations():
    ok = 0
    total = 0
    errors = []
    for q in generate_variations():
        total += 1
        try:
            resp = requests.post(
                f"{BASE_URL}/api/chat",
                data={"message": q},
                timeout=10,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200 and resp.json().get("response"):
                ok += 1
            else:
                errors.append((q, f"status={resp.status_code}", resp.text[:200]))
        except Exception as e:
            errors.append((q, str(e)))
        time.sleep(0.05)
    print(f"Success: {ok}/{total} ({ok/total*100:.1f}%)")
    if errors:
        print("Examples of failures (up to 5):")
        for item in errors[:5]:
            print("- ", item)

if __name__ == "__main__":
    test_variations()


