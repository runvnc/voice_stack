#!/usr/bin/env python3
"""Quick LLM latency test - measures time to first token and full response.

Usage: python3 test_llm_latency.py [--url http://localhost:8000/v1]
"""
import time
import json
import argparse
import urllib.request

def test_ttft(url, prompt, max_tokens=50):
    """Measure time-to-first-token (streaming) and total response time."""
    payload = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    first_token_time = None
    full_text = ""
    token_count = 0

    with urllib.request.urlopen(req, timeout=30) as resp:
        buffer = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buffer += chunk
            # Process complete SSE lines
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text += content
                        token_count += 1
                except json.JSONDecodeError:
                    pass

    t_end = time.perf_counter()
    ttft = (first_token_time - t_start) * 1000 if first_token_time else -1
    total = (t_end - t_start) * 1000

    return {
        "ttft_ms": ttft,
        "total_ms": total,
        "tokens": token_count,
        "text": full_text.strip(),
        "ms_per_token": total / max(token_count, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    prompts = [
        "Say just 'OK' and nothing else.",
        "The caller said 'yes'. Respond briefly.",
        "Which employee are you calling about?",
        "They never worked here. What should I say next?",
        "Hello, this is Target. How can I help you?",
    ]

    print(f"Testing LLM at {args.url}")
    print(f"{'#':<3} {'TTFT':>8} {'Total':>8} {'Tok':>4} {'ms/tok':>7}  Response")
    print("-" * 80)

    ttfts = []
    for i in range(args.runs):
        prompt = prompts[i % len(prompts)]
        try:
            r = test_ttft(args.url, prompt, max_tokens=30)
            ttfts.append(r["ttft_ms"])
            text_preview = r["text"][:50] + ("..." if len(r["text"]) > 50 else "")
            print(f"{i+1:<3} {r['ttft_ms']:>7.0f}ms {r['total_ms']:>7.0f}ms {r['tokens']:>4} {r['ms_per_token']:>6.1f}  {text_preview}")
        except Exception as e:
            print(f"{i+1:<3} ERROR: {e}")

    if ttfts:
        print(f"\nAvg TTFT: {sum(ttfts)/len(ttfts):.0f}ms")
        print(f"Min TTFT: {min(ttfts):.0f}ms")
        print(f"Max TTFT: {max(ttfts):.0f}ms")


if __name__ == "__main__":
    main()
