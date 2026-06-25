#!/usr/bin/env python3
"""
vLLM concurrent voice-stack TTFT tester.

Tests how many concurrent chat sessions can complete simultaneously before latency rises too much.

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Streaming responses
- Measures TTFT per conversation turn
- Each simulated session has 3 to 6 turns
- Simulated user think-time between turns
- Varies scenarios, user profiles, prompts, and conversation details
- Runs a concurrency ladder, e.g. 1,2,4,8,16,32 sessions
- Outputs summary table plus JSONL/CSV files

Example:
    python vllm_concurrency_ttft_test.py \
      --base-url http://127.0.0.1:8000 \
      --model Intel/Qwen3.6-27B-int4-AutoRound \
      --concurrency-levels 1,2,4,8,12,16,24,32 \
      --max-tokens 120
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    print("Missing dependency: aiohttp")
    print("Install with: pip install aiohttp")
    sys.exit(1)


SCENARIOS = [
    {
        "name": "restaurant_booking",
        "system": "You are a concise phone agent helping users make restaurant reservations. Keep replies natural and brief, suitable for voice.",
        "seed_user_messages": [
            "Hi, I want to book dinner for two tonight.",
            "Can you help me find a table somewhere quiet?",
            "I'm looking for a reservation around 7 or 8 PM.",
        ],
        "followups": [
            "Italian would be good, but I'm flexible.",
            "Actually make that three people.",
            "Around downtown if possible.",
            "Could you check if they have outdoor seating?",
            "That sounds good, can you confirm it?",
            "Please use the name {name}.",
        ],
    },
    {
        "name": "tech_support",
        "system": "You are a calm technical support phone agent. Ask one question at a time and keep responses short.",
        "seed_user_messages": [
            "My internet keeps dropping during video calls.",
            "I'm having trouble with my router.",
            "My laptop says connected but pages won't load.",
        ],
        "followups": [
            "It started yesterday afternoon.",
            "The WiFi icon still shows connected.",
            "I already restarted it once.",
            "There are three other devices on the network.",
            "The modem lights look normal to me.",
            "Can you walk me through the next step?",
        ],
    },
    {
        "name": "medical_scheduling",
        "system": "You are a medical appointment scheduling assistant. Do not provide medical diagnosis. Keep replies concise and voice-friendly.",
        "seed_user_messages": [
            "I need to schedule a follow-up appointment.",
            "Can I book a visit with Dr. Patel?",
            "I'm calling to reschedule my appointment.",
        ],
        "followups": [
            "Mornings are better for me.",
            "Next week would work.",
            "My date of birth is {dob}.",
            "I prefer in person if available.",
            "Thursday is not good for me.",
            "Can you send a confirmation text?",
        ],
    },
    {
        "name": "banking_customer_service",
        "system": "You are a banking customer service voice assistant. Be concise and avoid asking for full sensitive information.",
        "seed_user_messages": [
            "I have a question about a charge on my card.",
            "My debit card isn't working.",
            "I want to check whether a payment went through.",
        ],
        "followups": [
            "The charge was from yesterday.",
            "It was for about ${amount}.",
            "I don't recognize the merchant name.",
            "The last four digits are {last4}.",
            "Should I lock the card?",
            "Can you explain what happens next?",
        ],
    },
    {
        "name": "travel_planning",
        "system": "You are a travel planning voice assistant. Keep answers short, practical, and conversational.",
        "seed_user_messages": [
            "I need help planning a weekend trip.",
            "Can you suggest flights for a short trip?",
            "I'm looking for hotel options near the city center.",
        ],
        "followups": [
            "I'm leaving from {city}.",
            "My budget is around ${amount}.",
            "I prefer direct flights.",
            "Two nights should be enough.",
            "I'd like somewhere walkable.",
            "Can you summarize the best option?",
        ],
    },
]

NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Sam", "Jamie",
    "Avery", "Quinn", "Drew", "Cameron", "Robin", "Skyler", "Reese"
]

CITIES = [
    "Seattle", "Austin", "Denver", "Boston", "Chicago", "Phoenix", "Miami",
    "San Diego", "Portland", "Atlanta", "Nashville", "Minneapolis"
]


@dataclass
class TurnMetric:
    run_id: str
    concurrency_level: int
    session_id: str
    scenario: str
    turn_index: int
    total_turns: int
    prompt_chars: int
    input_messages: int
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]
    output_chars: int
    finish_reason: Optional[str]
    error: Optional[str]


@dataclass
class SessionMetric:
    run_id: str
    concurrency_level: int
    session_id: str
    scenario: str
    total_turns: int
    completed_turns: int
    failed_turns: int
    session_wall_ms: float


def now() -> float:
    return time.perf_counter()


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1f}"


def random_dob() -> str:
    year = random.randint(1950, 2002)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{month:02d}/{day:02d}/{year}"


def fill_template(s: str) -> str:
    return s.format(
        name=random.choice(NAMES),
        city=random.choice(CITIES),
        amount=random.randint(40, 900),
        last4=random.randint(1000, 9999),
        dob=random_dob(),
    )


def make_session_plan(session_num: int) -> Dict[str, Any]:
    scenario = random.choice(SCENARIOS)
    total_turns = random.randint(3, 6)
    name = random.choice(NAMES)
    city = random.choice(CITIES)

    system = (
        scenario["system"]
        + f" The caller's first name may be {name}. The caller may be located near {city}."
        + " Reply as if this is a real-time voice call: concise, no markdown, no bullet lists unless absolutely necessary."
    )

    first = fill_template(random.choice(scenario["seed_user_messages"]))
    followups = scenario["followups"][:]
    random.shuffle(followups)
    user_messages = [first] + [fill_template(x) for x in followups[: total_turns - 1]]

    return {
        "scenario": scenario["name"],
        "system": system,
        "user_messages": user_messages,
        "total_turns": total_turns,
    }


async def stream_chat_completion(
    http: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> Tuple[Optional[float], Optional[float], str, Optional[str], Optional[str]]:
    """
    Returns:
        ttft_ms, total_latency_ms, output_text, finish_reason, error
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = now()
    first_token_time = None
    output_parts: List[str] = []
    finish_reason = None

    try:
        async with http.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                text = await resp.text()
                elapsed = (now() - start) * 1000.0
                return None, elapsed, "", None, f"HTTP {resp.status}: {text[:500]}"

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # SSE can contain multiple lines, but vLLM usually emits one data line at a time.
                for subline in line.splitlines():
                    subline = subline.strip()
                    if not subline.startswith("data:"):
                        continue
                    data = subline[len("data:"):].strip()
                    if data == "[DONE]":
                        total_ms = (now() - start) * 1000.0
                        ttft_ms = None if first_token_time is None else (first_token_time - start) * 1000.0
                        return ttft_ms, total_ms, "".join(output_parts), finish_reason, None

                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = obj.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    content = delta.get("content")

                    # Some models/endpoints may send role-only first chunk; TTFT should measure first actual text/audio-relevant token.
                    if content:
                        if first_token_time is None:
                            first_token_time = now()
                        output_parts.append(content)

            total_ms = (now() - start) * 1000.0
            ttft_ms = None if first_token_time is None else (first_token_time - start) * 1000.0
            return ttft_ms, total_ms, "".join(output_parts), finish_reason, None

    except Exception as e:
        elapsed = (now() - start) * 1000.0
        return None, elapsed, "", None, repr(e)


async def run_one_session(
    run_id: str,
    concurrency_level: int,
    session_num: int,
    http: aiohttp.ClientSession,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    min_user_delay: float,
    max_user_delay: float,
    timeout_s: float,
    verbose: bool,
) -> Tuple[List[TurnMetric], SessionMetric]:
    session_id = f"s{session_num:04d}-{uuid.uuid4().hex[:8]}"
    plan = make_session_plan(session_num)
    scenario = plan["scenario"]
    total_turns = plan["total_turns"]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": plan["system"]}
    ]

    turn_metrics: List[TurnMetric] = []
    session_start = now()
    failed = 0

    # Random initial stagger so all sessions do not hit the server on the exact same millisecond.
    await asyncio.sleep(random.uniform(0.0, 0.35))

    for turn_index, user_msg in enumerate(plan["user_messages"], start=1):
        if turn_index > 1:
            # Simulate the human thinking/speaking delay after the assistant response.
            await asyncio.sleep(random.uniform(min_user_delay, max_user_delay))

        messages.append({"role": "user", "content": user_msg})
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        ttft_ms, total_ms, output_text, finish_reason, error = await stream_chat_completion(
            http=http,
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

        if error:
            failed += 1
            # Stop this session on failure; in a real voice stack this call would likely be degraded/lost.
            if verbose:
                print(f"[ERR] c={concurrency_level} {session_id} turn={turn_index}: {error}")
        else:
            messages.append({"role": "assistant", "content": output_text})

        metric = TurnMetric(
            run_id=run_id,
            concurrency_level=concurrency_level,
            session_id=session_id,
            scenario=scenario,
            turn_index=turn_index,
            total_turns=total_turns,
            prompt_chars=prompt_chars,
            input_messages=len(messages),
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            output_chars=len(output_text),
            finish_reason=finish_reason,
            error=error,
        )
        turn_metrics.append(metric)

        if verbose:
            status = "ERR" if error else "OK"
            print(
                f"[{status}] c={concurrency_level:<3} {session_id} "
                f"scenario={scenario:<24} turn={turn_index}/{total_turns} "
                f"ttft={fmt_ms(ttft_ms)}ms total={fmt_ms(total_ms)}ms chars={len(output_text)}"
            )

        if error:
            break

    session_wall_ms = (now() - session_start) * 1000.0
    session_metric = SessionMetric(
        run_id=run_id,
        concurrency_level=concurrency_level,
        session_id=session_id,
        scenario=scenario,
        total_turns=total_turns,
        completed_turns=len([m for m in turn_metrics if not m.error]),
        failed_turns=failed,
        session_wall_ms=session_wall_ms,
    )
    return turn_metrics, session_metric


async def health_check(base_url: str, model: str) -> None:
    url = base_url.rstrip("/") + "/v1/models"
    async with aiohttp.ClientSession() as http:
        try:
            async with http.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    print(f"WARNING: /v1/models returned HTTP {resp.status}: {text[:500]}")
                    return
                obj = json.loads(text)
                ids = [x.get("id") for x in obj.get("data", [])]
                print(f"Server models: {ids}")
                if model not in ids:
                    print(f"WARNING: requested model not listed by server: {model}")
        except Exception as e:
            print(f"WARNING: health check failed: {repr(e)}")


async def run_concurrency_level(
    args: argparse.Namespace,
    run_id: str,
    concurrency_level: int,
) -> Tuple[List[TurnMetric], List[SessionMetric]]:
    print(f"\n=== Running concurrency level: {concurrency_level} sessions ===")
    connector = aiohttp.TCPConnector(limit=max(concurrency_level * 2, 32), ttl_dns_cache=300)
    all_turns: List[TurnMetric] = []
    all_sessions: List[SessionMetric] = []

    async with aiohttp.ClientSession(connector=connector) as http:
        tasks = [
            run_one_session(
                run_id=run_id,
                concurrency_level=concurrency_level,
                session_num=i + 1,
                http=http,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                min_user_delay=args.min_user_delay,
                max_user_delay=args.max_user_delay,
                timeout_s=args.timeout,
                verbose=args.verbose,
            )
            for i in range(concurrency_level)
        ]

        results = await asyncio.gather(*tasks)
        for turns, session in results:
            all_turns.extend(turns)
            all_sessions.append(session)

    summarize_level(concurrency_level, all_turns, all_sessions)
    return all_turns, all_sessions


def summarize_level(
    concurrency_level: int,
    turns: List[TurnMetric],
    sessions: List[SessionMetric],
) -> None:
    ttfts = [t.ttft_ms for t in turns if t.ttft_ms is not None and not t.error]
    totals = [t.total_latency_ms for t in turns if t.total_latency_ms is not None and not t.error]
    errors = [t for t in turns if t.error]
    completed_sessions = [s for s in sessions if s.failed_turns == 0]

    print(f"\n--- Summary for concurrency {concurrency_level} ---")
    print(f"sessions completed: {len(completed_sessions)}/{len(sessions)}")
    print(f"turns completed:    {len(ttfts)}/{len(turns)}")
    print(f"turn errors:        {len(errors)}")

    if ttfts:
        print(
            "TTFT ms:           "
            f"mean={statistics.mean(ttfts):.1f} "
            f"p50={percentile(ttfts, 50):.1f} "
            f"p90={percentile(ttfts, 90):.1f} "
            f"p95={percentile(ttfts, 95):.1f} "
            f"p99={percentile(ttfts, 99):.1f} "
            f"max={max(ttfts):.1f}"
        )
    if totals:
        print(
            "Total latency ms:  "
            f"mean={statistics.mean(totals):.1f} "
            f"p50={percentile(totals, 50):.1f} "
            f"p90={percentile(totals, 90):.1f} "
            f"p95={percentile(totals, 95):.1f} "
            f"p99={percentile(totals, 99):.1f} "
            f"max={max(totals):.1f}"
        )


def write_outputs(out_prefix: str, turns: List[TurnMetric], sessions: List[SessionMetric]) -> None:
    jsonl_path = out_prefix + "_turns.jsonl"
    csv_path = out_prefix + "_turns.csv"
    sessions_path = out_prefix + "_sessions.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    with open(sessions_path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    if turns:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(turns[0]).keys()))
            writer.writeheader()
            for t in turns:
                writer.writerow(asdict(t))

    print(f"\nWrote:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print(f"  {sessions_path}")


def final_ladder_summary(turns: List[TurnMetric], sessions: List[SessionMetric]) -> None:
    print("\n=== Final concurrency ladder summary ===")
    levels = sorted(set(t.concurrency_level for t in turns))
    header = (
        "conc | sessions ok | turns ok | errors | "
        "ttft mean | ttft p50 | ttft p90 | ttft p95 | ttft p99 | ttft max | total p95"
    )
    print(header)
    print("-" * len(header))

    for c in levels:
        level_turns = [t for t in turns if t.concurrency_level == c]
        level_sessions = [s for s in sessions if s.concurrency_level == c]
        ok_sessions = len([s for s in level_sessions if s.failed_turns == 0])
        ok_ttfts = [t.ttft_ms for t in level_turns if t.ttft_ms is not None and not t.error]
        ok_totals = [t.total_latency_ms for t in level_turns if t.total_latency_ms is not None and not t.error]
        err_count = len([t for t in level_turns if t.error])

        if ok_ttfts:
            mean = statistics.mean(ok_ttfts)
            p50 = percentile(ok_ttfts, 50)
            p90 = percentile(ok_ttfts, 90)
            p95 = percentile(ok_ttfts, 95)
            p99 = percentile(ok_ttfts, 99)
            mx = max(ok_ttfts)
            total_p95 = percentile(ok_totals, 95) if ok_totals else None
            print(
                f"{c:>4} | "
                f"{ok_sessions:>3}/{len(level_sessions):<3}       | "
                f"{len(ok_ttfts):>3}/{len(level_turns):<3} | "
                f"{err_count:>6} | "
                f"{fmt_ms(mean):>9} | "
                f"{fmt_ms(p50):>8} | "
                f"{fmt_ms(p90):>8} | "
                f"{fmt_ms(p95):>8} | "
                f"{fmt_ms(p99):>8} | "
                f"{fmt_ms(mx):>8} | "
                f"{fmt_ms(total_p95):>9}"
            )
        else:
            print(
                f"{c:>4} | "
                f"{ok_sessions:>3}/{len(level_sessions):<3}       | "
                f"0/{len(level_turns):<3}   | "
                f"{err_count:>6} | no successful turns"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for vLLM OpenAI-compatible server")
    parser.add_argument("--model", default="Intel/Qwen3.6-27B-int4-AutoRound", help="Model name served by vLLM")
    parser.add_argument("--concurrency-levels", default="1,2,4,8,12,16,24,32", help="Comma-separated session counts")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max assistant tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-user-delay", type=float, default=0.8, help="Minimum simulated user delay between turns, seconds")
    parser.add_argument("--max-user-delay", type=float, default=3.5, help="Maximum simulated user delay between turns, seconds")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-turn HTTP timeout, seconds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-prefix", default=None, help="Output prefix. Default: ./vllm_ttft_<timestamp>")
    parser.add_argument("--no-health-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    levels = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]
    if not levels:
        raise ValueError("No concurrency levels supplied")

    if args.min_user_delay > args.max_user_delay:
        raise ValueError("--min-user-delay cannot be greater than --max-user-delay")

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_prefix = args.out_prefix or os.path.abspath(f"./vllm_ttft_{run_id}")

    print("vLLM voice-stack concurrency TTFT test")
    print(f"run_id:             {run_id}")
    print(f"base_url:           {args.base_url}")
    print(f"model:              {args.model}")
    print(f"concurrency levels: {levels}")
    print(f"turns/session:      random 3 to 6")
    print(f"user delay:         {args.min_user_delay:.2f}s to {args.max_user_delay:.2f}s")
    print(f"max_tokens:         {args.max_tokens}")

    if not args.no_health_check:
        await health_check(args.base_url, args.model)

    all_turns: List[TurnMetric] = []
    all_sessions: List[SessionMetric] = []

    for c in levels:
        turns, sessions = await run_concurrency_level(args, run_id, c)
        all_turns.extend(turns)
        all_sessions.extend(sessions)

        # Short cooldown to let server queues drain and GPU memory settle.
        await asyncio.sleep(2.0)

    final_ladder_summary(all_turns, all_sessions)
    write_outputs(out_prefix, all_turns, all_sessions)

    print("\nInterpretation tip:")
    print("For voice, watch TTFT p90/p95 more than mean. The practical max concurrency is usually the highest level before p95 TTFT jumps sharply or errors appear.")


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Interrupted")


if __name__ == "__main__":
    main()
