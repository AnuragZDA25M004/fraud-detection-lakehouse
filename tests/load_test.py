#!/usr/bin/env python3
"""
load_test.py — 10× Throughput Load Test for Fraud Detection BentoML API
Z5008 Big Data Lab · IIT Madras Zanzibar 2026

Tests the /predict endpoint under 10× normal load using concurrent threads.
Generates a summary report with latency percentiles and throughput metrics.

Usage:
    pip install requests
    python load_test.py --url http://localhost:3001 --workers 10 --requests 500
"""

import argparse
import concurrent.futures
import json
import random
import statistics
import time
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Sample transaction payloads (varied to avoid caching artefacts)
# ---------------------------------------------------------------------------
SAMPLE_TRANSACTIONS = [
    {
        "TransactionAmt": 1500.0, "card1": 9500, "card2": 321.0,
        "card3": 150.0, "card4": 0, "card5": 226.0, "card6": 1,
        "addr1": 299.0, "addr2": 87.0, "dist1": 0.0,
        "ProductCD": 0, "tx_hour": 2, "is_high_value": 1, "email_match": 0
    },
    {
        "TransactionAmt": 29.95, "card1": 4120, "card2": 111.0,
        "card3": 150.0, "card4": 1, "card5": 102.0, "card6": 0,
        "addr1": 325.0, "addr2": 87.0, "dist1": 10.0,
        "ProductCD": 1, "tx_hour": 14, "is_high_value": 0, "email_match": 1
    },
    {
        "TransactionAmt": 750.0, "card1": 7700, "card2": 200.0,
        "card3": 117.0, "card4": 2, "card5": 142.0, "card6": 1,
        "addr1": 204.0, "addr2": 87.0, "dist1": 0.0,
        "ProductCD": 2, "tx_hour": 22, "is_high_value": 1, "email_match": 0
    },
    {
        "TransactionAmt": 12.50, "card1": 2300, "card2": 400.0,
        "card3": 185.0, "card4": 1, "card5": 226.0, "card6": 0,
        "addr1": 440.0, "addr2": 96.0, "dist1": 50.0,
        "ProductCD": 3, "tx_hour": 9, "is_high_value": 0, "email_match": 1
    },
    {
        "TransactionAmt": 5000.0, "card1": 14000, "card2": 555.0,
        "card3": 150.0, "card4": 3, "card5": 102.0, "card6": 1,
        "addr1": 390.0, "addr2": 87.0, "dist1": 0.0,
        "ProductCD": 0, "tx_hour": 3, "is_high_value": 1, "email_match": 0
    },
]


def make_prediction(base_url: str, session: requests.Session) -> dict:
    """Send a single /predict request and return timing + result info."""
    payload = random.choice(SAMPLE_TRANSACTIONS).copy()
    # Slightly randomise the amount so each request is unique
    payload["TransactionAmt"] *= random.uniform(0.9, 1.1)

    start = time.perf_counter()
    try:
        resp = session.post(
            f"{base_url}/predict",
            json=payload,
            timeout=10,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "success": resp.status_code == 200,
            "status_code": resp.status_code,
            "latency_ms": elapsed_ms,
            "response": resp.json() if resp.status_code == 200 else None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "success": False,
            "status_code": None,
            "latency_ms": elapsed_ms,
            "error": str(exc),
        }


def run_load_test(base_url: str, num_workers: int, total_requests: int) -> None:
    """Run the load test with concurrent workers and print a full report."""

    print("=" * 65)
    print("  Fraud Detection API — 10× Load Test")
    print(f"  Target : {base_url}/predict")
    print(f"  Workers: {num_workers}  |  Total requests: {total_requests}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Warm-up: single request to ensure the container is ready
    print("\n[*] Warm-up request...")
    warmup_session = requests.Session()
    warmup = make_prediction(base_url, warmup_session)
    if not warmup["success"]:
        print(f"    ⚠  Warm-up failed ({warmup.get('error', warmup.get('status_code'))})")
        print("    Make sure the BentoML API is running: docker run -p 3001:3000 fraud-api")
    else:
        print(f"    ✓  Warm-up OK — {warmup['latency_ms']:.1f} ms  → {warmup['response']}")

    # Main load test
    print(f"\n[*] Sending {total_requests} requests across {num_workers} workers…\n")
    results = []
    test_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Each worker has its own session (connection pool)
        sessions = [requests.Session() for _ in range(num_workers)]

        futures = [
            pool.submit(make_prediction, base_url, sessions[i % num_workers])
            for i in range(total_requests)
        ]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % max(1, total_requests // 10) == 0:
                pct = completed / total_requests * 100
                print(f"    Progress: {completed}/{total_requests}  ({pct:.0f}%)", flush=True)

    wall_time_s = time.perf_counter() - test_start

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    successes   = [r for r in results if r["success"]]
    failures    = [r for r in results if not r["success"]]
    latencies   = [r["latency_ms"] for r in successes]

    throughput  = len(successes) / wall_time_s  # requests per second
    success_pct = len(successes) / total_requests * 100

    fraud_labels = [r["response"]["label"] for r in successes if r.get("response")]
    fraud_count  = fraud_labels.count("FRAUD")
    legit_count  = fraud_labels.count("LEGITIMATE")

    print("\n" + "=" * 65)
    print("  LOAD TEST RESULTS")
    print("=" * 65)
    print(f"  Total requests   : {total_requests}")
    print(f"  Successful       : {len(successes)}  ({success_pct:.1f}%)")
    print(f"  Failed           : {len(failures)}")
    print(f"  Wall-clock time  : {wall_time_s:.2f}s")
    print(f"  Throughput       : {throughput:.1f} req/s")
    print()

    if latencies:
        print("  Latency (ms) ─────────────────────────────────")
        print(f"    Min    : {min(latencies):.1f}")
        print(f"    Median : {statistics.median(latencies):.1f}")
        print(f"    Mean   : {statistics.mean(latencies):.1f}")
        print(f"    P95    : {sorted(latencies)[int(0.95 * len(latencies))]:.1f}")
        print(f"    P99    : {sorted(latencies)[int(0.99 * len(latencies))]:.1f}")
        print(f"    Max    : {max(latencies):.1f}")
        print()

    if fraud_labels:
        print("  Prediction distribution ─────────────────────")
        print(f"    FRAUD      : {fraud_count}  ({fraud_count/len(fraud_labels)*100:.1f}%)")
        print(f"    LEGITIMATE : {legit_count}  ({legit_count/len(fraud_labels)*100:.1f}%)")
        print()

    if failures:
        print("  Sample failures ─────────────────────────────")
        for f in failures[:3]:
            print(f"    • {f.get('error', f.get('status_code'))}")
        print()

    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"url": base_url, "workers": num_workers, "requests": total_requests},
        "summary": {
            "total": total_requests,
            "successful": len(successes),
            "failed": len(failures),
            "success_rate_pct": round(success_pct, 2),
            "wall_time_s": round(wall_time_s, 3),
            "throughput_rps": round(throughput, 2),
        },
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * len(latencies))], 2) if latencies else None,
            "p99": round(sorted(latencies)[int(0.99 * len(latencies))], 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "predictions": {"FRAUD": fraud_count, "LEGITIMATE": legit_count},
    }

    report_path = "load_test_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"  Full report saved → {report_path}")
    print("=" * 65)

    # Pass/fail verdict
    print()
    if success_pct >= 99.0 and throughput >= 10.0:
        print("  ✅  PASS — API handles 10× load with ≥99% success rate")
    else:
        print("  ⚠   REVIEW — check failure details above")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="10× load test for the Fraud Detection BentoML API"
    )
    parser.add_argument(
        "--url", default="http://localhost:3001",
        help="Base URL of the BentoML API (default: http://localhost:3001)"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of concurrent worker threads (default: 10 = 10× normal)"
    )
    parser.add_argument(
        "--requests", type=int, default=500,
        help="Total number of requests to send (default: 500)"
    )
    args = parser.parse_args()

    run_load_test(args.url, args.workers, args.requests)
