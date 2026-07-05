import time
import numpy as np

from kb.retrieve import retrieve

# List of 20 queries to evaluate warm cache-hit latencies
WARM_QUERIES = [
    "how to activate my debit card",
    "savings account interest rate eligibility",
    "personal loan application eligibility criteria",
    "rbi ombudsman complaints escalation address",
    "duplicate transaction chargeback dispute process",
    "unauthorized card block security guidelines",
    "credit card fees and annual charges info",
    "fixed deposit tenure settings premature withdrawal penalty",
    "kyc verification registered mobile PAN Aadhaar documents",
    "how do I close my bank account online",
    "how to check status of dispute ticket",
    "monthly maintenance charge threshold for savings accounts",
    "video kyc online troubleshooting registration issues",
    "how to report credit card fraud or wallet stolen",
    "compensation policy delayed credit reversals",
    "rbi grievance redressal redress timelines and TAT",
    "interest rate for high yield fixed deposits",
    "education loan interest rates repayment terms",
    "how to block cards online via web portal",
    "grievance redressal nodal officer escalation address details"
]

# List of 180 unique query phrases to evaluate cold cache-miss latencies
COLD_QUERIES = [
    f"unique query phrase number {i} to retrieve general banking information"
    for i in range(180)
]


def run_benchmark():
    print("Warming up resources (loading model and DB connection)...")
    retrieve("ping")  # Warm up the eager load caches

    cold_latencies = []
    warm_latencies = []

    # 1. Run 180 unique queries (cache misses)
    print("Executing 180 unique cold queries...")
    for q in COLD_QUERIES:
        res = retrieve(q)
        assert not res.cache_hit, "Expected cache miss for unique query phrase"
        cold_latencies.append(res.latency_ms)

    # 2. Run 20 warm queries, populating cache, then requesting them again (cache hits)
    print("Running 20 queries, populating cache, and repeating to measure hits...")
    for q in WARM_QUERIES:
        # First query (populate)
        res_cold = retrieve(q)
        cold_latencies.append(res_cold.latency_ms)

        # Subsequent queries (repeated twice to trigger cache hits)
        for _ in range(2):
            res_warm = retrieve(q)
            assert res_warm.cache_hit, "Expected cache hit for repeated query"
            warm_latencies.append(res_warm.latency_ms)

    all_latencies = cold_latencies + warm_latencies

    def print_percentile_stats(label: str, latencies: list) -> float:
        if not latencies:
            print(f"{label}: No records available.")
            return 0.0
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        print(f"{label} Latency stats (N = {len(latencies)}):")
        print(f"  p50:   {p50:.2f} ms")
        print(f"  p95:   {p95:.2f} ms")
        print(f"  p99:   {p99:.2f} ms")
        return p95

    print("\n" + "=" * 65)
    print("                 KNOWLEDGE BASE RETRIEVAL BENCHMARK")
    print("=" * 65)
    print_percentile_stats("Cold Search (Cache Misses)", cold_latencies)
    print("-" * 65)
    print_percentile_stats("Warm Search (Cache Hits)", warm_latencies)
    print("-" * 65)
    overall_p95 = print_percentile_stats("Overall retrieval", all_latencies)
    print("=" * 65)

    # Check SLA threshold (200ms)
    sla_threshold_ms = 200.0
    if overall_p95 < sla_threshold_ms:
        print(f"SUCCESS: Overall p95 retrieval latency is {overall_p95:.2f} ms (Under the {sla_threshold_ms:.0f} ms SLA).")
    else:
        print(f"WARNING: Overall p95 retrieval latency is {overall_p95:.2f} ms (EXCEEDS the {sla_threshold_ms:.0f} ms SLA!)")


if __name__ == "__main__":
    run_benchmark()
