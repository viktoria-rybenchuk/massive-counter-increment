import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import hazelcast
from hazelcast.config import Config

NUM_THREADS = 10
ITERATIONS_PER_THREAD = 10_000
EXPECTED_TOTAL = NUM_THREADS * ITERATIONS_PER_THREAD
COUNTER_KEY = "counter"
MAP_NAME = "counter-map"
ATOMIC_NAME = "counter"


def increment_map_no_lock(distributed_map, key):
    current = distributed_map.get(key)
    if current is None:
        current = 0
    distributed_map.put(key, current + 1)


def increment_map_pessimistic(distributed_map, key):
    distributed_map.lock(key)
    try:
        current = distributed_map.get(key)
        if current is None:
            current = 0
        distributed_map.put(key, current + 1)
    finally:
        distributed_map.unlock(key)


def increment_map_optimistic(distributed_map, key):
    while True:
        old = distributed_map.get(key)
        if old is None:
            old = 0
        new_val = old + 1
        if distributed_map.replace_if_same(key, old, new_val):
            break


def increment_iatomiclong(atomic_long):
    atomic_long.add_and_get(1)


def run_worker_map(inc_fn, distributed_map, key, iterations):
    for _ in range(iterations):
        inc_fn(distributed_map, key)


def run_worker_atomic(atomic_long, iterations):
    for _ in range(iterations):
        increment_iatomiclong(atomic_long)


def run_map_mode(client, mode: str):
    m = client.get_map(MAP_NAME).blocking()
    m.put(COUNTER_KEY, 0)

    if mode == "map_no_lock":
        inc_fn = increment_map_no_lock
    elif mode == "map_pessimistic":
        inc_fn = increment_map_pessimistic
    elif mode == "map_optimistic":
        inc_fn = increment_map_optimistic
    else:
        raise ValueError(f"Unknown map mode: {mode}")

    print(f"Mode: {mode}")
    print(f"Starting {NUM_THREADS} threads × {ITERATIONS_PER_THREAD} iterations...")

    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(run_worker_map, inc_fn, m, COUNTER_KEY, ITERATIONS_PER_THREAD)
                for _ in range(NUM_THREADS)
            ]
            for f in futures:
                f.result()
    except Exception as e:
        print(f"Exception during execution: {e}")

    elapsed = time.perf_counter() - start

    final = m.get(COUNTER_KEY)
    ok = final == EXPECTED_TOTAL
    print(f"  Final value: {final}, expected: {EXPECTED_TOTAL} → {'✓ SUCCESS' if ok else '✗ ERROR'}")
    print(f"  Time: {elapsed:.2f} s")
    return ok


def run_iatomiclong_mode(client):
    print("Mode: iatomiclong (CP Subsystem)")
    print(f"Starting {NUM_THREADS} threads × {ITERATIONS_PER_THREAD} iterations...")

    atomic = client.cp_subsystem.get_atomic_long(ATOMIC_NAME).blocking()
    atomic.set(0)

    start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [
                executor.submit(run_worker_atomic, atomic, ITERATIONS_PER_THREAD)
                for _ in range(NUM_THREADS)
            ]
            for f in futures:
                f.result()
    except Exception as e:
        print(f"Exception during execution: {e}")

    elapsed = time.perf_counter() - start

    final = atomic.get()
    ok = final == EXPECTED_TOTAL
    print(f"  Final value: {final}, expected: {EXPECTED_TOTAL} → {'✓ SUCCESS' if ok else '✗ ERROR'}")
    print(f"  Time: {elapsed:.2f} s")
    print()
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Run distributed counter: 10 threads × 10k increments (expected 100k)."
    )
    parser.add_argument(
        "mode",
        choices=["map_no_lock", "map_pessimistic", "map_optimistic", "iatomiclong"],
        help="Counter implementation mode",
    )
    parser.add_argument(
        "--cluster",
        nargs="+",
        default=None,
        metavar="HOST:PORT",
        help="Cluster members (e.g., --cluster 127.0.0.1:5701 127.0.0.1:5702 127.0.0.1:5703)",
    )
    parser.add_argument(
        "--redo-operation",
        action="store_true",
        help="Enable redo_operation for automatic retry on failures",
    )
    args = parser.parse_args()

    # Configure client
    config = Config()
    if args.cluster:
        config.cluster_members = args.cluster
        print(f"Connecting to cluster: {args.cluster}")
    else:
        config.cluster_members = ["127.0.0.1:5701"]
        print("Connecting to default: 127.0.0.1:5701")

    if args.redo_operation:
        config.redo_operation = True
        print("✓ Redo operation ENABLED")


    client = hazelcast.HazelcastClient(config=config)

    try:
        if args.mode == "iatomiclong":
            run_iatomiclong_mode(client)
        else:
            run_map_mode(client, args.mode)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()