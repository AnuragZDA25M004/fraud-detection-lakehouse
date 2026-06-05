"""
kafka_producer.py
-----------------
Replays the IEEE-CIS Fraud Detection dataset (train_transaction.csv)
into a Kafka topic row-by-row, simulating a live payment stream.

Usage:
    pip install kafka-python pandas
    python kafka_producer.py \
        --csv path/to/train_transaction.csv \
        --topic fraud-transactions \
        --rate 200
"""

import argparse
import json
import time
import pandas as pd
from kafka import KafkaProducer

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="IEEE-CIS Kafka producer")
parser.add_argument("--csv",    default="train_transaction.csv", help="Path to CSV file")
parser.add_argument("--topic",  default="fraud-transactions",    help="Kafka topic name")
parser.add_argument("--broker", default="localhost:29092",       help="Kafka broker address")
parser.add_argument("--rate",   type=int, default=200,           help="Messages per second")
parser.add_argument("--limit",  type=int, default=0,             help="Max rows to send (0 = all)")
args = parser.parse_args()

# ── Kafka producer ────────────────────────────────────────────────────────────
producer = KafkaProducer(
    bootstrap_servers=args.broker,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=5,          # small batching delay for throughput
    acks=1,
)

print(f"[INFO] Loading {args.csv} in chunks (memory-safe mode)...")

interval = 1.0 / args.rate
sent     = 0
CHUNK    = 5000   # rows per chunk — keeps RAM usage under ~50MB

start = time.time()
print(f"[INFO] Streaming → topic '{args.topic}' at {args.rate} tx/sec")
print(f"[INFO] Broker: {args.broker}")
print("-" * 60)

for chunk in pd.read_csv(args.csv, chunksize=CHUNK):
    # Fill NaN with None so JSON serialiser doesn't choke
    chunk = chunk.where(pd.notnull(chunk), None)

    for _, row in chunk.iterrows():
        record = row.to_dict()
        producer.send(args.topic, value=record)
        sent += 1

        if sent % 1000 == 0:
            elapsed = time.time() - start
            actual_rate = sent / elapsed if elapsed > 0 else 0
            print(f"[INFO] Sent {sent:,} messages | "
                  f"rate={actual_rate:.0f} tx/sec")

        time.sleep(interval)

        if args.limit > 0 and sent >= args.limit:
            break

    if args.limit > 0 and sent >= args.limit:
        break

producer.flush()
producer.close()

elapsed = time.time() - start
print("-" * 60)
print(f"[DONE] Sent {sent:,} messages in {elapsed:.1f}s "
      f"(avg {sent/elapsed:.0f} tx/sec)")
