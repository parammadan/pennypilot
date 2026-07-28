"""Run the Kafka→S3 archival sink (PennyData cold path).

    python scripts/run_s3_sink.py --kafka localhost:9092 \
        --bucket pennydata-771965334314-us-east-2 --from-beginning
"""
import argparse

from shoprl.platform.s3_sink import archive_from_kafka

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kafka", default="localhost:9092")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--from-beginning", action="store_true")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()
    archive_from_kafka(args.bucket, args.kafka,
                       from_beginning=args.from_beginning,
                       max_events=args.max_events,
                       batch_size=args.batch_size)
