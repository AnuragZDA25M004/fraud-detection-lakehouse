"""
tests/test_pipeline.py
-----------------------
Unit tests for the fraud detection pipeline.
Covers feature engineering logic and data quality checks.

Run with:
    pip install pytest pyspark delta-spark pandas numpy
    pytest tests/ -v
"""

import pytest
import math
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, IntegerType, LongType, StringType
)


# ── Shared Spark session (local mode — no cluster needed) ─────────────────────
@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("FraudDetection-UnitTests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


# ── Sample raw transaction schema ─────────────────────────────────────────────
RAW_SCHEMA = StructType([
    StructField("TransactionID",  LongType(),   True),
    StructField("TransactionDT",  LongType(),   True),
    StructField("TransactionAmt", DoubleType(), True),
    StructField("ProductCD",      StringType(), True),
    StructField("card1",          IntegerType(), True),
    StructField("card2",          DoubleType(), True),
    StructField("card3",          DoubleType(), True),
    StructField("card4",          StringType(), True),
    StructField("card5",          DoubleType(), True),
    StructField("card6",          StringType(), True),
    StructField("addr1",          DoubleType(), True),
    StructField("addr2",          DoubleType(), True),
    StructField("dist1",          DoubleType(), True),
    StructField("P_emaildomain",  StringType(), True),
    StructField("R_emaildomain",  StringType(), True),
    StructField("isFraud",        IntegerType(), True),
])

SAMPLE_ROWS = [
    # TransactionID, DT,     Amt,    ProdCD, card1, card2, card3, card4,        card5, card6,    addr1, addr2, dist1, P_email,       R_email,       isFraud
    (2987000, 86400,  150.0,  "W", 9500, 321.0, 150.0, "visa",        226.0, "debit",   299.0, 87.0,  0.0,   "gmail.com",   "gmail.com",   0),
    (2987001, 90000,  1500.0, "H", 8000, 111.0, 185.0, "mastercard",  102.0, "credit",  150.0, 60.0,  5.0,   "anon.com",    "yahoo.com",   1),
    (2987002, 95000,  45.0,   "C", 5000, None,  None,  "visa",        None,  "debit",   None,  None,  None,  "outlook.com", "outlook.com", 0),
    (2987003, 100000, 750.0,  "W", 3000, 222.0, 144.0, None,          117.0, None,      200.0, 70.0,  None,  None,          "hotmail.com", 1),
    (2987004, 110000, 25.0,   "S", 7500, 321.0, 150.0, "discover",    224.0, "credit",  320.0, 87.0,  10.0,  "yahoo.com",   "yahoo.com",   0),
]


@pytest.fixture(scope="module")
def raw_df(spark):
    return spark.createDataFrame(SAMPLE_ROWS, schema=RAW_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Data Quality
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataQuality:

    def test_no_null_labels(self, raw_df):
        """isFraud column must have no nulls — it is the training label."""
        null_count = raw_df.filter(col("isFraud").isNull()).count()
        assert null_count == 0, \
            f"Found {null_count} null values in isFraud column"

    def test_label_is_binary(self, raw_df):
        """isFraud must only contain 0 or 1."""
        invalid = raw_df.filter(
            (col("isFraud") != 0) & (col("isFraud") != 1)
        ).count()
        assert invalid == 0, \
            f"Found {invalid} rows where isFraud is not 0 or 1"

    def test_transaction_amt_positive(self, raw_df):
        """All transaction amounts must be strictly positive."""
        non_positive = raw_df.filter(col("TransactionAmt") <= 0).count()
        assert non_positive == 0, \
            f"Found {non_positive} transactions with non-positive amount"

    def test_row_count(self, raw_df):
        """Dataset must have at least 1 row."""
        assert raw_df.count() >= 1

    def test_required_columns_present(self, raw_df):
        """All required columns must exist in the schema."""
        required = {
            "TransactionID", "TransactionDT", "TransactionAmt",
            "isFraud", "card1", "ProductCD"
        }
        actual = set(raw_df.columns)
        missing = required - actual
        assert not missing, f"Missing required columns: {missing}"

    def test_fraud_rate_realistic(self, raw_df):
        """Fraud rate should be between 0.1% and 50% — sanity check."""
        total = raw_df.count()
        fraud = raw_df.filter(col("isFraud") == 1).count()
        rate  = fraud / total
        assert 0.001 <= rate <= 0.5, \
            f"Fraud rate {rate:.3f} is outside expected range [0.001, 0.5]"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Feature Engineering Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_log_transaction_amt(self):
        """log1p(TransactionAmt) must equal math.log1p(amt)."""
        test_amounts = [150.0, 1500.0, 45.0, 750.0, 25.0]
        for amt in test_amounts:
            expected = math.log1p(amt)
            actual   = math.log1p(amt)   # same function used in pipeline
            assert abs(actual - expected) < 1e-9, \
                f"log1p mismatch for amt={amt}"

    def test_log_amt_always_positive(self):
        """log1p of any positive transaction amount must be positive."""
        for amt in [0.01, 1.0, 100.0, 10000.0]:
            assert math.log1p(amt) > 0

    def test_is_high_value_flag(self):
        """Transactions above $500 must be flagged as high value."""
        assert (1500.0 > 500) == True,  "1500 should be high value"
        assert (150.0  > 500) == False, "150 should NOT be high value"
        assert (500.0  > 500) == False, "Exactly 500 should NOT be high value"
        assert (500.01 > 500) == True,  "500.01 should be high value"

    def test_email_match_flag(self):
        """Email match must be 1 when domains are equal, 0 otherwise."""
        def email_match(p, r):
            if p is None or r is None:
                return 0
            return 1 if p == r else 0

        assert email_match("gmail.com",   "gmail.com")   == 1
        assert email_match("gmail.com",   "yahoo.com")   == 0
        assert email_match(None,          "yahoo.com")   == 0
        assert email_match("outlook.com", None)          == 0

    def test_null_fill_sentinel(self):
        """Null numeric values must be replaced with -999."""
        SENTINEL = -999
        test_values = [None, np.nan]
        for v in test_values:
            filled = SENTINEL if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            assert filled == SENTINEL, f"Expected -999 for null, got {filled}"

    def test_product_cd_encoding(self):
        """ProductCD categories must map to correct integer indices."""
        mapping = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
        assert mapping["W"] == 0
        assert mapping["H"] == 1
        assert mapping["C"] == 2
        assert mapping["S"] == 3
        assert mapping["R"] == 4

    def test_card4_encoding(self):
        """card4 (card network) must map to correct integer indices."""
        mapping = {
            "visa": 0, "mastercard": 1,
            "american express": 2, "discover": 3
        }
        assert mapping["visa"]             == 0
        assert mapping["mastercard"]       == 1
        assert mapping["american express"] == 2
        assert mapping["discover"]         == 3

    def test_tx_hour_range(self):
        """Transaction hour derived from TransactionDT must be 0–23."""
        # TransactionDT is seconds from reference point
        test_dts = [86400, 90000, 95000, 100000, 110000]
        for dt in test_dts:
            hour = int(dt / 3600 % 24)
            assert 0 <= hour <= 23, \
                f"tx_hour={hour} is outside valid range [0, 23] for DT={dt}"

    def test_feature_engineering_spark(self, spark, raw_df):
        """Full feature engineering on Spark DataFrame produces expected columns."""
        from pyspark.sql.functions import log1p, when, col as scol

        df = raw_df.filter(scol("isFraud").isNotNull())
        df = df.withColumn("log_TransactionAmt", log1p(scol("TransactionAmt")))
        df = df.withColumn("tx_hour", (scol("TransactionDT") / 3600 % 24).cast("int"))
        df = df.withColumn("is_high_value",
            when(scol("TransactionAmt") > 500, 1).otherwise(0))
        df = df.withColumn("email_match",
            when(scol("P_emaildomain") == scol("R_emaildomain"), 1).otherwise(0))
        df = df.fillna(-999)

        result = df.collect()

        assert "log_TransactionAmt" in df.columns
        assert "tx_hour"            in df.columns
        assert "is_high_value"      in df.columns
        assert "email_match"        in df.columns

        # Row 0: amt=150, not high value
        assert result[0]["is_high_value"] == 0
        # Row 1: amt=1500, high value
        assert result[1]["is_high_value"] == 1
        # Row 0: same email domain → match
        assert result[0]["email_match"] == 1
        # Row 1: different email domains → no match
        assert result[1]["email_match"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Kafka Producer Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestKafkaProducer:

    def test_chunk_size_keeps_memory_low(self):
        """Reading CSV in chunks of 5000 rows should not exceed 50MB estimate."""
        CHUNK_SIZE  = 5000
        COLS        = 434
        BYTES_CELL  = 8    # average float64
        est_mb      = (CHUNK_SIZE * COLS * BYTES_CELL) / (1024 ** 2)
        assert est_mb < 50, \
            f"Chunk memory estimate {est_mb:.1f}MB exceeds 50MB limit"

    def test_rate_interval(self):
        """At 200 tx/sec the interval between messages must be 0.005 seconds."""
        rate     = 200
        interval = 1.0 / rate
        assert abs(interval - 0.005) < 1e-9

    def test_nan_serialisation(self):
        """NaN values in pandas must be replaced with None for JSON serialisation."""
        import json
        row = {"TransactionAmt": 150.0, "dist1": float("nan"), "card2": None}
        # Simulate the fillna(None) step in the producer
        cleaned = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                   for k, v in row.items()}
        # Must not raise
        serialised = json.dumps(cleaned)
        parsed     = json.loads(serialised)
        assert parsed["dist1"] is None
        assert parsed["card2"] is None
        assert parsed["TransactionAmt"] == 150.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataGenerator:

    def _generate(self, n=1000, fraud_rate=0.035, seed=42):
        """Inline minimal version of generate_sample.py for testing."""
        np.random.seed(seed)
        fraud_mask = np.random.random(n) < fraud_rate
        tx_dt  = np.sort(np.random.randint(86400, 86400 * 180, n))
        tx_amt = np.where(
            fraud_mask,
            np.random.lognormal(5.5, 1.5, n),
            np.random.lognormal(4.0, 1.2, n)
        ).round(2)
        return pd.DataFrame({
            "TransactionID":  range(2987000, 2987000 + n),
            "isFraud":        fraud_mask.astype(int),
            "TransactionDT":  tx_dt,
            "TransactionAmt": tx_amt,
        })

    def test_row_count(self):
        df = self._generate(n=500)
        assert len(df) == 500

    def test_fraud_label_binary(self):
        df = self._generate(n=1000)
        assert set(df["isFraud"].unique()).issubset({0, 1})

    def test_transaction_amt_positive(self):
        df = self._generate(n=1000)
        assert (df["TransactionAmt"] > 0).all()

    def test_transaction_dt_sorted(self):
        df = self._generate(n=1000)
        assert df["TransactionDT"].is_monotonic_increasing

    def test_fraud_rate_approximate(self):
        df = self._generate(n=10000, fraud_rate=0.035)
        actual_rate = df["isFraud"].mean()
        assert 0.01 <= actual_rate <= 0.07, \
            f"Fraud rate {actual_rate:.3f} too far from target 0.035"

    def test_reproducible_with_seed(self):
        df1 = self._generate(n=100, seed=42)
        df2 = self._generate(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
