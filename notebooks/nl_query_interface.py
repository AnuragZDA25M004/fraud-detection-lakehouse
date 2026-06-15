"""
nl_query/nl_query_interface.py
-------------------------------
Natural-language query interface for the fraud detection lakehouse.

Flow:
  User types plain English
       ↓
  Llama 3 (via Ollama) translates to Spark SQL
       ↓
  Safety validator (read-only, injection prevention)
       ↓
  Spark executes on Delta Lake tables
       ↓
  Results returned as a pandas DataFrame

Usage (standalone):
    python nl_query_interface.py

Usage (as a module):
    from nl_query_interface import NLQueryInterface
    nq = NLQueryInterface()
    result = nq.query("Show me the top 5 highest fraud transactions")
    print(result)
"""

import os
import re
import requests
import pandas as pd
from pyspark.sql import SparkSession

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL       = os.getenv("OLLAMA_URL",       "http://ollama:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3")
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "bigdata123")

# ── Table registry — only these tables can be queried ─────────────────────────
ALLOWED_TABLES = {
    "raw_transactions":     "s3a://warehouse/raw/transactions",
    "feature_transactions": "s3a://warehouse/features/transactions",
}

# ── SQL safety rules ──────────────────────────────────────────────────────────
BLOCKED_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "CREATE", "TRUNCATE", "REPLACE", "MERGE",
    "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "--", "/*", "*/", "xp_", "sp_",
]

SYSTEM_PROMPT = f"""You are a Spark SQL expert for a fraud detection data lakehouse.

Available tables (Delta Lake on MinIO):
  - raw_transactions      : Raw streaming transactions from Kafka
      Columns: TransactionID, TransactionDT, TransactionAmt, ProductCD,
               card1, card2, card3, card4, card5, card6,
               addr1, addr2, dist1, P_emaildomain, R_emaildomain,
               isFraud, ingested_at
  - feature_transactions  : ML-ready feature-engineered table
      Columns: TransactionAmt, log_TransactionAmt, tx_hour,
               is_high_value, email_match,
               card1-card6, addr1, addr2, dist1, ProductCD, isFraud

Rules you MUST follow:
  1. Output ONLY the SQL query — no explanation, no markdown, no backticks.
  2. Always use SELECT. Never use DROP, DELETE, UPDATE, INSERT, CREATE, ALTER.
  3. Always add LIMIT 100 unless the user asks for aggregations.
  4. Use the exact table names above (no schema prefix needed).
  5. isFraud = 1 means fraud, isFraud = 0 means legitimate.
  6. tx_hour is 0-23 (hour of day). is_high_value = 1 means amount > $500.

Example:
  User: "Show me the 5 largest fraud transactions"
  SQL:  SELECT TransactionID, TransactionAmt, card4, isFraud
        FROM raw_transactions
        WHERE isFraud = 1
        ORDER BY TransactionAmt DESC
        LIMIT 5
"""


class SQLSafetyError(Exception):
    """Raised when generated SQL fails safety validation."""
    pass


class NLQueryInterface:
    """Natural-language to Spark SQL interface for the fraud lakehouse."""

    def __init__(self):
        self.spark = self._init_spark()
        self._register_tables()
        print(f"[NLQuery] Ready — model: {OLLAMA_MODEL} | tables: {list(ALLOWED_TABLES.keys())}")

    # ── Spark session ─────────────────────────────────────────────────────────
    def _init_spark(self):
        spark = (
            SparkSession.builder
            .appName("FraudDetection-NLQuery")
            .master("local[2]")
            .config("spark.jars.packages",
                    "io.delta:delta-spark_2.12:3.1.0,"
                    "org.apache.hadoop:hadoop-aws:3.3.4")
            .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.hadoop.fs.s3a.endpoint",          MINIO_ENDPOINT)
            .config("spark.hadoop.fs.s3a.access.key",        MINIO_ACCESS_KEY)
            .config("spark.hadoop.fs.s3a.secret.key",        MINIO_SECRET_KEY)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        return spark

    # ── Register Delta tables as Spark SQL views ──────────────────────────────
    def _register_tables(self):
        for table_name, path in ALLOWED_TABLES.items():
            try:
                df = self.spark.read.format("delta").load(path)
                df.createOrReplaceTempView(table_name)
                print(f"[NLQuery] Registered table: {table_name} ({df.count():,} rows)")
            except Exception as e:
                print(f"[NLQuery] Warning: could not register {table_name}: {e}")

    # ── Step 1: Translate English → SQL via Ollama ────────────────────────────
    def _translate_to_sql(self, question: str) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\nUser question: {question}\n\nSQL:",
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temperature = deterministic output
                "top_p": 0.9,
                "num_predict": 300,
            }
        }
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            sql = response.json().get("response", "").strip()

            # Strip any markdown code fences the model adds
            sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"```",    "", sql)
            sql = sql.strip()

            return sql

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_URL}. "
                "Make sure Ollama is running and the model is pulled:\n"
                "  docker exec ollama ollama pull llama3"
            )

    # ── Step 2: Validate SQL for safety ──────────────────────────────────────
    def _validate_sql(self, sql: str) -> str:
        """
        Safety checks:
        1. Must start with SELECT
        2. Must not contain any blocked DML/DDL keywords
        3. Must only reference allowed tables
        4. Must not contain semicolons (prevents stacked queries)
        """
        sql_upper = sql.upper().strip()

        # Rule 1 — must be a SELECT
        if not sql_upper.startswith("SELECT"):
            raise SQLSafetyError(
                f"Only SELECT queries are allowed. Generated SQL started with: "
                f"{sql[:30]}..."
            )

        # Rule 2 — no dangerous keywords
        for keyword in BLOCKED_KEYWORDS:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, sql_upper):
                raise SQLSafetyError(
                    f"Blocked keyword detected: '{keyword}'. "
                    "Only read-only SELECT queries are permitted."
                )

        # Rule 3 — only allowed tables
        for word in re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", sql_upper):
            for table_ref in word:
                if table_ref and table_ref not in [t.upper() for t in ALLOWED_TABLES]:
                    raise SQLSafetyError(
                        f"Unknown table referenced: '{table_ref}'. "
                        f"Allowed tables: {list(ALLOWED_TABLES.keys())}"
                    )

        # Rule 4 — no semicolons (prevents query stacking)
        if ";" in sql:
            sql = sql.split(";")[0].strip()

        # Rule 5 — enforce LIMIT if missing (prevents full table scans)
        if "LIMIT" not in sql_upper:
            sql = sql.rstrip() + "\nLIMIT 100"

        return sql

    # ── Step 3: Execute SQL on Spark ──────────────────────────────────────────
    def _execute_sql(self, sql: str) -> pd.DataFrame:
        result_df = self.spark.sql(sql)
        return result_df.toPandas()

    # ── Main public method ────────────────────────────────────────────────────
    def query(self, question: str, verbose: bool = True) -> pd.DataFrame:
        """
        Translate a plain-English question to SQL and execute it.

        Args:
            question: Natural language question about fraud transactions
            verbose:  Print the generated SQL before executing

        Returns:
            pandas DataFrame with query results
        """
        if verbose:
            print(f"\n[Question] {question}")

        # Step 1 — LLM translation
        sql = self._translate_to_sql(question)
        if verbose:
            print(f"[Generated SQL]\n{sql}\n")

        # Step 2 — Safety validation
        try:
            safe_sql = self._validate_sql(sql)
        except SQLSafetyError as e:
            print(f"[Safety] ❌ Query blocked: {e}")
            return pd.DataFrame()

        if verbose and safe_sql != sql:
            print(f"[Safe SQL]\n{safe_sql}\n")

        # Step 3 — Execute
        try:
            result = self._execute_sql(safe_sql)
            if verbose:
                print(f"[Result] {len(result)} rows returned")
            return result
        except Exception as e:
            print(f"[Error] SQL execution failed: {e}")
            return pd.DataFrame()

    def stop(self):
        self.spark.stop()
        print("[NLQuery] Spark session stopped.")


# ── Interactive CLI mode ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Fraud Detection — Natural Language Query Interface")
    print("  Powered by Llama 3 (Ollama) + Apache Spark")
    print("=" * 60)
    print("Type your question in plain English. Type 'exit' to quit.\n")

    nq = NLQueryInterface()

    # Demo questions to show on startup
    demo_questions = [
        "How many fraud transactions are there in total?",
        "Show the top 5 highest value fraud transactions",
        "What is the average transaction amount for fraud vs legitimate?",
        "How many transactions happened per hour of the day?",
        "Show me all transactions above $1000 that are flagged as fraud",
    ]
    print("Example questions you can ask:")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            question = input("Your question: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break
            result = nq.query(question)
            if not result.empty:
                print(result.to_string(index=False))
            print()
        except KeyboardInterrupt:
            break

    nq.stop()
