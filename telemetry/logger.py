"""
Dual Telemetry Logging Pipeline for M3O (Mechanistic Model Merger & Optimizer).

Integrates:
1. Hot Path: MongoDB Atlas real-time layer-by-layer document logging (2000ms server selection timeout).
2. Cold Path: Snowflake Data Warehouse micro-batched relational analytics.
3. Fallback/Mock Mode: Non-blocking in-memory buffering and logging when credentials are unset or unreachable.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pymongo
from pymongo.errors import PyMongoError

try:
    import snowflake.connector
    from snowflake.connector.errors import Error as SnowflakeError
except ImportError:
    snowflake = None  # type: ignore
    SnowflakeError = Exception  # type: ignore


logger = logging.getLogger("m3o.telemetry")

DEFAULT_SNOWFLAKE_TABLE = "M3O_ENGINE.TELEMETRY.LAYER_METRICS"
DEFAULT_BATCH_SIZE = 50


@dataclass
class LayerMetricRecord:
    """Canonical schema for a single layer's telemetry metrics."""
    layer_name: str
    alpha: float
    frobenius_drift: float
    cos_sim: float
    status: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_snowflake_tuple(self) -> Tuple[str, float, float, float, str, str]:
        return (
            self.layer_name,
            float(self.alpha),
            float(self.frobenius_drift),
            float(self.cos_sim),
            self.status,
            self.timestamp,
        )


class TelemetryLogger:
    """
    Dual telemetry logger supporting MongoDB Atlas (hot path) and Snowflake (cold path)
    with graceful fallback/mock capability.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        mongo_db: Optional[str] = None,
        mongo_collection: Optional[str] = None,
        snowflake_config: Optional[Dict[str, Any]] = None,
        mock_mode: bool = False,
        auto_flush_threshold: int = DEFAULT_BATCH_SIZE,
    ):
        self.mock_mode = mock_mode or os.getenv("M3O_TELEMETRY_MOCK", "").lower() in ("true", "1", "yes")
        self.auto_flush_threshold = auto_flush_threshold

        # In-memory cold-path buffer and mock storage
        self.snowflake_buffer: List[LayerMetricRecord] = []
        self.mongo_mock_records: List[Dict[str, Any]] = []
        self.snowflake_flushed_mock: List[Dict[str, Any]] = []
        self.logged_records: List[Dict[str, Any]] = []

        # ---------------------------------------------------------
        # 1. Hot Path: MongoDB Atlas Setup
        # ---------------------------------------------------------
        self.mongo_client: Optional[pymongo.MongoClient] = None
        self.mongo_col: Optional[Any] = None
        self.is_mongo_active = False

        resolved_mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        self.mongo_db_name = mongo_db or os.getenv("MONGODB_DATABASE", "m3o_engine")
        self.mongo_col_name = mongo_collection or os.getenv("MONGODB_COLLECTION", "layer_metrics")

        if not self.mock_mode and resolved_mongo_uri:
            try:
                # 2000ms server selection timeout as required
                self.mongo_client = pymongo.MongoClient(
                    resolved_mongo_uri,
                    serverSelectionTimeoutMS=2000,
                    connectTimeoutMS=2000,
                )
                # Ping test to verify connectivity within timeout window
                self.mongo_client.admin.command("ping")
                self.mongo_col = self.mongo_client[self.mongo_db_name][self.mongo_col_name]
                self.is_mongo_active = True
                logger.info("Connected to MongoDB Atlas: %s.%s", self.mongo_db_name, self.mongo_col_name)
            except Exception as e:
                logger.warning("MongoDB Atlas connection failed; falling back to mock mode. Reason: %s", e)
                self.is_mongo_active = False
        else:
            self.is_mongo_active = False

        # ---------------------------------------------------------
        # 2. Cold Path: Snowflake Setup
        # ---------------------------------------------------------
        self.snowflake_conn: Optional[Any] = None
        self.is_snowflake_active = False
        self.snowflake_config = snowflake_config or self._load_snowflake_env()

        if not self.mock_mode and self.snowflake_config and self.snowflake_config.get("account"):
            try:
                if snowflake is not None:
                    db_name = self.snowflake_config.get("database", "M3O_ENGINE")
                    schema_name = self.snowflake_config.get("schema", "TELEMETRY")

                    auth_params: Dict[str, Any] = {
                        "account": self.snowflake_config.get("account"),
                        "warehouse": self.snowflake_config.get("warehouse", "COMPUTE_WH"),
                        "role": self.snowflake_config.get("role"),
                    }
                    if self.snowflake_config.get("host"):
                        auth_params["host"] = self.snowflake_config.get("host")

                    token = self.snowflake_config.get("token")
                    if token:
                        auth_params["authenticator"] = "programmatic_access_token"
                        auth_params["token"] = token
                    elif self.snowflake_config.get("password"):
                        auth_params["user"] = self.snowflake_config.get("user")
                        auth_params["password"] = self.snowflake_config.get("password")

                    self.snowflake_conn = snowflake.connector.connect(**auth_params)
                    init_cur = self.snowflake_conn.cursor()
                    wh_name = self.snowflake_config.get("warehouse", "SNOWFLAKE_LEARNING_WH")
                    # Quote identifiers to prevent SQL injection
                    db_quoted = self._quote_identifier(db_name)
                    schema_quoted = self._quote_identifier(schema_name)
                    wh_quoted = self._quote_identifier(wh_name)
                    try:
                        init_cur.execute(f"USE WAREHOUSE {wh_quoted};")
                    except Exception:
                        pass
                    init_cur.execute(f"CREATE DATABASE IF NOT EXISTS {db_quoted};")
                    init_cur.execute(f"CREATE SCHEMA IF NOT EXISTS {db_quoted}.{schema_quoted};")
                    init_cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {db_quoted}.{schema_quoted}.LAYER_METRICS (
                            LAYER_NAME VARCHAR,
                            ALPHA FLOAT,
                            FROBENIUS_DRIFT FLOAT,
                            COS_SIM FLOAT,
                            STATUS VARCHAR,
                            TIMESTAMP TIMESTAMP_NTZ
                        );
                    """)
                    init_cur.execute(f"USE DATABASE {db_quoted};")
                    init_cur.execute(f"USE SCHEMA {schema_quoted};")
                    init_cur.close()

                    self.is_snowflake_active = True
                    logger.info("Connected to Snowflake Data Warehouse: %s", self.snowflake_config.get("account"))
            except Exception as e:
                logger.warning("Snowflake connection failed; falling back to mock mode. Reason: %s", e)
                self.is_snowflake_active = False
        else:
            self.is_snowflake_active = False

    def _quote_identifier(self, identifier: str) -> str:
        """Quote a Snowflake identifier to prevent SQL injection.
        
        Snowflake uses double quotes for identifiers. Any double quotes
        within the identifier are escaped by doubling them.
        """
        # Escape any existing double quotes by doubling them
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _load_snowflake_env(self) -> Optional[Dict[str, Any]]:
        """Load Snowflake credentials from environment if available."""
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        if not account:
            return None
        return {
            "account": account,
            "host": os.getenv("SNOWFLAKE_HOST", f"{account}.snowflakecomputing.com"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "token": os.getenv("SNOWFLAKE_TOKEN") or os.getenv("SNOWFLAKE_PAT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "M3O_ENGINE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "TELEMETRY"),
            "table": os.getenv("SNOWFLAKE_TABLE", DEFAULT_SNOWFLAKE_TABLE),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        }

    def log_layer_metric(
        self,
        layer_name: str,
        alpha: float,
        frobenius_drift: float,
        cos_sim: float,
        status: str,
        timestamp: Optional[str] = None,
    ) -> LayerMetricRecord:
        """
        Ingest a single layer metric.
        - Hot Path: Dispatches to MongoDB Atlas (fail-safe).
        - Cold Path: Appends to in-memory snowflake_buffer.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        record = LayerMetricRecord(
            layer_name=str(layer_name),
            alpha=float(alpha),
            frobenius_drift=float(frobenius_drift),
            cos_sim=float(cos_sim),
            status=str(status),
            timestamp=ts,
        )

        # Hot Path: MongoDB Atlas
        payload = record.to_dict()
        self.logged_records.append(payload)
        if self.is_mongo_active and self.mongo_col is not None:
            try:
                self.mongo_col.insert_one(payload)
            except (PyMongoError, Exception) as e:
                logger.warning("Hot path MongoDB insert error for layer '%s': %s (storing in mock buffer)", layer_name, e)
                self.mongo_mock_records.append(payload)
        else:
            self.mongo_mock_records.append(payload)

        # Cold Path: Append to Snowflake buffer
        self.snowflake_buffer.append(record)

        # Micro-batch trigger
        if len(self.snowflake_buffer) >= self.auto_flush_threshold:
            self.flush_to_snowflake()

        return record

    def flush_to_snowflake(self, target_table: str = DEFAULT_SNOWFLAKE_TABLE) -> int:
        """
        Cold Path: Flush the in-memory snowflake_buffer into Snowflake table.
        Executes micro-batch INSERT query.
        Returns the number of flushed records.
        """
        if not self.snowflake_buffer:
            return 0

        pending_records = list(self.snowflake_buffer)
        record_count = len(pending_records)

        if self.is_snowflake_active and self.snowflake_conn is not None:
            # Quote table name to prevent SQL injection
            # Handle schema.table format (e.g., "M3O_ENGINE.TELEMETRY.LAYER_METRICS")
            if "." in target_table:
                parts = target_table.split(".")
                quoted_table = ".".join(self._quote_identifier(p) for p in parts)
            else:
                quoted_table = self._quote_identifier(target_table)
            
            query = f"""
            INSERT INTO {quoted_table} (
                LAYER_NAME,
                ALPHA,
                FROBENIUS_DRIFT,
                COS_SIM,
                STATUS,
                TIMESTAMP
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            data_tuples = [r.to_snowflake_tuple() for r in pending_records]
            try:
                cursor = self.snowflake_conn.cursor()
                cursor.executemany(query, data_tuples)
                self.snowflake_conn.commit()
                cursor.close()
                self.snowflake_buffer.clear()
                logger.info("Flushed %d layer metrics to Snowflake table %s", record_count, target_table)
                return record_count
            except (SnowflakeError, Exception) as e:
                logger.error("Snowflake micro-batch insert error: %s (falling back to mock store)", e)
                for r in pending_records:
                    self.snowflake_flushed_mock.append(r.to_dict())
                self.snowflake_buffer.clear()
                return record_count
        else:
            # Fallback/mock mode
            for r in pending_records:
                self.snowflake_flushed_mock.append(r.to_dict())
            self.snowflake_buffer.clear()
            logger.debug("Flushed %d records to mock Snowflake storage", record_count)
            return record_count

    def close(self) -> None:
        """Flush remaining cold-path records and cleanly close active connections."""
        if self.snowflake_buffer:
            self.flush_to_snowflake()

        if self.mongo_client is not None:
            try:
                self.mongo_client.close()
            except Exception:
                pass
            self.mongo_client = None
            self.is_mongo_active = False

        if self.snowflake_conn is not None:
            try:
                self.snowflake_conn.close()
            except Exception:
                pass
            self.snowflake_conn = None
            self.is_snowflake_active = False

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
