"""
ProcureLens ETL - Loading Module.

Loads validated DataFrames into SQLite database via SQLAlchemy.
Supports full table reload, batch appending, upserting, and failed record logging.
"""

import json
import logging
import os
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger("procurelens.etl.load")


class DataLoader:
    """Loads validated procurement datasets into the target relational database."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def load_to_database(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        chunksize: int = 1000
    ) -> int:
        """
        Load a DataFrame into the target database table.
        
        Args:
            df: Validated DataFrame to insert.
            table_name: Target table name.
            if_exists: 'append', 'replace', or 'fail'.
            chunksize: Batch size for chunked writing.
            
        Returns:
            Number of rows loaded.
        """
        if df.empty:
            logger.warning(f"No records to load into '{table_name}'")
            return 0

        try:
            # Map logical names to exact DB table names if needed
            table_map = {
                "suppliers": "Suppliers",
                "purchase_orders": "Purchase_Orders",
                "invoices": "Invoices",
                "payments": "Payments",
                "goods_receipts": "Goods_Receipts",
                "event_log": "Event_Log"
            }
            target_table = table_map.get(table_name.lower(), table_name)

            df.to_sql(
                name=target_table,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
                method="multi"
            )
            logger.info(f"Successfully loaded {len(df)} records into '{target_table}'")
            return len(df)
        except Exception as e:
            logger.error(f"Failed to load records into '{table_name}': {str(e)}")
            raise

    def upsert_records(
        self,
        df: pd.DataFrame,
        table_name: str,
        key_column: str
    ) -> int:
        """
        Upsert (insert or update on primary key) records in SQLite using INSERT OR REPLACE.
        """
        if df.empty:
            return 0

        table_map = {
            "suppliers": "Suppliers",
            "purchase_orders": "Purchase_Orders",
            "invoices": "Invoices",
            "payments": "Payments",
            "goods_receipts": "Goods_Receipts",
            "event_log": "Event_Log"
        }
        target_table = table_map.get(table_name.lower(), table_name)

        # Convert DataFrame to records
        cols = list(df.columns)
        col_names = ", ".join(cols)
        placeholders = ", ".join([f":{col}" for col in cols])
        sql = f"INSERT OR REPLACE INTO {target_table} ({col_names}) VALUES ({placeholders})"

        records = df.to_dict(orient="records")
        # Format datetime columns to ISO strings for SQLite
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, pd.Timestamp):
                    rec[k] = v.isoformat()
                elif pd.isna(v):
                    rec[k] = None

        with self.engine.begin() as conn:
            conn.execute(text(sql), records)

        logger.info(f"Upserted {len(records)} records into '{target_table}' on key '{key_column}'")
        return len(records)

    def log_failed_records(
        self,
        failed_df: pd.DataFrame,
        reason: str,
        output_dir: str = "data/processed/rejected"
    ) -> str:
        """
        Saves rejected or invalid records to a quarantine CSV file for audit and triage.
        """
        if failed_df.empty:
            return ""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        sanitized_reason = "".join(c if c.isalnum() else "_" for c in reason)[:30]
        filename = f"failed_{sanitized_reason}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        failed_copy = failed_df.copy()
        failed_copy["Rejection_Reason"] = reason
        failed_copy["Rejection_Timestamp"] = pd.Timestamp.utcnow().isoformat()
        failed_copy.to_csv(filepath, index=False)

        logger.warning(f"Quarantined {len(failed_df)} failed records to {filepath}")
        return filepath

    def generate_processing_report(
        self,
        report_data: Dict,
        output_file: str = "data/processed/etl_latest_report.json"
    ) -> str:
        """Saves a JSON execution report of the ETL cycle."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)
        return output_file
