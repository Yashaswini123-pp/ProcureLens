"""
ProcureLens ETL - Pipeline Orchestrator.

Orchestrates the entire Extract, Transform, Load lifecycle.
Provides robust execution reports, timing metrics, and error containment.
"""

import io
import logging
import os
import time
from typing import Dict, Optional, Tuple, Union
import pandas as pd
from sqlalchemy.engine import Engine

from app.database.connection import get_engine, init_db
from app.etl.extract import DataExtractor
from app.etl.transform import DataTransformer
from app.etl.load import DataLoader

logger = logging.getLogger("procurelens.etl.pipeline")


class ETLPipeline:
    """Enterprise ETL orchestrator for ProcureLens."""

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or get_engine()
        self.extractor = DataExtractor(self.engine)
        self.transformer = DataTransformer()
        self.loader = DataLoader(self.engine)

    def run(self, raw_dir: str = "data/raw", recreate_schema: bool = False) -> Dict:
        """
        Execute full ETL pipeline from raw data directory.
        
        Args:
            raw_dir: Directory containing raw CSV files.
            recreate_schema: Whether to recreate database tables.
            
        Returns:
            Dictionary containing ETL execution metrics and audit summary.
        """
        start_time = time.time()
        logger.info(f"Starting ProcureLens ETL run from '{raw_dir}'")

        if recreate_schema:
            init_db(self.engine)

        # 1. EXTRACT
        raw_data = self.extractor.extract_all_raw(raw_dir)
        if not raw_data:
            return {
                "status": "error",
                "message": f"No raw CSV datasets found in '{raw_dir}'",
                "duration_seconds": round(time.time() - start_time, 2)
            }

        # 2. TRANSFORM
        transformed_data = self.transformer.transform_all(raw_data)

        # 3. LOAD
        load_summary = {}
        # Ensure correct insertion order respecting foreign keys
        ordered_tables = [
            ("suppliers", "Suppliers"),
            ("purchase_orders", "Purchase_Orders"),
            ("invoices", "Invoices"),
            ("goods_receipts", "Goods_Receipts"),
            ("payments", "Payments"),
            ("event_log", "Event_Log")
        ]

        total_loaded = 0
        for key, table_name in ordered_tables:
            if key in transformed_data:
                df = transformed_data[key]
                # If recreating schema or empty table, append. Otherwise upsert or append.
                rows = self.loader.load_to_database(df, table_name, if_exists="append")
                load_summary[table_name] = rows
                total_loaded += rows

        duration = round(time.time() - start_time, 2)
        report = {
            "status": "success",
            "duration_seconds": duration,
            "total_records_loaded": total_loaded,
            "table_breakdown": load_summary,
            "transformation_audit": self.transformer.report.summary()
        }

        self.loader.generate_processing_report(report)
        logger.info(f"ETL completed in {duration}s. {total_loaded} records loaded.")
        return report

    def run_from_upload(
        self,
        file_content: Union[bytes, io.BytesIO],
        filename: str,
        target_table: str
    ) -> Dict:
        """
        Process an ad-hoc uploaded procurement CSV or Excel file.
        """
        start_time = time.time()
        try:
            if isinstance(file_content, bytes):
                file_obj = io.BytesIO(file_content)
            else:
                file_obj = file_content

            if filename.endswith(".csv"):
                df = pd.read_csv(file_obj)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_obj)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported file type for {filename}. Must be .csv or .xlsx",
                    "duration_seconds": round(time.time() - start_time, 2)
                }

            # Standardize and clean single table
            clean_df = self.transformer.handle_missing_values(df, table_name=target_table)
            clean_df, removed = self.transformer.remove_duplicates(clean_df, table_name=target_table)

            rows_loaded = self.loader.load_to_database(clean_df, target_table, if_exists="append")

            return {
                "status": "success",
                "filename": filename,
                "target_table": target_table,
                "rows_processed": len(df),
                "rows_loaded": rows_loaded,
                "duplicates_removed": removed,
                "duration_seconds": round(time.time() - start_time, 2)
            }
        except Exception as e:
            logger.error(f"Upload ETL failed: {str(e)}")
            return {
                "status": "error",
                "filename": filename,
                "error": str(e),
                "duration_seconds": round(time.time() - start_time, 2)
            }
