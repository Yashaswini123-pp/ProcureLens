"""
ProcureLens ETL - Extraction Module.

Extracts procurement data from CSV files, Excel files, and relational database tables.
Includes format validation and schema checks.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sqlalchemy.engine import Engine

logger = logging.getLogger("procurelens.etl.extract")


class DataExtractor:
    """Extracts procurement data from various data sources."""

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine

    def extract_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Extract records from a CSV file into a pandas DataFrame.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            Extracted DataFrame.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid CSV.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Source file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Extracted {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV {filepath}: {str(e)}")
            raise ValueError(f"Invalid CSV file {filepath}: {str(e)}") from e

    def extract_from_excel(self, filepath: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """Extract records from an Excel spreadsheet."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Source file not found: {filepath}")

        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            logger.info(f"Extracted {len(df)} records from {filepath} (sheet: {sheet_name})")
            return df
        except Exception as e:
            logger.error(f"Failed to read Excel {filepath}: {str(e)}")
            raise ValueError(f"Invalid Excel file {filepath}: {str(e)}") from e

    def extract_from_database(self, table_name: str, engine: Optional[Engine] = None) -> pd.DataFrame:
        """
        Extract records from a database table into a pandas DataFrame.
        """
        target_engine = engine or self.engine
        if target_engine is None:
            raise ValueError("No database engine provided for extraction")

        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, target_engine)
            logger.info(f"Extracted {len(df)} records from database table '{table_name}'")
            return df
        except Exception as e:
            logger.error(f"Failed to extract from table {table_name}: {str(e)}")
            raise

    def extract_event_logs(self, engine: Optional[Engine] = None) -> pd.DataFrame:
        """Extract the full event log table."""
        return self.extract_from_database("Event_Log", engine)

    def extract_all_raw(self, raw_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """
        Extract all procurement datasets from the raw CSV directory.
        
        Returns:
            Dictionary mapping table names to DataFrames.
        """
        tables = [
            "suppliers",
            "purchase_orders",
            "invoices",
            "goods_receipts",
            "payments",
            "event_log"
        ]
        result = {}
        for table in tables:
            path = os.path.join(raw_dir, f"{table}.csv")
            if os.path.exists(path):
                result[table] = self.extract_from_csv(path)
            else:
                logger.warning(f"File not found during batch extraction: {path}")
        return result

    @staticmethod
    def validate_file_format(filepath: str, expected_columns: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Validates file existence, format, and presence of mandatory columns.
        """
        if not os.path.exists(filepath):
            return False, f"File does not exist: {filepath}"

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in [".csv", ".xlsx", ".xls"]:
            return False, f"Unsupported file extension '{ext}'. Expected .csv, .xlsx, or .xls"

        try:
            if ext == ".csv":
                sample_df = pd.read_csv(filepath, nrows=5)
            else:
                sample_df = pd.read_excel(filepath, nrows=5)

            if expected_columns:
                missing = [col for col in expected_columns if col not in sample_df.columns]
                if missing:
                    return False, f"Missing required columns: {missing}"

            return True, f"File valid with {len(sample_df.columns)} columns"
        except Exception as e:
            return False, f"Failed to parse file header: {str(e)}"
