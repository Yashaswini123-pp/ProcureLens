"""
ProcureLens Celonis Connector Module.

Provides integration bridge to Celonis Process Intelligence platform (PyCelonis / REST API).
Gracefully handles absent credentials, returns transparent connection diagnostic state,
and separates cloud execution from local PM4Py analytics.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from app.celonis.config import CelonisConfig

logger = logging.getLogger("procurelens.celonis.connector")


class CelonisConnector:
    """
    Manages communication with Celonis EMS / Cloud platform.
    When credentials are not present, operates in transparent offline/local mode.
    """

    def __init__(self, config: Optional[CelonisConfig] = None):
        self.config = config or CelonisConfig()
        self.is_connected = False
        self._celonis_instance = None
        self._data_model = None

    def connect(self) -> Dict:
        """
        Attempts connection to Celonis Cloud using PyCelonis if available and configured.
        Never fabricates a successful connection.
        """
        if not self.config.is_configured():
            return {
                "success": False,
                "status": "DISCONNECTED",
                "message": (
                    "Celonis credentials are not configured in environment variables. "
                    "ProcureLens is running in standalone local mode using PM4Py and SQLite. "
                    "To connect Celonis, set CELONIS_URL and CELONIS_API_TOKEN in your .env file."
                ),
                "details": self.config.get_connection_status()
            }

        try:
            # Check for pycelonis package
            import pycelonis
            self._celonis_instance = pycelonis.get_celonis(
                base_url=self.config.url,
                api_token=self.config.api_token,
                key_type=self.config.key_type
            )
            self.is_connected = True
            logger.info(f"Connected to Celonis team at {self.config.url}")
            return {
                "success": True,
                "status": "CONNECTED",
                "message": f"Successfully connected to Celonis platform at {self.config.url}",
                "pycelonis_version": getattr(pycelonis, "__version__", "unknown")
            }
        except ImportError:
            return {
                "success": False,
                "status": "MISSING_LIBRARY",
                "message": (
                    "The 'pycelonis' Python library is not installed. "
                    "Install with 'pip install pycelonis' to enable live cloud integration."
                )
            }
        except Exception as e:
            self.is_connected = False
            logger.warning(f"Failed to authenticate with Celonis: {str(e)}")
            return {
                "success": False,
                "status": "AUTHENTICATION_FAILED",
                "message": f"Celonis connection failed: {str(e)}",
                "hint": "Verify that your API token has permissions for the specified Data Pool / Data Model."
            }

    def push_data(self, df: pd.DataFrame, table_name: str) -> Dict:
        """
        Pushes a DataFrame into the configured Celonis Data Pool table.
        """
        if not self.is_connected or not self._celonis_instance:
            return {
                "success": False,
                "message": "Cannot push data: Celonis is not connected. Operation performed in local database only.",
                "table_name": table_name,
                "row_count": len(df)
            }

        try:
            pool = self._celonis_instance.data_integration.get_data_pools().find(self.config.data_pool_id)
            pool.push_table(table_name=table_name, df=df)
            return {
                "success": True,
                "message": f"Successfully pushed {len(df)} rows to Celonis table '{table_name}'",
                "table_name": table_name
            }
        except Exception as e:
            return {"success": False, "error": str(e), "table_name": table_name}

    def execute_pql(self, pql_query: str) -> Dict:
        """
        Submits a PQL (Process Query Language) query to the Celonis Data Model calculation engine.
        """
        if not self.is_connected:
            return {
                "success": False,
                "executed_cloud": False,
                "message": "PQL query execution requires an active Celonis Data Model connection.",
                "query": pql_query,
                "local_fallback_available": True
            }

        try:
            # Query the Celonis data model
            data_model = self._celonis_instance.data_integration.get_data_models().find(self.config.data_model_id)
            query_obj = pycelonis.pql.PQL()
            query_obj.add(pycelonis.pql.PQLColumn(query=pql_query, name="Result"))
            df_result = data_model.export_data_frame(query_obj)
            return {
                "success": True,
                "executed_cloud": True,
                "data": df_result.to_dict(orient="records"),
                "row_count": len(df_result)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": pql_query}

    def get_data_model_info(self) -> Dict:
        """Retrieves schema metadata from the Celonis Data Model."""
        if not self.is_connected:
            return {
                "connected": False,
                "mode": "LOCAL_PM4PY",
                "tables": ["Suppliers", "Purchase_Orders", "Invoices", "Payments", "Goods_Receipts", "Event_Log"]
            }
        return {"connected": True, "data_model_id": self.config.data_model_id}
