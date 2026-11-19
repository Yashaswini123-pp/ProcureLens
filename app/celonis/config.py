"""
ProcureLens Celonis Configuration Module.

Manages connection parameters and API credentials for Celonis Process Intelligence platform.
Handles environment variables safely without hardcoding secrets.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()


class CelonisConfig:
    """Manages Celonis connection parameters and API tokens."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_token: Optional[str] = None,
        data_pool_id: Optional[str] = None,
        data_model_id: Optional[str] = None,
        key_type: str = "APP_KEY"
    ):
        self.url = (url or os.environ.get("CELONIS_URL", "")).rstrip("/")
        self.api_token = api_token or os.environ.get("CELONIS_API_TOKEN", "")
        self.data_pool_id = data_pool_id or os.environ.get("CELONIS_DATA_POOL_ID", "")
        self.data_model_id = data_model_id or os.environ.get("CELONIS_DATA_MODEL_ID", "")
        self.key_type = key_type

    def is_configured(self) -> bool:
        """Checks whether mandatory credentials are provided."""
        return bool(self.url and self.api_token and not self.url.startswith("https://your-team"))

    def validate_config(self) -> List[str]:
        """Returns a list of missing configuration fields."""
        missing = []
        if not self.url or "your-team" in self.url:
            missing.append("CELONIS_URL (e.g. https://<team>.<realm>.celonis.cloud)")
        if not self.api_token or "your_api_token" in self.api_token:
            missing.append("CELONIS_API_TOKEN")
        if not self.data_pool_id:
            missing.append("CELONIS_DATA_POOL_ID (optional for discovery, required for push)")
        if not self.data_model_id:
            missing.append("CELONIS_DATA_MODEL_ID (required for PQL execution)")
        return missing

    def get_connection_status(self) -> Dict:
        """Returns non-sensitive status metadata for dashboard display."""
        configured = self.is_configured()
        masked_token = (
            f"{self.api_token[:4]}...{self.api_token[-4:]}"
            if self.api_token and len(self.api_token) > 8
            else "Not Provided"
        )
        return {
            "is_configured": configured,
            "target_url": self.url if configured else "Not Configured",
            "token_masked": masked_token,
            "data_pool_id": self.data_pool_id or "Not Configured",
            "data_model_id": self.data_model_id or "Not Configured",
            "missing_parameters": self.validate_config()
        }
