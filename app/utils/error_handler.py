from typing import Dict, Any
from fastapi import HTTPException

class DataSparkError(Exception):
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

def handle_database_error(error: Exception) -> DataSparkError:
    return DataSparkError(
        message="Database operation failed",
        error_code="DB_ERROR",
        details={"original_error": str(error)}
    )
