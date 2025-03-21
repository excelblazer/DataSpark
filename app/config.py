import os
from pydantic import BaseSettings
from typing import List, Optional, Dict, Any


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API settings
    APP_NAME: str = "DataBloom Analytics Engine"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Security settings
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-for-development-only")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./analytics.db")
    DATABASE_USE_SSL: bool = os.getenv("DATABASE_USE_SSL", "false").lower() == "true"
    DATABASE_SSL_CA: Optional[str] = os.getenv("DATABASE_SSL_CA")
    DATABASE_SSL_CERT: Optional[str] = os.getenv("DATABASE_SSL_CERT")
    DATABASE_SSL_KEY: Optional[str] = os.getenv("DATABASE_SSL_KEY")
    
    # Warehouse settings
    WAREHOUSE_CONNECTION_STRING: str = os.getenv("WAREHOUSE_CONNECTION_STRING", "sqlite:///./warehouse.db")
    WAREHOUSE_ENCRYPTION_KEY: str = os.getenv("WAREHOUSE_ENCRYPTION_KEY", "")
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_DATE_FORMAT: str = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
    LOG_ROTATION: bool = os.getenv("LOG_ROTATION", "true").lower() == "true"
    LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Performance settings
    ENABLE_PERFORMANCE_TRACKING: bool = os.getenv("ENABLE_PERFORMANCE_TRACKING", "true").lower() == "true"
    PERFORMANCE_TRACKING_THRESHOLD_MS: int = int(os.getenv("PERFORMANCE_TRACKING_THRESHOLD_MS", "1000"))
    
    # Feature flags
    ENABLE_CROSS_MODULE_ANALYSIS: bool = os.getenv("ENABLE_CROSS_MODULE_ANALYSIS", "true").lower() == "true"
    ENABLE_ML_FEATURES: bool = os.getenv("ENABLE_ML_FEATURES", "true").lower() == "true"
    
    # Analytics settings
    ANALYTICS_CACHE_SIZE: int = int(os.getenv("ANALYTICS_CACHE_SIZE", "128"))
    ANALYTICS_CACHE_TTL: int = int(os.getenv("ANALYTICS_CACHE_TTL", "3600"))  # 1 hour in seconds
    ANALYTICS_MAX_DATAFRAME_SIZE: int = int(os.getenv("ANALYTICS_MAX_DATAFRAME_SIZE", "1000000"))  # Max rows
    ANALYTICS_SAMPLE_SIZE: int = int(os.getenv("ANALYTICS_SAMPLE_SIZE", "10000"))  # Rows to sample for large datasets
    ANALYTICS_CORRELATION_THRESHOLD: float = float(os.getenv("ANALYTICS_CORRELATION_THRESHOLD", "0.7"))
    ANALYTICS_OUTLIER_THRESHOLD: float = float(os.getenv("ANALYTICS_OUTLIER_THRESHOLD", "3.0"))  # Z-score threshold
    
    # Classification settings
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.75"))
    CLASSIFICATION_PATTERNS_PATH: Optional[str] = os.getenv("CLASSIFICATION_PATTERNS_PATH")
    CLASSIFICATION_ENABLE_AUTO_LEARNING: bool = os.getenv("CLASSIFICATION_ENABLE_AUTO_LEARNING", "true").lower() == "true"
    DATA_MAPPER_CONFIG_PATH: str = os.getenv("DATA_MAPPER_CONFIG_PATH", "./config/data_mapper.json")
    
    # Data mapping settings
    DATA_MAPPER_ENABLE_CUSTOM_TYPES: bool = os.getenv("DATA_MAPPER_ENABLE_CUSTOM_TYPES", "true").lower() == "true"
    
    # Error handling settings
    ERROR_INCLUDE_STACK_TRACE: bool = os.getenv("ERROR_INCLUDE_STACK_TRACE", "false").lower() == "true"
    ERROR_MAX_RETRIES: int = int(os.getenv("ERROR_MAX_RETRIES", "3"))
    ERROR_RETRY_DELAY_MS: int = int(os.getenv("ERROR_RETRY_DELAY_MS", "1000"))
    
    # Data processing settings
    CLEANER_CONFIG: Dict = {
        "handle_missing": True,
        "handle_duplicates": True,
        "normalize_data": True,
        "detect_outliers": True
    }
    
    CONSOLIDATION_CONFIG: Dict = {
        "check_unique_keys": True,
        "handle_type_inconsistencies": True,
        "clean_after_consolidation": True
    }
    
    # Access Control settings
    ACCESS_CONTROL_CONFIG_PATH: str = os.getenv("ACCESS_CONTROL_CONFIG_PATH", "./config/access_control.json")
    DEFAULT_PERMISSIONS_PATH: str = os.getenv("DEFAULT_PERMISSIONS_PATH", "./config/default_permissions.json")
    DEFAULT_ROLES_PATH: str = os.getenv("DEFAULT_ROLES_PATH", "./config/default_roles.json")
    ENABLE_TIERED_ACCESS: bool = os.getenv("ENABLE_TIERED_ACCESS", "true").lower() == "true"
    
    # Demo users (for testing only - in production this would come from a secure database)
    DEMO_USERS: Dict[str, Dict[str, Any]] = {
        "sales_manager": {
            "id": "user_sales_1",
            "password": "password",
            "roles": ["sales_manager"],
            "departments": ["sales"],
        },
        "marketing_manager": {
            "id": "user_marketing_1",
            "password": "password",
            "roles": ["marketing_manager"],
            "departments": ["marketing"],
        },
        "product_manager": {
            "id": "user_product_1",
            "password": "password",
            "roles": ["product_manager"],
            "departments": ["product"],
        },
        "regional_manager": {
            "id": "user_regional_1",
            "password": "password",
            "roles": ["regional_manager"],
            "departments": ["sales", "marketing"],
        },
        "business_analyst": {
            "id": "user_analyst_1",
            "password": "password", 
            "roles": ["business_analyst"],
            "departments": ["sales", "marketing", "product"],
        },
        "executive": {
            "id": "user_exec_1",
            "password": "password",
            "roles": ["executive"],
            "departments": ["sales", "marketing", "product", "finance", "hr", "operations"],
        },
        "admin": {
            "id": "user_admin_1",
            "password": "password",
            "roles": ["admin"],
            "departments": ["sales", "marketing", "product", "finance", "hr", "operations"],
        }
    }

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings object
settings = Settings()
