from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class DataSource(BaseModel):
    name: str
    type: str  # "erp", "saas", "file", etc.
    connection_details: Dict
    credentials: Optional[Dict]

class Dashboard(BaseModel):
    name: str
    description: Optional[str]
    widgets: List[Dict]
    refresh_interval: Optional[int]
    filters: Optional[List[Dict]]

class Report(BaseModel):
    name: str
    description: Optional[str]
    data_sources: List[str]
    metrics: List[str]
    visualization_type: str
    schedule: Optional[Dict]
    filters: Optional[List[Dict]]

class AnalysisRequest(BaseModel):
    data_source: str
    metrics: List[str]
    time_range: Optional[Dict]
    filters: Optional[List[Dict]]

class PredictionRequest(BaseModel):
    target_variable: str
    features: List[str]
    horizon: int
    confidence_interval: float = 0.95

class BudgetForecastRequest(BaseModel):
    data: Dict
    config: Dict = {
        "periods": 12,
        "target_variables": ["revenue", "cost"],
        "growth_assumptions": {"revenue": 0.05, "cost": 0.03},
        "seasonality": True,
        "include_historical": True
    }

# New models for access control
class Permission(BaseModel):
    """Permission model defining access levels to resources"""
    id: str
    name: str
    description: Optional[str] = None
    resource_type: str  # 'module', 'department', 'analytics', etc.
    access_level: str  # 'read', 'write', 'admin'

class Role(BaseModel):
    """Role model defining user roles in the system with hierarchical access"""
    id: str
    name: str
    description: Optional[str] = None
    permissions: List[str]  # List of permission IDs
    tier: int = 1  # Access tier level (1: Department, 2: Cross-department, 3: Executive)
    
    class Config:
        schema_extra = {
            "example": {
                "id": "sales_manager",
                "name": "Sales Manager",
                "description": "Department head for sales",
                "permissions": ["view_sales_data", "edit_sales_reports"],
                "tier": 1
            }
        }

class User(BaseModel):
    """User model with authentication and role information"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    departments: List[str] = []
    roles: List[str] = []
    is_active: bool = True
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    
    class Config:
        schema_extra = {
            "example": {
                "id": "user123",
                "username": "johndoe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "departments": ["sales"],
                "roles": ["sales_manager"],
                "is_active": True
            }
        }

class DepartmentModule(BaseModel):
    """Model representing department and module relationships"""
    department_id: str
    department_name: str
    modules: List[str]
    parent_department: Optional[str] = None
    metadata: Optional[Dict] = None
