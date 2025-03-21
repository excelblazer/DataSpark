from typing import Dict, List, Optional
from pydantic import BaseModel

class DataModuleDefinition(BaseModel):
    """Definition of a data module"""
    name: str
    description: str
    department: str
    key_metrics: List[str]
    typical_columns: List[str]
    related_modules: List[str] = []
    visualization_defaults: Dict = {}
    expected_update_frequency: Optional[str] = None
    
class DataModuleRegistry:
    """Registry for all data module definitions"""
    def __init__(self):
        self.modules = self._init_modules()
        
    def _init_modules(self) -> Dict[str, DataModuleDefinition]:
        """Initialize the module registry with standard definitions"""
        modules = {}
        
        # CRM Module
        modules["Customer Relationship Management"] = DataModuleDefinition(
            name="Customer Relationship Management",
            description="Data related to customer interactions, leads, and sales opportunities",
            department="Sales",
            key_metrics=[
                "conversion_rate", "lead_quality_score", "days_to_close", 
                "deal_size", "win_rate", "customer_retention_rate"
            ],
            typical_columns=[
                "customer_id", "customer_name", "lead_source", "opportunity_stage",
                "expected_close_date", "deal_value", "contact_info"
            ],
            related_modules=["Marketing & Social Media", "Customer Demographics"],
            visualization_defaults={
                "default_charts": ["pipeline_funnel", "conversion_trend", "deal_size_distribution"],
                "color_scheme": "blue"
            },
            expected_update_frequency="daily"
        )
        
        # Marketing Module
        modules["Marketing & Social Media"] = DataModuleDefinition(
            name="Marketing & Social Media",
            description="Campaign performance, channel metrics, and engagement data",
            department="Marketing",
            key_metrics=[
                "campaign_roi", "ctr", "conversion_rate", "cost_per_lead", 
                "engagement_rate", "reach", "impressions"
            ],
            typical_columns=[
                "campaign_name", "channel", "start_date", "end_date", "budget",
                "impressions", "clicks", "conversions", "engagement"
            ],
            related_modules=["Customer Relationship Management", "Website Traffic"],
            visualization_defaults={
                "default_charts": ["channel_performance", "campaign_roi", "engagement_trend"],
                "color_scheme": "green"
            },
            expected_update_frequency="daily"
        )
        
        # Add more module definitions here...
        # [Include definitions for all other modules]
        
        return modules
    
    def get_module(self, name: str) -> Optional[DataModuleDefinition]:
        """Get a module by name"""
        return self.modules.get(name)
    
    def get_department_modules(self, department: str) -> List[DataModuleDefinition]:
        """Get all modules for a specific department"""
        return [m for m in self.modules.values() if m.department == department]
    
    def get_related_modules(self, module_name: str) -> List[DataModuleDefinition]:
        """Get related modules for a given module"""
        module = self.get_module(module_name)
        if not module:
            return []
            
        return [self.get_module(name) for name in module.related_modules if name in self.modules]
    
    def get_all_modules(self) -> List[DataModuleDefinition]:
        """Get all registered modules"""
        return list(self.modules.values())
    
    def get_all_departments(self) -> List[str]:
        """Get unique list of all departments"""
        return list(set(m.department for m in self.modules.values()))
