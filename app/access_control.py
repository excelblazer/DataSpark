import logging
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
import json
from functools import lru_cache
from .models import User, Role, Permission, DepartmentModule
from .config import settings

logger = logging.getLogger("databloom.access_control")

class AccessControlError(Exception):
    """Base class for access control exceptions"""
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = f"[{datetime.now().isoformat()}] AccessControlError-{error_type}: {message}"
        super().__init__(self.message)

class AccessControlService:
    """
    Service to manage role-based access control with hierarchical permissions.
    Implements the three-tier access model:
    - Tier 1: Department level (e.g., Sales Manager) - module-specific access
    - Tier 2: Cross-department level (e.g., Regional Manager) - cross-department analytics
    - Tier 3: Executive level (e.g., C-Suite) - all departments, consolidated insights
    """
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the access control service"""
        logger.debug("Initializing AccessControlService")
        
        # Default roles and permissions
        self.permissions = self._init_default_permissions()
        self.roles = self._init_default_roles()
        self.department_modules = self._init_default_department_modules()
        self.users = {}
        
        # Load from config if provided
        if config_path:
            self._load_config(config_path)
    
    def _init_default_permissions(self) -> Dict[str, Permission]:
        """Initialize default permissions"""
        permissions = {}
        
        # Add basic permissions
        basic_resources = ["dashboard", "report", "analytics", "data"]
        access_levels = ["view", "edit", "admin"]
        
        for resource in basic_resources:
            for level in access_levels:
                perm_id = f"{level}_{resource}"
                permissions[perm_id] = Permission(
                    id=perm_id,
                    name=f"{level.capitalize()} {resource}",
                    resource_type=resource,
                    access_level=level
                )
        
        # Department specific permissions
        departments = ["sales", "marketing", "product", "finance", "hr", "operations"]
        for dept in departments:
            for level in access_levels:
                perm_id = f"{level}_{dept}_data"
                permissions[perm_id] = Permission(
                    id=perm_id,
                    name=f"{level.capitalize()} {dept.capitalize()} data",
                    resource_type="department_data",
                    access_level=level
                )
        
        # Cross-department permissions
        permissions["view_cross_department"] = Permission(
            id="view_cross_department",
            name="View Cross Department Analytics",
            resource_type="cross_department",
            access_level="view"
        )
        
        permissions["edit_cross_department"] = Permission(
            id="edit_cross_department",
            name="Edit Cross Department Analytics",
            resource_type="cross_department",
            access_level="edit"
        )
        
        # Executive permissions
        permissions["view_executive"] = Permission(
            id="view_executive",
            name="View Executive Insights",
            resource_type="executive",
            access_level="view"
        )
        
        permissions["edit_executive"] = Permission(
            id="edit_executive",
            name="Edit Executive Insights",
            resource_type="executive",
            access_level="edit"
        )
        
        return permissions
    
    def _init_default_roles(self) -> Dict[str, Role]:
        """Initialize default roles with hierarchical tiers"""
        roles = {}
        
        # Tier 1: Department-level roles
        departments = ["sales", "marketing", "product", "finance", "hr", "operations"]
        for dept in departments:
            # Department viewer
            viewer_id = f"{dept}_viewer"
            roles[viewer_id] = Role(
                id=viewer_id,
                name=f"{dept.capitalize()} Viewer",
                description=f"Can view {dept} data and analytics",
                permissions=[f"view_{dept}_data", "view_dashboard", "view_report", "view_analytics"],
                tier=1
            )
            
            # Department manager
            manager_id = f"{dept}_manager"
            roles[manager_id] = Role(
                id=manager_id,
                name=f"{dept.capitalize()} Manager",
                description=f"Can manage {dept} data and analytics",
                permissions=[
                    f"view_{dept}_data", f"edit_{dept}_data", 
                    "view_dashboard", "edit_dashboard", 
                    "view_report", "edit_report",
                    "view_analytics"
                ],
                tier=1
            )
        
        # Tier 2: Cross-department roles
        roles["regional_manager"] = Role(
            id="regional_manager",
            name="Regional Manager",
            description="Can access cross-department analytics and insights",
            permissions=[
                "view_cross_department", 
                "view_dashboard", "edit_dashboard",
                "view_report", "edit_report", 
                "view_analytics"
            ] + [f"view_{dept}_data" for dept in departments],
            tier=2
        )
        
        roles["business_analyst"] = Role(
            id="business_analyst",
            name="Business Analyst",
            description="Can analyze data across departments",
            permissions=[
                "view_cross_department", "edit_cross_department",
                "view_dashboard", "edit_dashboard", 
                "view_report", "edit_report",
                "view_analytics", "edit_analytics"
            ] + [f"view_{dept}_data" for dept in departments],
            tier=2
        )
        
        # Tier 3: Executive roles
        roles["executive"] = Role(
            id="executive",
            name="Executive",
            description="Executive with access to all insights and analytics",
            permissions=[
                "view_executive", 
                "view_cross_department",
                "view_dashboard", "edit_dashboard", 
                "view_report", "edit_report",
                "view_analytics"
            ] + [f"view_{dept}_data" for dept in departments],
            tier=3
        )
        
        roles["admin"] = Role(
            id="admin",
            name="Administrator",
            description="Full system access",
            permissions=list(self.permissions.keys()),  # All permissions
            tier=3
        )
        
        return roles
    
    def _init_default_department_modules(self) -> Dict[str, DepartmentModule]:
        """Initialize default department modules"""
        departments = {}
        
        # Sales department
        departments["sales"] = DepartmentModule(
            department_id="sales",
            department_name="Sales",
            modules=["crm", "sales_forecasting", "pipeline_management"]
        )
        
        # Marketing department
        departments["marketing"] = DepartmentModule(
            department_id="marketing",
            department_name="Marketing",
            modules=["campaign_analytics", "lead_generation", "social_media"]
        )
        
        # Product department
        departments["product"] = DepartmentModule(
            department_id="product",
            department_name="Product",
            modules=["product_development", "feature_usage", "user_feedback"]
        )
        
        # Finance department
        departments["finance"] = DepartmentModule(
            department_id="finance",
            department_name="Finance",
            modules=["accounting", "budget_management", "financial_reporting"]
        )
        
        # HR department
        departments["hr"] = DepartmentModule(
            department_id="hr",
            department_name="Human Resources",
            modules=["employee_management", "recruitment", "performance"]
        )
        
        # Operations department
        departments["operations"] = DepartmentModule(
            department_id="operations",
            department_name="Operations",
            modules=["logistics", "supply_chain", "inventory"]
        )
        
        return departments
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load permissions
            if 'permissions' in config:
                for perm_data in config['permissions']:
                    perm = Permission(**perm_data)
                    self.permissions[perm.id] = perm
            
            # Load roles
            if 'roles' in config:
                for role_data in config['roles']:
                    role = Role(**role_data)
                    self.roles[role.id] = role
            
            # Load departments
            if 'departments' in config:
                for dept_data in config['departments']:
                    dept = DepartmentModule(**dept_data)
                    self.department_modules[dept.department_id] = dept
            
            # Load users
            if 'users' in config:
                for user_data in config['users']:
                    user = User(**user_data)
                    self.users[user.id] = user
            
            logger.info(f"Loaded access control configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading access control config: {str(e)}")
            raise AccessControlError("config_load", f"Failed to load config: {str(e)}")
    
    def save_config(self, config_path: str) -> None:
        """Save configuration to file"""
        try:
            config = {
                'permissions': [perm.dict() for perm in self.permissions.values()],
                'roles': [role.dict() for role in self.roles.values()],
                'departments': [dept.dict() for dept in self.department_modules.values()],
                'users': [user.dict() for user in self.users.values()]
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Saved access control configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving access control config: {str(e)}")
            raise AccessControlError("config_save", f"Failed to save config: {str(e)}")
    
    @lru_cache(maxsize=100)
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user based on their roles"""
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return set()
        
        user = self.users[user_id]
        permissions = set()
        
        for role_id in user.roles:
            if role_id in self.roles:
                permissions.update(self.roles[role_id].permissions)
        
        return permissions
    
    @lru_cache(maxsize=100)
    def get_user_access_tier(self, user_id: str) -> int:
        """Get the highest access tier level for a user"""
        if user_id not in self.users:
            logger.warning(f"User {user_id} not found")
            return 0
        
        user = self.users[user_id]
        max_tier = 0
        
        for role_id in user.roles:
            if role_id in self.roles:
                max_tier = max(max_tier, self.roles[role_id].tier)
        
        return max_tier
    
    def can_access_department(self, user_id: str, department_id: str) -> bool:
        """Check if user can access a specific department"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Direct department access
        if department_id in user.departments:
            return True
        
        # Check tier level
        tier = self.get_user_access_tier(user_id)
        
        # Tier 2+ can access any department
        if tier >= 2:
            return True
        
        return False
    
    def can_access_module(self, user_id: str, module_id: str) -> bool:
        """Check if user can access a specific module"""
        if user_id not in self.users:
            return False
        
        # Find which department the module belongs to
        for dept_id, dept in self.department_modules.items():
            if module_id in dept.modules:
                # Check if user can access this department
                return self.can_access_department(user_id, dept_id)
        
        return False
    
    def can_access_cross_department(self, user_id: str) -> bool:
        """Check if user can access cross-department analytics"""
        tier = self.get_user_access_tier(user_id)
        permissions = self.get_user_permissions(user_id)
        
        # Need tier 2+ or specific cross-department permission
        return tier >= 2 or "view_cross_department" in permissions
    
    def can_access_executive_insights(self, user_id: str) -> bool:
        """Check if user can access executive-level insights"""
        tier = self.get_user_access_tier(user_id)
        permissions = self.get_user_permissions(user_id)
        
        # Need tier 3 or specific executive permission
        return tier >= 3 or "view_executive" in permissions
    
    def add_user(self, user: User) -> None:
        """Add a new user"""
        self.users[user.id] = user
        logger.info(f"Added user {user.id}")
        
        # Invalidate cache for this user
        if hasattr(self.get_user_permissions, 'cache_clear'):
            self.get_user_permissions.cache_clear()
        if hasattr(self.get_user_access_tier, 'cache_clear'):
            self.get_user_access_tier.cache_clear()
    
    def update_user(self, user_id: str, **updates) -> None:
        """Update an existing user"""
        if user_id not in self.users:
            raise AccessControlError("user_not_found", f"User {user_id} not found")
        
        user = self.users[user_id]
        
        # Update user fields
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        user.updated_at = datetime.now()
        logger.info(f"Updated user {user_id}")
        
        # Invalidate cache for this user
        if hasattr(self.get_user_permissions, 'cache_clear'):
            self.get_user_permissions.cache_clear()
        if hasattr(self.get_user_access_tier, 'cache_clear'):
            self.get_user_access_tier.cache_clear()
    
    def get_accessible_departments(self, user_id: str) -> List[str]:
        """Get list of departments accessible to a user"""
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        tier = self.get_user_access_tier(user_id)
        
        # Tier 2+ can access all departments
        if tier >= 2:
            return list(self.department_modules.keys())
        
        # Tier 1 can only access their assigned departments
        return user.departments
    
    def get_accessible_modules(self, user_id: str) -> List[str]:
        """Get list of modules accessible to a user"""
        if user_id not in self.users:
            return []
        
        accessible_modules = []
        accessible_departments = self.get_accessible_departments(user_id)
        
        for dept_id in accessible_departments:
            if dept_id in self.department_modules:
                accessible_modules.extend(self.department_modules[dept_id].modules)
        
        return accessible_modules
    
    def get_accessible_analytics(self, user_id: str) -> Dict[str, bool]:
        """Get dictionary of accessible analytics features for a user"""
        if user_id not in self.users:
            return {}
        
        tier = self.get_user_access_tier(user_id)
        
        return {
            "department_level": True,  # All users can access department level
            "cross_department": tier >= 2 or self.can_access_cross_department(user_id),
            "executive_insights": tier >= 3 or self.can_access_executive_insights(user_id)
        }
