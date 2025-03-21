from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, APIRouter, Form, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Optional, Union, Any, Generator
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from prophet import Prophet
import plotly.express as px
import json
import logging
from datetime import datetime, timedelta
import os
import jwt
from .models import DataSource, Dashboard, Report, User, Role, Permission, DepartmentModule
from .database import get_db
from .ml_engine import MLEngine
from .data_processor import DataProcessor
from .nlp_engine import NaturalLanguageProcessor
from .export_manager import ExportManager
from .scenario_simulator import ScenarioSimulator
from .anomaly_detector import AnomalyDetector
from .data_cleaner import DataCleaner
from .summarizer import Summarizer
from .warehouse_manager import WarehouseManager
from .cross_module_analysis import CrossModuleAnalysis
from .analytics_recommender import AnalyticsRecommender
from .module_registry import DataModuleRegistry
from .classification_service import DataClassificationService
from .config import settings
from .access_control import AccessControlService, AccessControlError
import random
import itertools

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG
)

# JWT Authentication settings
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# OAuth2 scheme for token authentication
auth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_user_context(authorization: Optional[str] = Header(None)):
    """Extract user context from authorization header with proper JWT validation"""
    if not authorization:
        return None
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        
        # Decode and validate token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        departments = payload.get("departments", [])
        roles = payload.get("roles", [])
        exp = payload.get("exp")
        
        # Check if token is expired
        if datetime.fromtimestamp(exp) < datetime.utcnow():
            return None
            
        # Get user's access tier from access control service
        access_tier = 1  # Default to tier 1
        if user_id:
            access_tier = access_control_service.get_user_access_tier(user_id)
        
        # Return user context with access tier
        return {
            "user_id": user_id,
            "username": username,
            "departments": departments,
            "roles": roles,
            "access_tier": access_tier
        }
    except jwt.PyJWTError:
        return None
    except Exception as e:
        logging.error(f"Error validating token: {str(e)}")
        return None

def get_current_user(token: str = Depends(auth2_scheme)):
    """Validate JWT token and return user information"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        username = payload.get("sub")
        roles = payload.get("roles", [])
        departments = payload.get("departments", [])
        
        if username is None or user_id is None:
            raise credentials_exception
        
        # Get user's access tier from access control service
        access_tier = access_control_service.get_user_access_tier(user_id)
        
        # Return comprehensive user info with access tier
        return {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "departments": departments,
            "access_tier": access_tier
        }
    except jwt.PyJWTError:
        raise credentials_exception

# Check if user has department-level access
def get_department_user(current_user: dict = Depends(get_current_user)):
    """Verify user has at least department-level access (Tier 1+)"""
    if current_user.get("access_tier", 0) < 1:
        raise HTTPException(status_code=403, detail="Insufficient permissions - requires department access")
    return current_user

# Check if user has cross-department access
def get_cross_department_user(current_user: dict = Depends(get_current_user)):
    """Verify user has cross-department access (Tier 2+)"""
    if current_user.get("access_tier", 0) < 2:
        raise HTTPException(status_code=403, detail="Insufficient permissions - requires cross-department access")
    return current_user

# Check if user has executive-level access
def get_executive_user(current_user: dict = Depends(get_current_user)):
    """Verify user has executive-level access (Tier 3+)"""
    if current_user.get("access_tier", 0) < 3:
        raise HTTPException(status_code=403, detail="Insufficient permissions - requires executive access")
    return current_user

# Admin-level access check
def get_admin_user(current_user: dict = Depends(get_current_user)):
    """Verify user has admin role"""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user

# Initialize warehouse manager
warehouse_manager = WarehouseManager(
    connection_string=settings.WAREHOUSE_CONNECTION_STRING,
    encryption_key=settings.WAREHOUSE_ENCRYPTION_KEY
)
# Initialize cross module analysis service
cross_module_analysis = CrossModuleAnalysis(warehouse_manager=warehouse_manager)

# Initialize access control service
access_control_service = AccessControlService()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Routers
router = APIRouter()

# API versioning
v1_router = APIRouter(prefix="/api/v1")

# Include router in the app
app.include_router(v1_router)
app.include_router(router)

# Initialize components
components = {
    "ml_engine": MLEngine(),
    "data_processor": DataProcessor(),
    "nlp_engine": NaturalLanguageProcessor(),
    "export_manager": ExportManager(),
    "scenario_simulator": ScenarioSimulator(),
    "anomaly_detector": AnomalyDetector(),
    "data_cleaner": DataCleaner(),
    "summarizer": Summarizer(),
    "warehouse_manager": warehouse_manager,
    "cross_module_analysis": cross_module_analysis
}



class UserRequest(BaseModel):
    input_data: str
    request_type: str

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """JWT token login endpoint"""
    # In a real application, validate against database
    # For demo purposes, we use hardcoded values
    if form_data.username in settings.DEMO_USERS:
        user_data = settings.DEMO_USERS[form_data.username]
        
        if form_data.password == user_data.get("password", ""):
            # User authenticated, create access token
            user_id = user_data.get("id", form_data.username)
            roles = user_data.get("roles", ["viewer"])
            departments = user_data.get("departments", [])
            
            # Create token data with role and department info
            token_data = {
                "sub": form_data.username,
                "user_id": user_id,
                "roles": roles,
                "departments": departments
            }
            
            # If this is a real user in the access control service, get their access tier
            if user_id in access_control_service.users:
                token_data["access_tier"] = access_control_service.get_user_access_tier(user_id)
            else:
                # Determine access tier based on roles
                if "admin" in roles:
                    token_data["access_tier"] = 3
                elif any(r in roles for r in ["regional_manager", "business_analyst"]):
                    token_data["access_tier"] = 2
                else:
                    token_data["access_tier"] = 1
            
            # Create and return the access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data=token_data,
                expires_delta=access_token_expires
            )
            return {
                "access_token": access_token, 
                "token_type": "bearer",
                "user_info": {
                    "username": form_data.username,
                    "roles": roles,
                    "departments": departments,
                    "access_tier": token_data.get("access_tier", 1)
                }
            }
    
    # Authentication failed
    raise HTTPException(
        status_code=401,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

# New endpoints for executive-level analytics

@app.get("/api/executive/overview", dependencies=[Depends(get_executive_user)])
async def executive_overview():
    """Get company-wide executive overview"""
    try:
        # This would fetch data from all departments and create an executive summary
        result = {
            "summary": {
                "total_departments": len(access_control_service.department_modules),
                "active_modules": sum(len(dept.modules) for dept in access_control_service.department_modules.values()),
                "performance_indicators": {
                    "revenue_growth": 0.08,  # Example data
                    "customer_satisfaction": 92,
                    "operational_efficiency": 0.75,
                    "employee_engagement": 87
                }
            },
            "department_metrics": {},
            "alerts": [
                {
                    "severity": "high",
                    "message": "Sales target at risk for Q2",
                    "impact": "Revenue projection may need to be adjusted by 5%",
                    "recommended_action": "Review pipeline and accelerate key accounts"
                },
                {
                    "severity": "medium",
                    "message": "Marketing campaign performance below expectation",
                    "impact": "Lead generation down 8% from forecast",
                    "recommended_action": "Adjust channel mix and review messaging"
                }
            ],
            "cross_department_insights": [
                {
                    "insight": "Product launch delay affecting sales pipeline",
                    "departments": ["product", "sales"],
                    "impact": "high",
                    "recommendation": "Accelerate release or adjust sales targets"
                },
                {
                    "insight": "New hiring plan aligns with projected growth",
                    "departments": ["hr", "finance"],
                    "impact": "positive",
                    "recommendation": "Continue as planned"
                }
            ]
        }
        
        # Add metrics for each department
        for dept_id, dept in access_control_service.department_modules.items():
            # This would be real data in production
            result["department_metrics"][dept_id] = {
                "performance_score": round(70 + 30 * random.random(), 1),  # Example score between 70-100
                "trend": "up" if random.random() > 0.3 else "down",
                "key_metrics": {
                    "budget_variance": round(-5 + 10 * random.random(), 1),  # -5% to +5%
                    "goal_completion": round(75 + 25 * random.random(), 1),  # 75-100%
                    "risk_score": round(1 + 4 * random.random(), 1),  # 1-5 scale
                }
            }
        
        return result
    except Exception as e:
        logger.error(f"Error in executive_overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dashboard/create", dependencies=[Depends(get_department_user)])
async def create_dashboard(dashboard: Dashboard):
    """Create new dashboard"""
    try:
        return {"message": "Dashboard created", "id": "dashboard_id"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/report/generate", dependencies=[Depends(get_department_user)])
async def generate_report(report: Report):
    """Generate automated report"""
    try:
        report_data = components["data_processor"].generate_report(report)
        return report_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
@app.post("/api/analysis/budget_forecast", dependencies=[Depends(get_department_user)])
async def create_budget_forecast(request: Dict):
    """Generate budget forecasts based on operational and financial data"""
    try:
        forecast = components["ml_engine"].create_budget_forecast(
            data=request.get('data', {}),
            config=request.get('config', {})
        )
        return forecast
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
@app.get("/api/metrics/summary", dependencies=[Depends(get_department_user)])
async def get_metrics_summary():
    """Get summary of key metrics"""
    try:
        return {"metrics": ["metric1", "metric2"], "values": [100, 200]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
@app.post("/api/data/clean", dependencies=[Depends(get_department_user)])
async def clean_data(data: Dict, config: Optional[Dict] = None):
    """Clean, normalize and handle missing data"""
    try:
        data_cleaner = DataCleaner()
        cleaned_data = data_cleaner.clean(data, config)
        return cleaned_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analysis/detect_anomalies", dependencies=[Depends(get_department_user)])
async def detect_data_anomalies(data: Dict):
    """Detect anomalies in the data"""
    try:
        anomalies = components["ml_engine"].detect_anomalies(data)
        # Count anomalies and calculate percentage
        anomaly_count = sum(1 for val in anomalies if val == -1)
        anomaly_percentage = (anomaly_count / len(anomalies)) * 100 if anomalies else 0
        
        return {
            "anomalies": anomalies,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "threshold_exceeded": anomaly_percentage > 5  # Alert if > 5% anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/warehouse/load", dependencies=[Depends(get_department_user)])
async def load_to_warehouse(
    file: UploadFile = File(...),
    name: str = Form(...),
    department: Optional[str] = Form(None),
    module_type: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Load data into warehouse with encryption"""
    try:
        contents = await file.read()
        
        # Parse metadata if provided
        metadata_dict = json.loads(metadata) if metadata else None
        
        # Get user context for access control
        user_context = get_user_context(authorization=request.headers.get("Authorization"))
        
        # Load data into warehouse
        catalog_id = warehouse_manager.load(
            contents,
            name=name,
            department=department,
            module_type=module_type,
            metadata=metadata_dict,
            access_control={"public": False, "users": [user_context["user_id"]]} if user_context else {"public": True}
        )
        
        return {
            "message": "Data loaded successfully into warehouse",
            "catalog_id": catalog_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/warehouse/datasets", dependencies=[Depends(get_department_user)])
async def list_warehouse_datasets(
    request: Request,
    department: Optional[str] = None,
    module_type: Optional[str] = None
):
    """List datasets in the warehouse"""
    try:
        # Get user context for access control
        user_context = get_user_context(authorization=request.headers.get("Authorization"))
        
        # Get datasets
        datasets = warehouse_manager.list_datasets(
            department=department,
            module_type=module_type,
            user_context=user_context
        )
        
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/warehouse/departments")
async def list_departments():
    """Get list of departments"""
    try:
        departments = warehouse_manager.get_departments()
        return {"departments": departments}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/warehouse/module-types")
async def list_module_types(department: Optional[str] = None):
    """Get list of module types"""
    try:
        module_types = warehouse_manager.get_module_types(department=department)
        return {"module_types": module_types}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/warehouse/data/{catalog_id}", dependencies=[Depends(get_department_user)])
async def get_warehouse_data(catalog_id: str, request: Request):
    """Retrieve data from warehouse"""
    try:
        # Get user context for access control
        user_context = get_user_context(authorization=request.headers.get("Authorization"))
        
        # Get data
        df = warehouse_manager.retrieve(
            catalog_id=catalog_id,
            decrypt=True,
            user_context=user_context
        )
        
        # Convert to dict for JSON response
        return {
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "row_count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
# Add these to app/main.py
@app.get("/api/modules/registry")
async def get_module_registry():
    """Get all registered data modules"""
    try:
        registry = DataModuleRegistry()
        return {
            "modules": [m.dict() for m in registry.get_all_modules()],
            "departments": registry.get_all_departments()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/modules/by-department/{department}")
async def get_modules_by_department(department: str):
    """Get modules for a specific department"""
    try:
        registry = DataModuleRegistry()
        modules = registry.get_department_modules(department)
        return {"modules": [m.dict() for m in modules]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/warehouse/classification-info/{catalog_id}", dependencies=[Depends(get_department_user)])
async def get_classification_info(catalog_id: str):
    """Get classification details for a dataset"""
    try:
        # Create Session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Get catalog entry
            result = session.execute(
                data_catalog.select().where(data_catalog.c.id == catalog_id)
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Dataset not found")
                
            # Extract classification info from metadata
            metadata = dict(result).get("metadata", {})
            classification = metadata.get("classification", {})
            
            # Get module definition from registry
            registry = DataModuleRegistry()
            module_def = registry.get_module(dict(result).get("module_type"))
            
            return {
                "catalog_id": catalog_id,
                "name": dict(result).get("name"),
                "department": dict(result).get("department"),
                "module_type": dict(result).get("module_type"),
                "classification": classification,
                "module_definition": module_def.dict() if module_def else None
            }
        finally:
            session.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/warehouse/reclassify/{catalog_id}", dependencies=[Depends(get_department_user)])
async def reclassify_dataset(catalog_id: str):
    """Reclassify an existing dataset"""
    try:
        # Create warehouse manager
        warehouse_manager = WarehouseManager()
        
        # Create Session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Get catalog entry
            result = session.execute(
                data_catalog.select().where(data_catalog.c.id == catalog_id)
            ).fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            # Retrieve the dataset
            df = warehouse_manager.retrieve(catalog_id)
            
            # Reclassify with the classification service
            classifier = DataClassificationService(
                config_path=settings.DATA_MAPPER_CONFIG_PATH,
                threshold=settings.CLASSIFICATION_CONFIDENCE_THRESHOLD
            )
            classification = classifier.classify_dataset(df)
            
            # Update metadata with new classification
            metadata = dict(result).get("metadata", {})
            metadata["classification"] = classification
            
            # Update catalog entry
            update_values = {
                "department": classification["primary_classification"]["department"],
                "module_type": classification["primary_classification"]["module_type"],
                "metadata": metadata,
                "classification_confidence": classification["primary_classification"]["confidence"],
                "updated_at": datetime.now()
            }
            
            session.execute(
                data_catalog.update()
                .where(data_catalog.c.id == catalog_id)
                .values(**update_values)
            )
            
            session.commit()
            
            return {
                "catalog_id": catalog_id,
                "name": dict(result).get("name"),
                "new_classification": classification["primary_classification"],
                "previous_department": dict(result).get("department"),
                "previous_module_type": dict(result).get("module_type"),
                "new_department": classification["primary_classification"]["department"],
                "new_module_type": classification["primary_classification"]["module_type"]
            }
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/api/analysis/cross-module/join", dependencies=[Depends(get_cross_department_user)])
async def join_datasets(
    primary_dataset_id: str,
    secondary_dataset_id: str,
    join_columns: Dict[str, str],
    request: Request
):
    """Join two datasets from different modules"""
    try:
        user_context = get_user_context(authorization=request.headers.get("Authorization"))
        
        joined_data = components["cross_module_analysis"].join_datasets(
            primary_dataset_id,
            secondary_dataset_id,
            join_columns,
            user_context
        )
        
        return {
            "success": True,
            "rows": len(joined_data),
            "columns": joined_data.columns.tolist(),
            "data": joined_data.head(100).to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error in executive_risk_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add department-specific endpoint
@app.get("/api/department/{department_id}/overview", dependencies=[Depends(get_department_user)])
async def department_overview(department_id: str, current_user: dict = Depends(get_department_user)):
    """Get overview for a specific department"""
    try:
        # Check if user can access this department
        if not access_control_service.can_access_department(current_user.get("user_id"), department_id):
            raise HTTPException(status_code=403, detail=f"Access denied to department {department_id}")
        
        # Check if department exists
        if department_id not in access_control_service.department_modules:
            raise HTTPException(status_code=404, detail=f"Department {department_id} not found")
        
        department = access_control_service.department_modules[department_id]
        
        # This would fetch real data in production
        overview = {
            "department_name": department.department_name,
            "modules": department.modules,
            "metrics": {
                "performance_score": round(70 + 30 * random.random(), 1),
                "trend": "up" if random.random() > 0.3 else "down",
                "key_metrics": {
                    "budget_variance": round(-5 + 10 * random.random(), 1),
                    "goal_completion": round(75 + 25 * random.random(), 1),
                    "risk_score": round(1 + 4 * random.random(), 1),
                }
            },
            "alerts": [
                {
                    "severity": random.choice(["low", "medium", "high"]),
                    "message": f"{department.department_name} alert example",
                    "impact": random.choice(["minimal", "moderate", "significant"]),
                    "recommended_action": "Review and address as needed"
                }
            ],
            "module_metrics": {}
        }
        
        # Add metrics for each module
        for module in department.modules:
            overview["module_metrics"][module] = {
                "performance_score": round(70 + 30 * random.random(), 1),
                "utilization": round(60 + 40 * random.random(), 1),
                "status": random.choice(["green", "yellow", "red"])
            }
        
        return overview
    except AccessControlError as e:
        logger.error(f"Access control error: {e.message}")
        raise HTTPException(status_code=403, detail=e.message)
    except Exception as e:
        logger.error(f"Error in department_overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add cross-department analytics endpoint
@app.get("/api/cross-department/overview", dependencies=[Depends(get_cross_department_user)])
async def cross_department_overview(current_user: dict = Depends(get_cross_department_user)):
    """Get cross-department analytics overview"""
    try:
        # Get list of departments the user can access
        user_id = current_user.get("user_id")
        accessible_departments = access_control_service.get_accessible_departments(user_id)
        
        if not accessible_departments:
            raise HTTPException(status_code=403, detail="No departments accessible")
        
        overview = {
            "accessible_departments": accessible_departments,
            "cross_department_metrics": {
                "coordination_score": round(60 + 40 * random.random(), 1),
                "process_alignment": round(70 + 30 * random.random(), 1),
                "communication_effectiveness": round(65 + 35 * random.random(), 1)
            },
            "department_comparisons": {},
            "process_bottlenecks": [
                {
                    "process": "Customer onboarding",
                    "involved_departments": ["sales", "operations", "product"],
                    "current_duration": "14 days",
                    "target_duration": "7 days",
                    "bottleneck": "Documentation approval",
                    "recommendation": "Implement digital signature system"
                },
                {
                    "process": "Product feedback implementation",
                    "involved_departments": ["sales", "product", "marketing"],
                    "current_duration": "45 days",
                    "target_duration": "30 days",
                    "bottleneck": "Prioritization and scheduling",
                    "recommendation": "Weekly cross-department prioritization meeting"
                }
            ],
            "collaboration_opportunities": []
        }
        
        # Add department comparisons
        for dept_id in accessible_departments:
            if dept_id in access_control_service.department_modules:
                overview["department_comparisons"][dept_id] = {
                    "performance_score": round(70 + 30 * random.random(), 1),
                    "resource_utilization": round(60 + 40 * random.random(), 1),
                    "process_efficiency": round(50 + 50 * random.random(), 1)
                }
        
        # Add collaboration opportunities
        department_pairs = list(itertools.combinations(accessible_departments, 2))
        for dept1, dept2 in random.sample(department_pairs, min(3, len(department_pairs))):
            overview["collaboration_opportunities"].append({
                "departments": [dept1, dept2],
                "opportunity": f"Improve {dept1}-{dept2} handoff process",
                "potential_impact": random.choice(["medium", "high"]),
                "effort_required": random.choice(["low", "medium", "high"]),
                "recommendation": f"Create shared metrics for {dept1}-{dept2} collaboration"
            })
        
        return overview
    except AccessControlError as e:
        logger.error(f"Access control error: {e.message}")
        raise HTTPException(status_code=403, detail=e.message)
    except Exception as e:
        logger.error(f"Error in cross_department_overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Data processing pipeline functions
async def process_dataset(dataset, user_context=None):
    """Process data through the cleaning pipeline with access control"""
    cleaner = DataCleaner()
    try:
        async for chunk in cleaner.clean(dataset, config=settings.CLEANER_CONFIG):
            await validate_data(chunk)
            
            # Add user context to chunk metadata
            if user_context and 'metadata' in chunk:
                chunk['metadata']['processed_by'] = user_context.get('user_id')
                chunk['metadata']['access_tier'] = user_context.get('access_tier', 1)
                chunk['metadata']['departments'] = user_context.get('departments', [])
            
            yield chunk
    except DataCleaningError as e:
        logger.error(f"Cleaning failed: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {e.message}")

async def classify_dataset(dataset, user_context=None):
    """Classify the dataset and assign to appropriate department/module"""
    try:
        # Create classifier service
        classifier = DataClassificationService(
            config_path=settings.DATA_MAPPER_CONFIG_PATH,
            threshold=settings.CLASSIFICATION_CONFIDENCE_THRESHOLD
        )
        
        # Convert dataset to DataFrame if it's not already
        if isinstance(dataset, dict) and 'data' in dataset:
            df = pd.DataFrame(dataset['data'])
        elif isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise ValueError("Invalid dataset format")
        
        # Classify the dataset
        classification_result = classifier.classify_dataset(df)
        
        # Check if user has access to the classified department
        if user_context and classification_result.get('primary_classification'):
            dept = classification_result['primary_classification'].get('department')
            if dept and not access_control_service.can_access_department(
                user_context.get('user_id'), dept
            ):
                logger.warning(f"User {user_context.get('user_id')} attempted to access data for department {dept}")
                # Don't block the classification, but add access warning to metadata
                classification_result['access_warning'] = f"User has limited access to department {dept}"
        
        # Add classification to dataset metadata
        if isinstance(dataset, dict) and 'metadata' in dataset:
            dataset['metadata']['classification'] = classification_result
        else:
            # Create metadata if it doesn't exist
            metadata = {'classification': classification_result}
            if isinstance(dataset, dict):
                dataset['metadata'] = metadata
        
        return dataset
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data classification failed: {str(e)}")

async def consolidate_datasets(datasets, user_context=None):
    """Consolidate multiple datasets with access control"""
    consolidator = DataConsolidator()
    try:
        # Filter datasets based on user's department access
        if user_context:
            filtered_datasets = []
            for ds in datasets:
                # If dataset has department metadata, check access
                if isinstance(ds, dict) and 'metadata' in ds:
                    dept = ds['metadata'].get('department')
                    if dept and not access_control_service.can_access_department(
                        user_context.get('user_id'), dept
                    ):
                        logger.warning(f"User {user_context.get('user_id')} skipped dataset for department {dept}")
                        continue
                filtered_datasets.append(ds)
            datasets = filtered_datasets
        
        # Process datasets through consolidator
        async for chunk in consolidator.consolidate(datasets, config=settings.CONSOLIDATION_CONFIG):
            await validate_consolidated_data(chunk)
            
            # Add user context to chunk metadata
            if user_context and 'metadata' in chunk:
                chunk['metadata']['consolidated_by'] = user_context.get('user_id')
                chunk['metadata']['access_tier'] = user_context.get('access_tier', 1)
            
            yield chunk
    except DataConsolidationError as e:
        logger.error(f"Consolidation failed: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data consolidation failed: {e.message}")

async def map_to_module(dataset, module_type=None, user_context=None):
    """Map dataset to module schema with access control"""
    try:
        # If module_type not provided, get it from classification
        if not module_type and isinstance(dataset, dict) and 'metadata' in dataset:
            classification = dataset['metadata'].get('classification', {})
            primary = classification.get('primary_classification', {})
            module_type = primary.get('module_type')
        
        if not module_type:
            raise ValueError("Module type not provided and could not be determined from classification")
        
        # Check if user has access to this module
        if user_context and not access_control_service.can_access_module(
            user_context.get('user_id'), module_type
        ):
            logger.warning(f"User {user_context.get('user_id')} attempted to map data to restricted module {module_type}")
            raise HTTPException(status_code=403, detail=f"Access denied to module {module_type}")
        
        # Use data mapper to map dataset to module schema
        mapper = DataMapper(config_path=settings.DATA_MAPPER_CONFIG_PATH)
        mapped_data = mapper.map_to_module(dataset, module_type)
        
        # Add mapping metadata
        if isinstance(mapped_data, dict) and 'metadata' in mapped_data:
            mapped_data['metadata']['mapped_at'] = datetime.now().isoformat()
            mapped_data['metadata']['mapped_by'] = user_context.get('user_id') if user_context else 'system'
            mapped_data['metadata']['module_type'] = module_type
        
        return mapped_data
    except Exception as e:
        logger.error(f"Module mapping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data mapping failed: {str(e)}")

async def validate_data(data):
    """Validate data integrity after processing"""
    # Implement validation logic
    return True

async def validate_consolidated_data(data):
    """Validate consolidated data integrity"""
    # Implement validation logic
    return True

# Add data processing endpoints with appropriate access control
@app.post("/api/data/process", dependencies=[Depends(get_department_user)])
async def process_data(
    request: Dict,
    current_user: dict = Depends(get_department_user)
):
    """Process data through the pipeline: clean, classify, map to module"""
    try:
        data = request.get("data", {})
        
        # Process through pipeline
        cleaned_data = await anext(process_dataset(data, current_user))
        classified_data = await classify_dataset(cleaned_data, current_user)
        
        # Get module type from classification or request
        module_type = request.get("module_type")
        if not module_type and 'metadata' in classified_data:
            classification = classified_data['metadata'].get('classification', {})
            primary = classification.get('primary_classification', {})
            module_type = primary.get('module_type')
        
        # Map to module if module type is available
        result = classified_data
        if module_type:
            result = await map_to_module(classified_data, module_type, current_user)
        
        return {
            "success": True,
            "processed_data": result,
            "classification": classified_data.get('metadata', {}).get('classification', {})
        }
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/consolidate", dependencies=[Depends(get_cross_department_user)])
async def consolidate_multiple_datasets(
    request: Dict,
    current_user: dict = Depends(get_cross_department_user)
):
    """Consolidate multiple datasets - requires cross-department access"""
    try:
        datasets = request.get("datasets", [])
        if not datasets:
            raise HTTPException(status_code=400, detail="No datasets provided")
        
        # Check access to all datasets
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict) and 'metadata' in dataset:
                dept = dataset['metadata'].get('department')
                if dept and not access_control_service.can_access_department(
                    current_user.get('user_id'), dept
                ):
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Access denied to dataset {i} for department {dept}"
                    )
        
        # Process datasets through consolidation pipeline
        consolidated_chunks = []
        async for chunk in consolidate_datasets(datasets, current_user):
            consolidated_chunks.append(chunk)
        
        # If we got multiple chunks, combine them
        if len(consolidated_chunks) > 1:
            # Simple combination - in a real app you'd handle this more carefully
            combined = consolidated_chunks[0]
            if 'data' in combined and isinstance(combined['data'], list):
                for chunk in consolidated_chunks[1:]:
                    combined['data'].extend(chunk.get('data', []))
            result = combined
        elif consolidated_chunks:
            result = consolidated_chunks[0]
        else:
            result = {"data": [], "metadata": {"warning": "No data after consolidation"}}
        
        return {
            "success": True,
            "consolidated_data": result
        }
    except Exception as e:
        logger.error(f"Data consolidation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/store", dependencies=[Depends(get_department_user)])
async def store_processed_data(
    request: Dict,
    current_user: dict = Depends(get_department_user)
):
    """Store processed data in the data warehouse with access control"""
    try:
        data = request.get("data", {})
        
        # Extract metadata
        name = request.get("name", f"Dataset-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        department = request.get("department")
        module_type = request.get("module_type")
        
        # If not provided, try to get from data metadata
        if isinstance(data, dict) and 'metadata' in data:
            if not department:
                department = data['metadata'].get('department')
            if not module_type:
                module_type = data['metadata'].get('module_type')
            
            # Or from classification
            if not department or not module_type:
                classification = data['metadata'].get('classification', {})
                primary = classification.get('primary_classification', {})
                if not department:
                    department = primary.get('department')
                if not module_type:
                    module_type = primary.get('module_type')
        
        # Check department access
        if department and not access_control_service.can_access_department(
            current_user.get('user_id'), department
        ):
            raise HTTPException(status_code=403, detail=f"Access denied to department {department}")
        
        # Create access control metadata
        access_control = {
            "creator_id": current_user.get('user_id'),
            "department": department,
            "access_tier": current_user.get('access_tier', 1),
            "public": False,
            "creation_date": datetime.now().isoformat(),
            "allowed_departments": current_user.get('departments', []) if current_user.get('access_tier', 0) < 2 else None
        }
        
        # Store in warehouse
        catalog_id = warehouse_manager.store(
            data=data.get('data', []) if isinstance(data, dict) else data,
            name=name,
            department=department,
            module_type=module_type,
            metadata=data.get('metadata', {}),
            access_control=access_control
        )
        
        return {
            "success": True,
            "catalog_id": catalog_id,
            "name": name,
            "department": department,
            "module_type": module_type
        }
    except Exception as e:
        logger.error(f"Data storage failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
