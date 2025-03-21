import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, BinaryIO
import os
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from datetime import datetime
import io
import hashlib
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, DateTime, MetaData, JSON, Binary, Text
from sqlalchemy.orm import sessionmaker
import uuid

class WarehouseManager:
    def __init__(self, connection_string=None, encryption_key=None):
        """
        Initialize the Warehouse Manager
        
        Args:
            connection_string: Database connection string
            encryption_key: Encryption key for data (if None, will generate one)
        """
        # Set up database connection
        self.connection_string = connection_string or os.getenv("WAREHOUSE_CONNECTION_STRING", "sqlite:///./warehouse.db")
        self.engine = create_engine(self.connection_string)
        self.metadata = MetaData()
        
        # Set up encryption
        if encryption_key:
            self.encryption_key = encryption_key
        else:
            # Generate a key or load from environment
            stored_key = os.getenv("WAREHOUSE_ENCRYPTION_KEY")
            if stored_key:
                self.encryption_key = stored_key
            else:
                self.encryption_key = self._generate_encryption_key()
        
        self.cipher_suite = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
        
        # Initialize warehouse tables
        self._initialize_warehouse()
        
    def _generate_encryption_key(self) -> str:
        """Generate a new encryption key"""
        key = Fernet.generate_key()
        return key.decode()
    
    def _initialize_warehouse(self):
        """Set up the warehouse tables if they don't exist"""
        # Data catalog table to track all datasets
        self.data_catalog = Table(
            'data_catalog', 
            self.metadata,
            Column('id', String(50), primary_key=True),
            Column('name', String(255)),
            Column('department', String(100)),
            Column('module_type', String(100)),
            Column('source_type', String(50)),
            Column('schema', JSON),
            Column('row_count', Integer),
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
            Column('metadata', JSON),
            Column('access_control', JSON)
        )
        
        # Data storage table with encrypted content
        self.data_storage = Table(
            'data_storage',
            self.metadata,
            Column('id', String(50), primary_key=True),
            Column('catalog_id', String(50)),
            Column('chunk_index', Integer),
            Column('encrypted_data', Binary),
            Column('checksum', String(64)),
            Column('created_at', DateTime)
        )
        
        # Create tables if they don't exist
        self.metadata.create_all(self.engine)
    
    # Updated load method in WarehouseManager class
def load(self, file_content: Union[bytes, BinaryIO], 
         name: str, 
         department: str = None, 
         module_type: str = None,
         metadata: Dict = None,
         access_control: Dict = None) -> str:
    """
    Load data into warehouse with enhanced classification
    """
    try:
        # Generate a unique ID for this dataset
        catalog_id = str(uuid.uuid4())
        
        # Determine file type and load into DataFrame
        if isinstance(file_content, bytes):
            content = file_content
        else:
            content = file_content.read()
            
        # Try to detect and load the data format
        df = self._detect_and_load_data(content)
        
        # Use the classification service to classify the data
        from classification_service import DataClassificationService
        classifier = DataClassificationService()
        classification = classifier.classify_dataset(df, metadata)
        
        # Use provided department/module or use classified ones if not provided
        department = department or classification["primary_classification"]["department"]
        module_type = module_type or classification["primary_classification"]["module_type"]
        
        # Store classification information in metadata
        if not metadata:
            metadata = {}
        metadata["classification"] = classification
        
        # Store catalog entry
        schema = self._extract_schema(df)
        current_time = datetime.now()
        
        # Create Session
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            # Insert catalog entry with classification info
            catalog_entry = {
                'id': catalog_id,
                'name': name,
                'department': department,
                'module_type': module_type,
                'source_type': self._detect_source_type(content),
                'schema': schema,
                'row_count': len(df),
                'created_at': current_time,
                'updated_at': current_time,
                'metadata': metadata,
                'access_control': access_control or {'public': True},
                'classification_confidence': classification["primary_classification"]["confidence"]
            }
            
            session.execute(self.data_catalog.insert().values(**catalog_entry))
            
            # Continue with the rest of the loading process as before...
            # [Rest of the loading code remains the same]
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    except Exception as e:
        raise Exception(f"Error loading data into warehouse: {str(e)}")
    
    def retrieve(self, catalog_id: str, 
                 decrypt: bool = True, 
                 user_context: Dict = None) -> pd.DataFrame:
        """
        Retrieve dataset from warehouse
        
        Args:
            catalog_id: ID of the dataset to retrieve
            decrypt: Whether to decrypt the data
            user_context: User context for access control
            
        Returns:
            DataFrame with the requested data
        """
        try:
            # Create Session
            Session = sessionmaker(bind=self.engine)
            session = Session()
            
            try:
                # Get catalog entry
                catalog_result = session.execute(
                    self.data_catalog.select().where(self.data_catalog.c.id == catalog_id)
                ).fetchone()
                
                if not catalog_result:
                    raise Exception(f"Dataset with ID {catalog_id} not found")
                
                # Check access control
                if user_context and not self._check_access(dict(catalog_result), user_context):
                    raise Exception("Access denied to this dataset")
                
                # Get data chunks
                chunks_result = session.execute(
                    self.data_storage.select().where(
                        self.data_storage.c.catalog_id == catalog_id
                    ).order_by(self.data_storage.c.chunk_index)
                ).fetchall()
                
                # Combine and decrypt chunks
                all_data = []
                for chunk in chunks_result:
                    if decrypt:
                        # Decrypt the data
                        decrypted_data = self.cipher_suite.decrypt(chunk.encrypted_data)
                        # Verify checksum
                        if hashlib.sha256(decrypted_data).hexdigest() != chunk.checksum:
                            raise Exception(f"Data integrity check failed for chunk {chunk.chunk_index}")
                        # Load as DataFrame
                        chunk_df = pd.read_json(io.StringIO(decrypted_data.decode()))
                        all_data.append(chunk_df)
                    else:
                        # Return encrypted data
                        all_data.append({'chunk_index': chunk.chunk_index, 'encrypted_data': chunk.encrypted_data})
                
                if decrypt:
                    # Combine all chunks
                    if all_data:
                        return pd.concat(all_data, ignore_index=True)
                    return pd.DataFrame()
                else:
                    return all_data
                    
            finally:
                session.close()
                
        except Exception as e:
            raise Exception(f"Error retrieving data from warehouse: {str(e)}")
    
    def _detect_and_load_data(self, content: bytes) -> pd.DataFrame:
        """Detect data format and load into DataFrame"""
        # Try different formats
        try:
            # Try CSV
            return pd.read_csv(io.BytesIO(content))
        except:
            try:
                # Try Excel
                return pd.read_excel(io.BytesIO(content))
            except:
                try:
                    # Try JSON
                    return pd.read_json(io.BytesIO(content))
                except:
                    raise Exception("Unsupported data format")
    
    def _detect_source_type(self, content: bytes) -> str:
        """Detect the source type of the data"""
        try:
            pd.read_csv(io.BytesIO(content))
            return "csv"
        except:
            try:
                pd.read_excel(io.BytesIO(content))
                return "excel"
            except:
                try:
                    pd.read_json(io.BytesIO(content))
                    return "json"
                except:
                    return "unknown"
    
    def _extract_schema(self, df: pd.DataFrame) -> Dict:
        """Extract schema information from DataFrame"""
        schema = {}
        for column in df.columns:
            dtype = str(df[column].dtype)
            # Get sample values
            sample = df[column].head(3).tolist()
            schema[column] = {
                'type': dtype,
                'sample': sample,
                'nullable': df[column].isnull().any()
            }
        return schema
    
    def _check_access(self, catalog_entry: Dict, user_context: Dict) -> bool:
        """Check if the user has access to this dataset"""
        access_control = catalog_entry.get('access_control', {})
        
        # Public datasets are accessible to everyone
        if access_control.get('public', False):
            return True
        
        # Check department access
        user_dept = user_context.get('department')
        if user_dept and user_dept == catalog_entry.get('department'):
            return True
        
        # Check specific user access
        allowed_users = access_control.get('users', [])
        user_id = user_context.get('user_id')
        if user_id and user_id in allowed_users:
            return True
        
        # Check role access
        allowed_roles = access_control.get('roles', [])
        user_roles = user_context.get('roles', [])
        if any(role in allowed_roles for role in user_roles):
            return True
        
        return False
    
    def _classify_data(self, df: pd.DataFrame) -> tuple:
    """
    Advanced classification of data into department and module type
    using column names, data patterns, and content analysis
    """
    columns = set(col.lower() for col in df.columns)
    column_list = [col.lower() for col in df.columns]
    sample_data = df.head(50)  # Analyze a sample for patterns
    
    # Initialize with default values
    department = "Unknown"
    module_type = "Unknown"
    confidence_score = 0
    
    # Dictionary mapping patterns to department and module
    classification_patterns = {
        # CRM data patterns
        "CRM": {
            "department": "Sales",
            "module_type": "Customer Relationship Management",
            "column_patterns": [
                'customer_id', 'customer_name', 'lead', 'opportunity', 'contact', 
                'account', 'deal', 'pipeline', 'sales_stage', 'customer_lifetime_value'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_monetary_values": True,
                "has_status_fields": True
            }
        },
        # Marketing data patterns
        "Marketing": {
            "department": "Marketing",
            "module_type": "Marketing & Social Media",
            "column_patterns": [
                'campaign', 'channel', 'impressions', 'clicks', 'conversion', 'ctr', 'cpc', 
                'ad_spend', 'roi', 'engagement', 'reach', 'followers', 'likes', 'shares'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_monetary_values": True,
                "has_percentage_metrics": True
            }
        },
        # Website data patterns
        "Website": {
            "department": "Marketing", 
            "module_type": "Website Traffic",
            "column_patterns": [
                'page', 'url', 'session', 'visit', 'visitor', 'pageview', 'bounce_rate',
                'time_on_page', 'referrer', 'source', 'medium', 'browser', 'device'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_urls": True,
                "has_time_metrics": True
            }
        },
        # HR data patterns
        "HR": {
            "department": "Human Resources",
            "module_type": "Talent Management",
            "column_patterns": [
                'employee', 'salary', 'hire_date', 'position', 'title', 'compensation',
                'benefits', 'performance', 'review', 'attendance', 'training', 'onboarding'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_monetary_values": True,
                "has_person_names": True
            }
        },
        # Project data patterns
        "Project": {
            "department": "Operations",
            "module_type": "Project Management",
            "column_patterns": [
                'project', 'task', 'milestone', 'deadline', 'status', 'assignee', 
                'priority', 'completion', 'budget', 'actual_cost', 'planned_hours'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_status_fields": True,
                "has_percentage_metrics": True
            }
        },
        # Supply chain patterns
        "Supply": {
            "department": "Operations",
            "module_type": "Supply Chain Management",
            "column_patterns": [
                'supplier', 'vendor', 'purchase_order', 'shipment', 'delivery', 'lead_time',
                'order_quantity', 'logistics', 'receiving', 'procurement', 'sourcing'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_monetary_values": True,
                "has_quantity_fields": True
            }
        },
        # Inventory patterns
        "Inventory": {
            "department": "Operations",
            "module_type": "Inventory Management",
            "column_patterns": [
                'inventory', 'stock', 'sku', 'product_id', 'warehouse', 'quantity', 
                'reorder_point', 'stockout', 'backorder', 'bin_location', 'lot_number'
            ],
            "data_patterns": {
                "has_quantity_fields": True,
                "has_codes": True,
                "has_monetary_values": True
            }
        },
        # Finance patterns
        "Finance": {
            "department": "Finance",
            "module_type": "Finance & Accounting",
            "column_patterns": [
                'revenue', 'expense', 'profit', 'cost', 'budget', 'invoice', 'payment',
                'transaction', 'cash_flow', 'asset', 'liability', 'equity', 'tax'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_monetary_values": True,
                "has_accounting_structure": True
            }
        },
        # Customer feedback patterns
        "Feedback": {
            "department": "Customer Service",
            "module_type": "Complaints and Feedback",
            "column_patterns": [
                'feedback', 'rating', 'review', 'complaint', 'survey', 'satisfaction',
                'nps', 'csat', 'sentiment', 'comment', 'issue', 'resolution'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_text_fields": True,
                "has_rating_fields": True
            }
        },
        # IT/Security patterns
        "IT": {
            "department": "IT",
            "module_type": "IT & Security Management",
            "column_patterns": [
                'log', 'incident', 'security', 'access', 'ip_address', 'hostname', 
                'server', 'application', 'error', 'vulnerability', 'response_time'
            ],
            "data_patterns": {
                "has_date_sequences": True,
                "has_ip_addresses": True,
                "has_technical_ids": True
            }
        },
        # Customer Demographics
        "Demographics": {
            "department": "Marketing",
            "module_type": "Customer Demographics",
            "column_patterns": [
                'age', 'gender', 'income', 'education', 'occupation', 'location',
                'zip_code', 'country', 'state', 'city', 'language', 'household'
            ],
            "data_patterns": {
                "has_person_attributes": True,
                "has_location_fields": True,
                "has_demographic_fields": True
            }
        }
    }
    
    # Detect data patterns in the sample data
    data_patterns = {
        "has_date_sequences": any(pd.api.types.is_datetime64_any_dtype(sample_data[col]) 
                                 for col in sample_data.columns if col in df),
        "has_monetary_values": any(col.endswith(('price', 'cost', 'revenue', 'income', 'expense', 'budget')) 
                                  for col in column_list),
        "has_status_fields": any(col.endswith(('status', 'stage', 'phase', 'state')) 
                               for col in column_list),
        "has_percentage_metrics": any(col.endswith(('rate', 'ratio', 'percentage', 'pct', '%')) 
                                    for col in column_list),
        "has_urls": any('url' in col or 'link' in col for col in column_list),
        "has_time_metrics": any('time' in col or 'duration' in col for col in column_list),
        "has_person_names": any(col.endswith(('name', 'first_name', 'last_name')) 
                              for col in column_list),
        "has_quantity_fields": any(col.endswith(('quantity', 'count', 'amount', 'number', 'qty')) 
                                 for col in column_list),
        "has_codes": any(col.endswith(('code', 'id', 'sku', 'reference')) 
                       for col in column_list),
        "has_accounting_structure": any(col in ('debit', 'credit', 'account_code', 'gl_code') 
                                      for col in column_list),
        "has_text_fields": any(df[col].dtype == 'object' and 
                              df[col].str.len().mean() > 20 if hasattr(df[col], 'str') else False 
                              for col in sample_data.columns if col in df),
        "has_rating_fields": any(col.endswith(('rating', 'score', 'stars', 'grade')) 
                              for col in column_list),
        "has_ip_addresses": any('ip' in col for col in column_list),
        "has_technical_ids": any(col.endswith(('_id', '_key', '_code')) 
                               for col in column_list),
        "has_person_attributes": any(col in ('age', 'gender', 'occupation', 'education') 
                                   for col in column_list),
        "has_location_fields": any(col in ('country', 'state', 'city', 'region', 'postal_code', 'zip') 
                                 for col in column_list),
        "has_demographic_fields": any(col in ('income', 'household_size', 'marital_status') 
                                    for col in column_list)
    }
    
    # Calculate match scores for each category
    for category, patterns in classification_patterns.items():
        col_score = sum(1 for pattern in patterns["column_patterns"] 
                       if any(pattern in col for col in column_list))
        col_score_weighted = col_score / len(patterns["column_patterns"]) * 0.7
        
        data_score = sum(1 for pattern, exists in patterns["data_patterns"].items() 
                        if exists and data_patterns.get(pattern, False))
        data_score_weighted = data_score / len(patterns["data_patterns"]) * 0.3
        
        total_score = col_score_weighted + data_score_weighted
        
        if total_score > confidence_score:
            confidence_score = total_score
            department = patterns["department"]
            module_type = patterns["module_type"]
    
    # If confidence is too low, keep as "Unknown"
    if confidence_score < 0.3:
        return "Unknown", "Unknown"
    
    return department, module_type
        
    def list_datasets(self, 
                      department: str = None, 
                      module_type: str = None,
                      user_context: Dict = None) -> List[Dict]:
        """
        List available datasets, optionally filtered by department/module
        
        Args:
            department: Filter by department
            module_type: Filter by module type
            user_context: User context for access control
            
        Returns:
            List of dataset metadata
        """
        try:
            # Create Session
            Session = sessionmaker(bind=self.engine)
            session = Session()
            
            try:
                # Build query
                query = self.data_catalog.select()
                
                if department:
                    query = query.where(self.data_catalog.c.department == department)
                
                if module_type:
                    query = query.where(self.data_catalog.c.module_type == module_type)
                
                # Execute query
                results = session.execute(query).fetchall()
                
                # Filter by access control if user_context provided
                if user_context:
                    accessible_results = []
                    for row in results:
                        row_dict = dict(row)
                        if self._check_access(row_dict, user_context):
                            # Remove sensitive fields
                            if 'access_control' in row_dict:
                                del row_dict['access_control']
                            accessible_results.append(row_dict)
                    return accessible_results
                else:
                    # Convert to list of dicts
                    return [dict(row) for row in results]
                    
            finally:
                session.close()
                
        except Exception as e:
            raise Exception(f"Error listing datasets: {str(e)}")
    
    def get_departments(self) -> List[str]:
        """Get list of all departments with data"""
        try:
            # Create Session
            Session = sessionmaker(bind=self.engine)
            session = Session()
            
            try:
                query = self.data_catalog.select(self.data_catalog.c.department).distinct()
                results = session.execute(query).fetchall()
                return [row[0] for row in results if row[0]]
            finally:
                session.close()
        except Exception as e:
            raise Exception(f"Error getting departments: {str(e)}")
    
    def get_module_types(self, department: str = None) -> List[str]:
        """Get list of all module types, optionally filtered by department"""
        try:
            # Create Session
            Session = sessionmaker(bind=self.engine)
            session = Session()
            
            try:
                query = self.data_catalog.select(self.data_catalog.c.module_type).distinct()
                
                if department:
                    query = query.where(self.data_catalog.c.department == department)
                    
                results = session.execute(query).fetchall()
                return [row[0] for row in results if row[0]]
            finally:
                session.close()
        except Exception as e:
            raise Exception(f"Error getting module types: {str(e)}")
