import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import re
from collections import Counter
import logging
from .data_mapper import DataMapper
import json
from datetime import datetime

logger = logging.getLogger("databloom.classification_service")

# Define classification patterns
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
        "module_type": "Web Analytics",
        "column_patterns": [
            'page', 'url', 'session', 'visitor', 'browser', 'device', 'referrer', 
            'bounce_rate', 'time_on_page', 'exit_rate', 'page_views'
        ],
        "data_patterns": {
            "has_urls": True,
            "has_time_metrics": True,
            "has_percentage_metrics": True
        }
    },
    # Sales data patterns
    "Sales": {
        "department": "Sales",
        "module_type": "Sales Performance",
        "column_patterns": [
            'order_id', 'product_id', 'quantity', 'price', 'discount', 'revenue', 
            'sales_rep', 'region', 'customer_segment', 'product_category'
        ],
        "data_patterns": {
            "has_monetary_values": True,
            "has_quantity_fields": True,
            "has_codes": True
        }
    },
    # Finance data patterns
    "Finance": {
        "department": "Finance",
        "module_type": "Financial Analysis",
        "column_patterns": [
            'account', 'transaction', 'debit', 'credit', 'balance', 'gl_code', 
            'cost_center', 'fiscal_year', 'quarter', 'budget', 'actual'
        ],
        "data_patterns": {
            "has_monetary_values": True,
            "has_accounting_structure": True,
            "has_date_sequences": True
        }
    }
}

class DataClassificationService:
    """
    Service for classifying datasets into appropriate departments and module types.
    Uses pattern matching and machine learning techniques to determine the most
    likely classification for a given dataset.
    """
    
    def __init__(self, config_path: Optional[str] = None, threshold: float = 0.3):
        """
        Initialize the DataClassificationService with classification patterns and rules
        
        Args:
            config_path: Optional path to config file with custom patterns
            threshold: Confidence threshold for classification (0-1)
        """
        try:
            # Initialize with classification patterns and rules
            self.module_patterns = self._init_module_patterns()
            self.threshold = threshold
            self.data_mapper = DataMapper(config_path)
            logger.info("DataClassificationService initialized")
        except Exception as e:
            logger.error(f"Error initializing DataClassificationService: {str(e)}")
            raise
        
    def _init_module_patterns(self) -> Dict:
        """Initialize the classification patterns dictionary"""
        try:
            return classification_patterns
        except Exception as e:
            logger.error(f"Error initializing module patterns: {str(e)}")
            return {}
    
    def classify_dataset(self, df: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict:
        """
        Classify a dataset into appropriate department and module type
        
        Args:
            df: DataFrame to classify
            metadata: Optional metadata about the dataset
            
        Returns:
            Dictionary with classification results and confidence scores
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning("Empty DataFrame provided for classification")
                return {
                    "error": "Empty DataFrame provided",
                    "primary_classification": {
                        "department": "Unknown",
                        "module_type": "Unknown",
                        "confidence": 0.0
                    }
                }
                
            # Extract column information
            column_info = self._extract_column_info(df)
            
            # Detect data patterns
            data_patterns = self._detect_data_patterns(df)
            
            # Match against module patterns
            results = self._match_patterns(column_info, data_patterns)
            
            # If metadata contains hints, use them to improve classification
            if metadata:
                results = self._refine_with_metadata(results, metadata)
            
            # Return top matches with confidence scores
            classification_result = {
                "primary_classification": {
                    "department": results[0]["department"] if results else "Unknown",
                    "module_type": results[0]["module_type"] if results else "Unknown",
                    "confidence": results[0]["confidence"] if results else 0.0
                },
                "alternate_classifications": results[1:3] if len(results) > 1 else [],  # Return top 3 matches
                "column_analysis": column_info,
                "data_patterns_detected": data_patterns,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log the classification result
            logger.info(f"Dataset classified as {classification_result['primary_classification']['department']} - "
                        f"{classification_result['primary_classification']['module_type']} "
                        f"with confidence {classification_result['primary_classification']['confidence']:.2f}")
                        
            return classification_result
            
        except Exception as e:
            logger.error(f"Error in classify_dataset: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "primary_classification": {
                    "department": "Unknown",
                    "module_type": "Unknown",
                    "confidence": 0.0
                }
            }
    
    def _extract_column_info(self, df: pd.DataFrame) -> Dict:
        """Extract useful information about columns"""
        column_info = {}
        
        try:
            for col in df.columns:
                try:
                    # Get dtype information
                    dtype = str(df[col].dtype)
                    
                    # Analyze column name
                    col_lower = col.lower()
                    
                    # Get semantic type from data_mapper
                    semantic_type = self.data_mapper.get_semantic_type(col)
                    
                    column_info[col] = {
                        "dtype": dtype,
                        "name_tokens": re.findall(r'[a-zA-Z]+', col_lower),
                        "is_date": pd.api.types.is_datetime64_any_dtype(df[col]),
                        "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                        "is_categorical": pd.api.types.is_categorical_dtype(df[col]) or 
                                        (df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.2)
                                        if len(df) > 0 else False,
                        "null_percentage": (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0,
                        "semantic_type": semantic_type,
                        "unique_count": df[col].nunique() if len(df) > 0 else 0,
                        "unique_percentage": (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0
                    }
                    
                    # For text columns, analyze content
                    if df[col].dtype == 'object':
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            avg_len = sample.str.len().mean() if hasattr(sample, 'str') else 0
                            column_info[col]["avg_text_length"] = avg_len
                            column_info[col]["contains_urls"] = any(('http://' in str(x) or 'www.' in str(x)) 
                                                                for x in sample.head(10))
                            
                            # Check for common patterns
                            if avg_len > 0:
                                # Check for email pattern
                                email_pattern = r'[^@]+@[^@]+\.[^@]+'
                                column_info[col]["contains_emails"] = any(re.match(email_pattern, str(x)) 
                                                                        for x in sample.head(10) if isinstance(x, str))
                                
                                # Check for phone number pattern
                                phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
                                column_info[col]["contains_phones"] = any(re.search(phone_pattern, str(x)) 
                                                                        for x in sample.head(10) if isinstance(x, str))
                except Exception as col_error:
                    logger.warning(f"Error analyzing column {col}: {str(col_error)}")
                    column_info[col] = {
                        "error": str(col_error),
                        "dtype": str(df[col].dtype) if col in df else "unknown"
                    }
                    
        except Exception as e:
            logger.error(f"Error extracting column info: {str(e)}")
            
        return column_info
    
    def _detect_data_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect patterns in the dataset"""
        data_patterns = {}
        
        try:
            column_list = [col.lower() for col in df.columns]
            sample_data = df.head(50)  # Analyze a sample for patterns
            
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
            
            # Additional patterns detection
            data_patterns["has_missing_values"] = df.isna().any().any()
            data_patterns["has_duplicate_rows"] = df.duplicated().any()
            
            # Check for time series data
            if data_patterns["has_date_sequences"]:
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                if date_cols and any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
                    data_patterns["appears_to_be_time_series"] = True
                    
            # Check for transactional data
            if data_patterns["has_monetary_values"] and data_patterns["has_date_sequences"]:
                data_patterns["appears_to_be_transactional"] = True
                
        except Exception as e:
            logger.error(f"Error detecting data patterns: {str(e)}")
            
        return data_patterns
    
    def _match_patterns(self, column_info: Dict, data_patterns: Dict) -> List[Dict]:
        """Match column info and data patterns against module patterns"""
        results = []
        
        try:
            column_names = list(column_info.keys())
            
            for category, patterns in self.module_patterns.items():
                col_match_score = self._calculate_column_match_score(column_info, patterns["column_patterns"])
                data_pattern_score = self._calculate_data_pattern_score(data_patterns, patterns["data_patterns"])
                
                # Calculate weighted score
                total_score = col_match_score * 0.7 + data_pattern_score * 0.3
                
                # Only include if above threshold
                if total_score >= self.threshold:
                    results.append({
                        "department": patterns["department"],
                        "module_type": patterns["module_type"],
                        "confidence": total_score,
                        "category": category,
                        "column_match_score": col_match_score,
                        "data_pattern_score": data_pattern_score
                    })
            
            # If no results above threshold, include the best match anyway
            if not results and self.module_patterns:
                best_scores = []
                for category, patterns in self.module_patterns.items():
                    col_match_score = self._calculate_column_match_score(column_info, patterns["column_patterns"])
                    data_pattern_score = self._calculate_data_pattern_score(data_patterns, patterns["data_patterns"])
                    total_score = col_match_score * 0.7 + data_pattern_score * 0.3
                    
                    best_scores.append({
                        "department": patterns["department"],
                        "module_type": patterns["module_type"],
                        "confidence": total_score,
                        "category": category,
                        "column_match_score": col_match_score,
                        "data_pattern_score": data_pattern_score
                    })
                
                if best_scores:
                    results.append(max(best_scores, key=lambda x: x["confidence"]))
            
            # Sort by confidence score
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error matching patterns: {str(e)}")
            
        return results
    
    def _calculate_column_match_score(self, column_info: Dict, pattern_list: List[str]) -> float:
        """Calculate how well columns match expected patterns"""
        try:
            column_names = [col.lower() for col in column_info.keys()]
            
            if not pattern_list:
                return 0
                
            # Count matches
            matches = 0
            for pattern in pattern_list:
                if any(pattern in col_name for col_name in column_names):
                    matches += 1
            
            return matches / len(pattern_list)
            
        except Exception as e:
            logger.error(f"Error calculating column match score: {str(e)}")
            return 0
    
    def _calculate_data_pattern_score(self, data_patterns: Dict, expected_patterns: Dict) -> float:
        """Calculate how well data patterns match expected patterns"""
        try:
            if not expected_patterns:
                return 0
                
            matches = sum(1 for pattern, expected in expected_patterns.items() 
                        if data_patterns.get(pattern, False) == expected)
                        
            return matches / len(expected_patterns)
            
        except Exception as e:
            logger.error(f"Error calculating data pattern score: {str(e)}")
            return 0
    
    def _refine_with_metadata(self, results: List[Dict], metadata: Dict) -> List[Dict]:
        """Use metadata to refine classification results"""
        try:
            # If metadata contains explicit department or module_type, boost those matches
            if "department" in metadata and metadata["department"]:
                for result in results:
                    if result["department"].lower() == metadata["department"].lower():
                        result["confidence"] += 0.2
                        
            if "module_type" in metadata and metadata["module_type"]:
                for result in results:
                    if result["module_type"].lower() == metadata["module_type"].lower():
                        result["confidence"] += 0.2
                        
            # If metadata contains tags or keywords, use them for additional matching
            if "tags" in metadata and metadata["tags"]:
                for result in results:
                    category = result["category"].lower()
                    matching_tags = sum(1 for tag in metadata["tags"] 
                                      if tag.lower() in category or category in tag.lower())
                    if matching_tags > 0:
                        result["confidence"] += 0.1 * min(matching_tags, 2)  # Cap the boost
            
            # Re-sort after adjustments
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error refining with metadata: {str(e)}")
            return results
            
    def add_custom_pattern(self, category: str, department: str, module_type: str, 
                          column_patterns: List[str], data_patterns: Dict) -> bool:
        """
        Add a custom classification pattern
        
        Args:
            category: Category name for the pattern
            department: Department name
            module_type: Module type name
            column_patterns: List of column name patterns
            data_patterns: Dictionary of data patterns
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.module_patterns[category] = {
                "department": department,
                "module_type": module_type,
                "column_patterns": column_patterns,
                "data_patterns": data_patterns
            }
            
            logger.info(f"Added custom pattern for category: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom pattern: {str(e)}")
            return False
            
    def save_patterns_to_file(self, filepath: str) -> bool:
        """
        Save current classification patterns to a file
        
        Args:
            filepath: Path to save the patterns
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.module_patterns, f, indent=2)
                
            logger.info(f"Saved classification patterns to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving patterns to file: {str(e)}")
            return False
            
    def load_patterns_from_file(self, filepath: str) -> bool:
        """
        Load classification patterns from a file
        
        Args:
            filepath: Path to load the patterns from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                patterns = json.load(f)
                
            self.module_patterns = patterns
            logger.info(f"Loaded classification patterns from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading patterns from file: {str(e)}")
            return False
            
    def get_department_modules(self, department: str) -> List[str]:
        """
        Get all module types for a specific department
        
        Args:
            department: Department name
            
        Returns:
            List of module types
        """
        try:
            modules = []
            for category, patterns in self.module_patterns.items():
                if patterns["department"].lower() == department.lower():
                    modules.append(patterns["module_type"])
                    
            return list(set(modules))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error getting department modules: {str(e)}")
            return []
