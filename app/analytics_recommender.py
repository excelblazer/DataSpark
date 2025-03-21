import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from .data_mapper import DataMapper
import re
import logging
from functools import lru_cache
import hashlib
import json
from datetime import datetime

logger = logging.getLogger("databloom.analytics_recommender")

class AnalyticsRecommender:
    """
    Recommends analytics operations based on dataset characteristics.
    Uses the centralized DataMapper for data type mappings.
    """
    
    def __init__(self, config_path: Optional[str] = None, cache_size: int = 128):
        """
        Initialize the AnalyticsRecommender with DataMapper
        
        Args:
            config_path: Optional path to config file for custom mappings
            cache_size: Size of the LRU cache for recommendations
        """
        try:
            self.data_mapper = DataMapper(config_path)
            self.cache_size = cache_size
            logger.info(f"AnalyticsRecommender initialized with cache size {cache_size}")
        except Exception as e:
            logger.error(f"Error initializing AnalyticsRecommender: {str(e)}")
            raise
    
    @lru_cache(maxsize=128)
    def _get_cached_recommendation(self, df_hash: str) -> Dict[str, Any]:
        """
        Internal method to support caching of recommendations
        
        Args:
            df_hash: Hash of the DataFrame to use as cache key
            
        Returns:
            Cached recommendations or None
        """
        # This is just a placeholder method to work with the lru_cache decorator
        # The actual implementation would need to handle the DataFrame hashing outside
        # since we can't hash a DataFrame directly
        pass
        
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """
        Create a hash of a DataFrame for caching purposes
        
        Args:
            df: DataFrame to hash
            
        Returns:
            String hash representation
        """
        # Create a simplified representation of the DataFrame for hashing
        df_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'shape': df.shape,
            'sample_values': {col: str(df[col].head(3).tolist()) for col in df.columns},
            'timestamp': datetime.now().strftime('%Y%m%d%H')  # Cache expires hourly
        }
        
        # Convert to string and hash
        df_str = json.dumps(df_info, sort_keys=True)
        return hashlib.md5(df_str.encode()).hexdigest()
        
    def recommend_analytics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a DataFrame and recommend analytics operations
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of analytics recommendations
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning("Empty DataFrame provided for recommendations")
                return {
                    "error": "Empty DataFrame provided",
                    "recommendations": {}
                }
                
            # Try to use cached results
            df_hash = self._hash_dataframe(df)
            try:
                cached_result = self._get_cached_recommendation(df_hash)
                if cached_result:
                    logger.info("Using cached recommendations")
                    return cached_result
            except Exception as cache_error:
                logger.warning(f"Cache retrieval failed: {str(cache_error)}")
            
            # Analyze the columns in the dataset
            column_analysis = self._analyze_columns(df)
            
            # Get recommendations for columns
            column_recommendations = {}
            for col, info in column_analysis.items():
                semantic_type = info["semantic_type"]
                column_recommendations[col] = {
                    "recommended_analytics": self.data_mapper.get_recommended_analytics(semantic_type),
                    "semantic_type": semantic_type
                }
            
            # Find relationships between columns
            relationships = self._detect_relationships(df, column_analysis)
            
            # Get visualization recommendations
            visualization_recommendations = {}
            for rel_type, cols in relationships.items():
                visualization_recommendations[rel_type] = {
                    "columns": cols,
                    "visualizations": self.data_mapper.get_visualization_types(rel_type)
                }
            
            # Compile all recommendations
            result = {
                "column_analysis": column_analysis,
                "column_recommendations": column_recommendations,
                "relationships": relationships,
                "visualization_recommendations": visualization_recommendations,
                "dataset_level_recommendations": self._get_dataset_level_recommendations(df, column_analysis),
                "data_quality_assessment": self._assess_data_quality(df),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            try:
                self._get_cached_recommendation.cache_clear()  # Clear old cache entries
                # We need to manually store this in the cache
                self._get_cached_recommendation.__wrapped__.__cache__[df_hash] = result
            except Exception as cache_error:
                logger.warning(f"Cache storage failed: {str(cache_error)}")
                
            return result
        except Exception as e:
            logger.error(f"Error in recommend_analytics: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "recommendations": {}
            }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze columns to determine their semantic types
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of column information
        """
        column_info = {}
        
        for col in df.columns:
            try:
                # Get column name tokens for semantic type detection
                col_lower = col.lower()
                
                # Determine basic data type
                dtype = str(df[col].dtype)
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
                is_categorical = pd.api.types.is_categorical_dtype(df[col]) or (
                    df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.2
                ) if len(df) > 0 else False
                
                # Determine semantic type based on column name
                semantic_type = self.data_mapper.get_semantic_type(col)
                
                # If semantic type is unknown, try to infer from data type
                if semantic_type == "unknown":
                    if is_datetime:
                        semantic_type = "datetime"
                    elif is_numeric:
                        semantic_type = "numerical"
                    elif is_categorical:
                        semantic_type = "categorical"
                    elif df[col].dtype == "object":
                        # Check if it's potentially text data
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            avg_len = sample.str.len().mean() if hasattr(sample, 'str') else 0
                            if avg_len > 50:  # Arbitrary threshold for text
                                semantic_type = "text"
                            else:
                                semantic_type = "categorical"
                
                # Store column information
                column_info[col] = {
                    "dtype": dtype,
                    "is_numeric": is_numeric,
                    "is_datetime": is_datetime,
                    "is_categorical": is_categorical,
                    "semantic_type": semantic_type,
                    "unique_values": df[col].nunique() if len(df) > 0 else 0,
                    "null_percentage": (df[col].isna().sum() / len(df)) * 100 if len(df) > 0 else 0,
                    "statistics": self._get_column_statistics(df[col], is_numeric, is_datetime)
                }
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {str(e)}")
                column_info[col] = {
                    "error": str(e),
                    "semantic_type": "unknown"
                }
                
        return column_info
    
    def _get_column_statistics(self, series: pd.Series, is_numeric: bool, is_datetime: bool) -> Dict[str, Any]:
        """
        Get statistics for a column based on its type
        
        Args:
            series: Series to analyze
            is_numeric: Whether the series is numeric
            is_datetime: Whether the series is datetime
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Skip if series is empty
        if series.empty:
            return stats
            
        try:
            # Common statistics
            stats["count"] = len(series)
            stats["null_count"] = series.isna().sum()
            
            if is_numeric:
                # Numeric statistics
                stats["min"] = float(series.min()) if not pd.isna(series.min()) else None
                stats["max"] = float(series.max()) if not pd.isna(series.max()) else None
                stats["mean"] = float(series.mean()) if not pd.isna(series.mean()) else None
                stats["median"] = float(series.median()) if not pd.isna(series.median()) else None
                stats["std"] = float(series.std()) if not pd.isna(series.std()) else None
                
                # Check for outliers using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outlier_low = Q1 - 1.5 * IQR
                outlier_high = Q3 + 1.5 * IQR
                outliers = series[(series < outlier_low) | (series > outlier_high)]
                stats["outlier_count"] = len(outliers)
                stats["outlier_percentage"] = (len(outliers) / len(series)) * 100
                
            elif is_datetime:
                # Datetime statistics
                stats["min"] = series.min().isoformat() if not pd.isna(series.min()) else None
                stats["max"] = series.max().isoformat() if not pd.isna(series.max()) else None
                stats["range_days"] = (series.max() - series.min()).days if not pd.isna(series.min()) and not pd.isna(series.max()) else None
                
            else:
                # Categorical/text statistics
                value_counts = series.value_counts()
                if not value_counts.empty:
                    stats["most_common"] = value_counts.index[0]
                    stats["most_common_count"] = int(value_counts.iloc[0])
                    stats["unique_count"] = len(value_counts)
                    stats["unique_percentage"] = (len(value_counts) / len(series)) * 100
        
        except Exception as e:
            logger.warning(f"Error calculating statistics: {str(e)}")
            stats["error"] = str(e)
            
        return stats
    
    def _detect_relationships(self, df: pd.DataFrame, column_analysis: Dict) -> Dict:
        """
        Detect potential relationships between columns
        
        Args:
            df: DataFrame to analyze
            column_analysis: Column analysis information
            
        Returns:
            Dictionary of detected relationships
        """
        relationships = {
            "numerical_vs_numerical": [],
            "numerical_vs_categorical": [],
            "categorical_vs_categorical": [],
            "numerical_distribution": [],
            "categorical_distribution": [],
            "time_series": [],
            "geospatial": []
        }
        
        try:
            numerical_cols = [col for col, info in column_analysis.items() 
                            if info["semantic_type"] == "numerical"]
            categorical_cols = [col for col, info in column_analysis.items() 
                                if info["semantic_type"] == "categorical"]
            datetime_cols = [col for col, info in column_analysis.items() 
                            if info["semantic_type"] == "datetime"]
            geospatial_cols = [col for col, info in column_analysis.items() 
                            if info["semantic_type"] == "geospatial"]
            
            # Numerical vs numerical relationships
            for i, col1 in enumerate(numerical_cols):
                relationships["numerical_distribution"].append(col1)
                for col2 in numerical_cols[i+1:]:
                    relationships["numerical_vs_numerical"].append((col1, col2))
            
            # Numerical vs categorical relationships
            for num_col in numerical_cols:
                for cat_col in categorical_cols:
                    relationships["numerical_vs_categorical"].append((num_col, cat_col))
            
            # Categorical vs categorical relationships
            for i, col1 in enumerate(categorical_cols):
                relationships["categorical_distribution"].append(col1)
                for col2 in categorical_cols[i+1:]:
                    relationships["categorical_vs_categorical"].append((col1, col2))
            
            # Time series relationships
            for date_col in datetime_cols:
                for num_col in numerical_cols:
                    relationships["time_series"].append((date_col, num_col))
            
            # Geospatial relationships
            if len(geospatial_cols) >= 2:
                relationships["geospatial"] = geospatial_cols
                
            # Calculate correlation matrix for numerical columns if there are at least 2
            if len(numerical_cols) >= 2:
                try:
                    corr_matrix = df[numerical_cols].corr().abs()
                    # Find highly correlated pairs (above 0.7)
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.7:
                                high_corr_pairs.append((
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    round(corr_matrix.iloc[i, j], 2)
                                ))
                    
                    if high_corr_pairs:
                        relationships["highly_correlated"] = high_corr_pairs
                except Exception as corr_error:
                    logger.warning(f"Error calculating correlations: {str(corr_error)}")
        
        except Exception as e:
            logger.error(f"Error detecting relationships: {str(e)}")
            
        return relationships
    
    def _get_dataset_level_recommendations(self, df: pd.DataFrame, column_analysis: Dict) -> List[str]:
        """
        Get recommendations for dataset-level analytics
        
        Args:
            df: DataFrame to analyze
            column_analysis: Column analysis information
            
        Returns:
            List of recommended dataset-level analytics
        """
        recommendations = []
        
        try:
            # Check if we have datetime columns
            has_datetime = any(info["semantic_type"] == "datetime" for info in column_analysis.values())
            
            # Check if we have numerical columns
            has_numerical = any(info["semantic_type"] == "numerical" for info in column_analysis.values())
            
            # Check if we have categorical columns
            has_categorical = any(info["semantic_type"] == "categorical" for info in column_analysis.values())
            
            # Check if we have text columns
            has_text = any(info["semantic_type"] == "text" for info in column_analysis.values())
            
            # Dataset-level recommendations
            if has_datetime and has_numerical:
                recommendations.extend(["time_series_analysis", "trend_detection", "seasonality_analysis", "forecasting"])
                
            if has_numerical:
                recommendations.extend(["correlation_analysis", "principal_component_analysis"])
                
                # If sufficient numerical columns, recommend clustering
                if sum(1 for info in column_analysis.values() if info["semantic_type"] == "numerical") >= 3:
                    recommendations.append("clustering")
                    
            if has_categorical and has_numerical:
                recommendations.extend(["group_comparison", "anova_analysis", "chi_square_analysis"])
                
            if has_text:
                recommendations.extend(["sentiment_analysis", "keyword_extraction", "topic_modeling"])
                
            # Check for time-based patterns if we have datetime and sufficient data
            if has_datetime and len(df) > 30:
                recommendations.append("time_pattern_detection")
                
            # Add data quality recommendations based on assessment
            data_quality = self._assess_data_quality(df)
            if data_quality["missing_data_percentage"] > 5:
                recommendations.append("missing_data_imputation")
                
            if data_quality["outlier_percentage"] > 5:
                recommendations.append("outlier_treatment")
                
            # Add machine learning recommendations if appropriate
            if has_numerical and len(df) > 100:
                if has_categorical:
                    recommendations.append("classification_models")
                else:
                    recommendations.append("regression_models")
        
        except Exception as e:
            logger.error(f"Error generating dataset recommendations: {str(e)}")
            
        return recommendations
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess the quality of the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        quality = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_data_percentage": 0,
            "outlier_percentage": 0,
            "duplicate_rows_percentage": 0,
            "issues": []
        }
        
        try:
            if df.empty:
                quality["issues"].append("Empty dataset")
                return quality
                
            # Calculate missing data percentage
            missing_data = df.isna().sum().sum()
            quality["missing_data_percentage"] = (missing_data / (len(df) * len(df.columns))) * 100
            
            if quality["missing_data_percentage"] > 5:
                quality["issues"].append(f"High percentage of missing data: {quality['missing_data_percentage']:.2f}%")
                
            # Calculate duplicate rows
            duplicate_count = df.duplicated().sum()
            quality["duplicate_rows_percentage"] = (duplicate_count / len(df)) * 100
            
            if quality["duplicate_rows_percentage"] > 1:
                quality["issues"].append(f"Dataset contains {quality['duplicate_rows_percentage']:.2f}% duplicate rows")
                
            # Detect outliers in numerical columns
            outlier_count = 0
            total_numeric_values = 0
            
            for col in df.select_dtypes(include=['number']).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_low = Q1 - 1.5 * IQR
                outlier_high = Q3 + 1.5 * IQR
                outliers = df[(df[col] < outlier_low) | (df[col] > outlier_high)]
                outlier_count += len(outliers)
                total_numeric_values += len(df[col].dropna())
                
            if total_numeric_values > 0:
                quality["outlier_percentage"] = (outlier_count / total_numeric_values) * 100
                
                if quality["outlier_percentage"] > 5:
                    quality["issues"].append(f"High percentage of outliers: {quality['outlier_percentage']:.2f}%")
            
            # Check for skewed distributions
            for col in df.select_dtypes(include=['number']).columns:
                if abs(df[col].skew()) > 1:
                    quality["issues"].append(f"Column '{col}' has a skewed distribution (skew: {df[col].skew():.2f})")
                    
            # Check for constant columns
            for col in df.columns:
                if df[col].nunique() == 1:
                    quality["issues"].append(f"Column '{col}' has only one unique value")
                    
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            quality["issues"].append(f"Error during quality assessment: {str(e)}")
            
        return quality
        
    def recommend_for_specific_columns(self, columns: List[str], df: pd.DataFrame) -> Dict:
        """
        Get recommendations for specific columns
        
        Args:
            columns: List of column names to analyze
            df: DataFrame containing the columns
            
        Returns:
            Dictionary of recommendations for the specified columns
        """
        try:
            # Validate columns
            invalid_columns = [col for col in columns if col not in df.columns]
            if invalid_columns:
                logger.warning(f"Invalid columns specified: {invalid_columns}")
                return {
                    "error": f"Invalid columns: {invalid_columns}",
                    "valid_columns": [col for col in df.columns]
                }
                
            # Filter the DataFrame to include only the specified columns
            subset_df = df[columns].copy()
            
            # Get recommendations for the subset
            recommendations = self.recommend_analytics(subset_df)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error in recommend_for_specific_columns: {str(e)}")
            return {
                "error": str(e),
                "recommendations": {}
            }
            
    def get_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Calculate correlation matrix for numerical columns
        
        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Dictionary with correlation matrix and related information
        """
        try:
            # Get numerical columns
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numerical_cols) < 2:
                return {
                    "error": "Insufficient numerical columns for correlation analysis",
                    "numerical_columns": numerical_cols
                }
                
            # Calculate correlation matrix
            corr_matrix = df[numerical_cols].corr(method=method)
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": round(float(corr_matrix.iloc[i, j]), 2)
                        })
            
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "method": method,
                "numerical_columns": numerical_cols,
                "high_correlations": high_corr_pairs
            }
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return {
                "error": str(e)
            }
