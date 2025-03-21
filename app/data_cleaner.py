import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import json
from datetime import datetime

class DataCleaningError(Exception):
    """Base class for data cleaning exceptions"""
    
    def __init__(self, error_type: str, message: str, column: str = None):
        self.error_type = error_type
        self.column = column
        self.message = f"[{datetime.now().isoformat()}] CleaningError-{error_type}: {message}"
        super().__init__(self.message)


class DataCleaner:
    def __init__(self):
        self.scalers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
    def clean(self, data: Dict, config: Optional[Dict] = None) -> Dict:
        """
        Clean, normalize and identify missing data
        
        Args:
            data: Input data dictionary with values
            config: Configuration for cleaning operations
            
        Returns:
            Dict with cleaned data and cleaning report
        """
        # Validate and set default configuration
        config = self._validate_config(config)
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(data)
        
        # Original shape for reporting
        original_shape = df.shape
        
        # Generate data quality report
        quality_report = self._generate_quality_report(df)
        
        # Handle missing data if configured
        if config.get('handle_missing', True):
            df = self._handle_missing_values(df, method=config.get('impute_method', 'mean'))
        
        # Remove duplicates if configured
        if config.get('handle_duplicates', True):
            df, duplicate_count = self._remove_duplicates(df)
            quality_report['duplicates_removed'] = duplicate_count
        
        # Normalize data if configured
        if config.get('normalize', False):
            df = self._normalize_data(df, method=config.get('normalize_method', 'minmax'))
            quality_report['normalization_applied'] = config.get('normalize_method', 'minmax')
        
        # Detect outliers if configured
        if config.get('detect_outliers', True):
            outliers = self._detect_outliers(df)
            quality_report['outliers_detected'] = len(outliers)
            quality_report['outlier_indices'] = outliers
        
        # Final shape for reporting
        final_shape = df.shape
        quality_report['rows_before'] = original_shape[0]
        quality_report['rows_after'] = final_shape[0]
        
        # Return cleaned data and report
        return {
            'cleaned_data': df.to_dict('records'),
            'quality_report': quality_report
        }
    
    def _convert_to_dataframe(self, data: Dict) -> pd.DataFrame:
        """Convert dictionary data to pandas DataFrame"""
        if 'values' in data:
            return pd.DataFrame(data['values'])
        return pd.DataFrame(data)
    
    def _generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report"""
        # Count missing values by column
        missing_counts = df.isnull().sum().to_dict()
        missing_percentage = ((df.isnull().sum() / len(df)) * 100).to_dict()
        
        # Identify columns with all missing or all same values
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        all_missing_columns = [col for col in df.columns if df[col].isnull().sum() == len(df)]
        
        # Get data types
        data_types = df.dtypes.astype(str).to_dict()
        
        # Statistical summary for numeric columns
        numeric_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            numeric_stats[col] = {
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
            }
        
        # Value counts for categorical columns (top 5)
        categorical_stats = {}
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 10:  # Only for columns with reasonable number of categories
                value_counts = df[col].value_counts().head(5).to_dict()
                categorical_stats[col] = value_counts
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_counts': missing_counts,
            'missing_percentage': missing_percentage,
            'constant_columns': constant_columns,
            'all_missing_columns': all_missing_columns,
            'data_types': data_types,
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats
        }
        
        return report
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """Handle missing values using specified method"""
        df_cleaned = df.copy()
        
        if method == 'none':
            return df_cleaned
        
        # Handle numerical columns
        num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if method == 'mean':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif method == 'median':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif method == 'zero':
                df_cleaned[col] = df_cleaned[col].fillna(0)
        
        # Handle categorical and datetime columns
        cat_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if method == 'mode':
                mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None
                df_cleaned[col] = df_cleaned[col].fillna(mode_value)
            elif method == 'zero':
                df_cleaned[col] = df_cleaned[col].fillna('0')
        
        # For these methods, apply to entire dataframe
        if method == 'ffill':
            df_cleaned = df_cleaned.fillna(method='ffill')
        elif method == 'bfill':
            df_cleaned = df_cleaned.fillna(method='bfill')
        
        return df_cleaned
    
    def _remove_duplicates(self, df: pd.DataFrame) -> tuple:
        """Remove duplicate rows from dataframe"""
        duplicate_count = df.duplicated().sum()
        df_no_duplicates = df.drop_duplicates()
        return df_no_duplicates, duplicate_count
    
    def _normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize numerical data using specified method"""
        df_normalized = df.copy()
        
        # Only normalize numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if not numeric_cols.empty:
            try:
                scaler = self.scalers.get(method, self.scalers['minmax'])
                df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            except Exception as e:
                # In case of any errors, return original data
                pass
        
        return df_normalized
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[int]:
        """Detect outliers using IQR method"""
        outlier_indices = []
        
        # Only check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate Q1 and Q3
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            outlier_indices.extend(col_outliers)
        
        # Return unique indices
        return list(set(outlier_indices))
    
    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default cleaning configuration"""
        valid_config = {
            'handle_missing': True,
            'handle_duplicates': True,
            'normalize': False,
            'normalize_method': 'minmax',
            'detect_outliers': True,
            'impute_method': 'mean',  # Options: mean, median, mode, ffill, bfill, zero, none
            'chunk_size': 5000
        }
        
        if config:
            for key in config:
                if key not in valid_config:
                    raise DataCleaningError("config", f"Invalid config key: {key}")
            valid_config.update(config)
            
        return valid_config
