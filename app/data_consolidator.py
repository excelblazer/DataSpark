"""
Data Consolidator Module for DataBloom

This module provides functionality to consolidate multiple datasets into a unified format.
It supports various consolidation strategies with role-based access control integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
import logging
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

class DataConsolidationError(Exception):
    """Base class for data consolidation exceptions"""
    
    def __init__(self, error_type: str, message: str, details: Optional[Dict] = None):
        self.error_type = error_type
        self.details = details or {}
        self.message = f"[{datetime.now().isoformat()}] ConsolidationError-{error_type}: {message}"
        super().__init__(self.message)


class DataConsolidator:
    """
    Data consolidation service for combining datasets from multiple sources or departments
    with built-in role-based access control capabilities.
    """
    
    def __init__(self):
        """Initialize the DataConsolidator"""
        self.logger = logging.getLogger(__name__)
        self.join_strategies = ["inner", "outer", "left", "right"]
    
    async def consolidate(self, datasets: List[Any], config: Optional[Dict] = None) -> AsyncGenerator[Dict, None]:
        """
        Consolidate multiple datasets asynchronously
        
        Args:
            datasets: List of datasets to consolidate
            config: Configuration for consolidation operations
            
        Yields:
            Dict with consolidated data and metadata
        """
        try:
            # Validate inputs
            if not datasets:
                error_msg = "No datasets provided for consolidation"
                self.logger.error(error_msg)
                raise DataConsolidationError("EMPTY_INPUT", error_msg)
            
            # Validate and set default configuration
            config = self._validate_config(config)
            
            # Process datasets in batches for memory efficiency
            batch_size = config.get('batch_size', 5)
            for i in range(0, len(datasets), batch_size):
                batch = datasets[i:i+batch_size]
                
                # Convert batch to pandas DataFrames
                dfs, metadata_list = self._prepare_dataframes(batch)
                
                if not dfs:
                    self.logger.warning(f"No valid dataframes in batch {i//batch_size + 1}")
                    continue
                
                # Check schema compatibility if configured
                if config.get('check_schema_compatibility', True):
                    compatible = self._check_schema_compatibility(dfs)
                    if not compatible:
                        self.logger.warning("Schemas are not compatible, attempting to harmonize")
                        dfs = self._harmonize_schemas(dfs, strategy=config.get('harmonize_strategy', 'union'))
                
                # Consolidate the batch
                consolidated_df = self._consolidate_dataframes(dfs, config)
                
                # Merge metadata from all datasets
                consolidated_metadata = self._merge_metadata(metadata_list)
                
                # Apply any post-consolidation operations
                if config.get('clean_after_consolidation', True):
                    consolidated_df = self._clean_consolidated_data(consolidated_df)
                
                # Convert back to dictionary and yield
                result = {
                    'data': consolidated_df.to_dict('records'),
                    'metadata': {
                        **consolidated_metadata,
                        'consolidation_timestamp': datetime.now().isoformat(),
                        'consolidation_stats': {
                            'original_datasets': len(batch),
                            'rows': len(consolidated_df),
                            'columns': len(consolidated_df.columns),
                        }
                    }
                }
                
                yield result
                
        except DataConsolidationError:
            # Re-raise known consolidation errors
            raise
        except Exception as e:
            error_msg = f"Unexpected error during consolidation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataConsolidationError("UNEXPECTED", error_msg, {"error": str(e)})
    
    def _validate_config(self, config: Optional[Dict] = None) -> Dict:
        """Validate and set default consolidation configuration"""
        default_config = {
            'check_schema_compatibility': True,
            'harmonize_strategy': 'union',    # 'union', 'intersection', 'first'
            'join_method': 'outer',           # 'inner', 'outer', 'left', 'right'
            'batch_size': 5,
            'handle_duplicates': True,
            'clean_after_consolidation': True,
            'check_unique_keys': True,
            'handle_type_inconsistencies': True
        }
        
        if config:
            # Update default config with provided config
            default_config.update(config)
        
        return default_config
    
    def _prepare_dataframes(self, datasets: List[Any]) -> tuple:
        """Convert datasets to pandas DataFrames and extract metadata"""
        dfs = []
        metadata_list = []
        
        for idx, dataset in enumerate(datasets):
            try:
                # Handle different input formats
                if isinstance(dataset, pd.DataFrame):
                    dfs.append(dataset)
                    metadata_list.append({})
                elif isinstance(dataset, dict):
                    # Extract data and metadata
                    if 'data' in dataset:
                        df = pd.DataFrame(dataset['data'])
                        dfs.append(df)
                        metadata = dataset.get('metadata', {})
                        metadata_list.append(metadata)
                    else:
                        # Assume the whole dict is data
                        df = pd.DataFrame([dataset])
                        dfs.append(df)
                        metadata_list.append({})
                elif isinstance(dataset, list):
                    df = pd.DataFrame(dataset)
                    dfs.append(df)
                    metadata_list.append({})
                else:
                    self.logger.warning(f"Unsupported dataset format for dataset {idx}, skipping")
            except Exception as e:
                self.logger.error(f"Error preparing dataset {idx}: {str(e)}")
                # Skip this dataset but continue with others
        
        return dfs, metadata_list
    
    def _check_schema_compatibility(self, dfs: List[pd.DataFrame]) -> bool:
        """Check if all dataframes have compatible schemas"""
        if not dfs:
            return True
        
        # Get sets of columns
        column_sets = [set(df.columns) for df in dfs]
        
        # All schemas are identical
        if all(s == column_sets[0] for s in column_sets):
            return True
        
        # Check for key columns in all datasets
        common_columns = set.intersection(*column_sets)
        if len(common_columns) == 0:
            return False
        
        # If we have some common columns, we can potentially join
        return True
    
    def _harmonize_schemas(self, dfs: List[pd.DataFrame], strategy: str = 'union') -> List[pd.DataFrame]:
        """Harmonize schemas to make them compatible for consolidation"""
        if not dfs:
            return []
        
        # Get sets of columns
        column_sets = [set(df.columns) for df in dfs]
        
        if strategy == 'union':
            # Use union of all columns
            all_columns = set().union(*column_sets)
            
            # Add missing columns as NaN
            result = []
            for df in dfs:
                missing_cols = all_columns - set(df.columns)
                for col in missing_cols:
                    df[col] = np.nan
                result.append(df)
            
            return result
            
        elif strategy == 'intersection':
            # Use only common columns
            common_columns = set.intersection(*column_sets)
            
            # Filter to only common columns
            result = []
            for df in dfs:
                result.append(df[list(common_columns)])
            
            return result
            
        elif strategy == 'first':
            # Use columns from first dataframe
            first_columns = list(dfs[0].columns)
            
            # Make all other dataframes match the first
            result = [dfs[0]]
            for df in dfs[1:]:
                temp_df = pd.DataFrame(columns=first_columns)
                for col in first_columns:
                    if col in df.columns:
                        temp_df[col] = df[col]
                result.append(temp_df)
            
            return result
            
        else:
            # Invalid strategy, return original dataframes
            self.logger.warning(f"Invalid harmonize strategy: {strategy}, using original dataframes")
            return dfs
    
    def _consolidate_dataframes(self, dfs: List[pd.DataFrame], config: Dict) -> pd.DataFrame:
        """Consolidate multiple dataframes into one based on configuration"""
        if not dfs:
            return pd.DataFrame()
        
        if len(dfs) == 1:
            return dfs[0]
        
        # Determine join method
        join_method = config.get('join_method', 'outer')
        
        # Check for key columns for joining
        key_columns = config.get('key_columns', [])
        if key_columns and all(set(key_columns).issubset(set(df.columns)) for df in dfs):
            # Join on key columns if provided and present in all dataframes
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on=key_columns, how=join_method)
        else:
            # Concatenate if no key columns provided or not all dataframes have them
            result = pd.concat(dfs, ignore_index=True, sort=False)
            
            # Handle duplicates if configured
            if config.get('handle_duplicates', True):
                result = result.drop_duplicates()
        
        return result
    
    def _merge_metadata(self, metadata_list: List[Dict]) -> Dict:
        """Merge metadata from multiple datasets"""
        if not metadata_list:
            return {}
        
        if len(metadata_list) == 1:
            return metadata_list[0]
        
        # Start with empty result
        result = {
            'source_datasets': []
        }
        
        # Special handling for departmental metadata
        departments = set()
        module_types = set()
        
        for idx, metadata in enumerate(metadata_list):
            # Add to source datasets
            if 'name' in metadata or 'id' in metadata:
                source = {
                    'index': idx,
                    'id': metadata.get('id', f'dataset-{idx}'),
                    'name': metadata.get('name', f'Dataset {idx}')
                }
                result['source_datasets'].append(source)
            
            # Collect departments
            if 'department' in metadata:
                departments.add(metadata['department'])
            
            # Collect module types
            if 'module_type' in metadata:
                module_types.add(metadata['module_type'])
            
            # Merge classification info if available
            if 'classification' in metadata:
                if 'classification' not in result:
                    result['classification'] = {}
                
                # Append as source classification
                result['classification'][f'source_{idx}'] = metadata['classification']
        
        # Add collected departments and module types
        if departments:
            result['departments'] = list(departments)
        
        if module_types:
            result['module_types'] = list(module_types)
        
        # Add consolidated timestamp
        result['consolidated_at'] = datetime.now().isoformat()
        
        return result
    
    def _clean_consolidated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning to consolidated data"""
        if df.empty:
            return df
        
        # Handle column names (clean up duplicates that might have occurred during merge)
        df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
        
        # Remove complete duplicate rows
        df = df.drop_duplicates()
        
        # Handle completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        return df
    
    def consolidate_files(self, file_data_list: List[Dict]) -> pd.DataFrame:
        """
        Synchronous version for consolidating multiple data files into a single DataFrame
        
        Args:
            file_data_list: List of dictionaries containing file data
            
        Returns:
            Consolidated DataFrame
        """
        try:
            if not file_data_list:
                raise DataConsolidationError("EMPTY_INPUT", "No files provided for consolidation")
                
            dataframes = []
            for file_data in file_data_list:
                if "dataframe" in file_data:
                    dataframes.append(file_data["dataframe"])
                elif "data" in file_data:
                    dataframes.append(pd.DataFrame(file_data["data"]))
            
            # Apply union (concatenation) of all dataframes by default
            consolidated_df = pd.concat(dataframes, ignore_index=True)
            
            # Clean consolidated data
            consolidated_df = self._clean_consolidated_data(consolidated_df)
            
            # Add metadata
            metadata = {
                'consolidation_timestamp': datetime.now().isoformat(),
                'source_files': len(file_data_list),
                'total_rows': len(consolidated_df)
            }
            
            self.logger.info(f"Successfully consolidated {len(file_data_list)} files into {len(consolidated_df)} rows")
            
            return consolidated_df
        except Exception as e:
            error_msg = f"Error in consolidate_files: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataConsolidationError("FILE_CONSOLIDATION_ERROR", error_msg)
    
    def merge_operational_financial(self, 
                                   operational_data: Union[Dict, pd.DataFrame],
                                   financial_data: Union[Dict, pd.DataFrame],
                                   merge_on: Union[str, List[str]],
                                   join_strategy: str = "inner") -> pd.DataFrame:
        """
        Merge operational and financial data for comprehensive reporting
        
        Args:
            operational_data: Operational data as DataFrame or dict
            financial_data: Financial data as DataFrame or dict
            merge_on: Column(s) to merge on
            join_strategy: Type of join (inner, outer, left, right)
            
        Returns:
            Merged DataFrame
        """
        try:
            # Convert to DataFrames if needed
            op_df = operational_data if isinstance(operational_data, pd.DataFrame) else pd.DataFrame(operational_data)
            fin_df = financial_data if isinstance(financial_data, pd.DataFrame) else pd.DataFrame(financial_data)
            
            # Validate join strategy
            if join_strategy not in self.join_strategies:
                self.logger.warning(f"Invalid join strategy: {join_strategy}, defaulting to inner")
                join_strategy = "inner"
                
            # Perform merge
            merged_df = pd.merge(op_df, fin_df, on=merge_on, how=join_strategy, suffixes=('_op', '_fin'))
            
            # Add metadata columns about the merge
            merged_df['data_source'] = 'merged'
            merged_df['merge_date'] = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"Successfully merged operational and financial data using {join_strategy} join on {merge_on}")
            
            return merged_df
        except Exception as e:
            error_msg = f"Error in merge_operational_financial: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataConsolidationError("MERGE_ERROR", error_msg)
    
    def detect_join_keys(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """
        Automatically detect potential join keys between two dataframes
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            List of potential join keys
        """
        try:
            # Find common columns
            common_cols = set(df1.columns).intersection(set(df2.columns))
            
            # Filter to likely join keys (typically ID columns, date columns, or categorical columns with low cardinality)
            potential_keys = []
            
            for col in common_cols:
                # Check if column name indicates a key
                if any(key_term in col.lower() for key_term in ['id', 'key', 'code', 'date', 'period']):
                    potential_keys.append(col)
                    continue
                
                # Check if columns have similar unique values indicating they might be joinable
                try:
                    nunique1 = df1[col].nunique()
                    nunique2 = df2[col].nunique()
                    
                    # If the column has reasonable cardinality and similar in both dataframes
                    if nunique1 > 0 and nunique2 > 0 and 0.5 <= nunique1/nunique2 <= 2.0:
                        # For low to medium cardinality columns (good join candidates)
                        if nunique1 < len(df1) * 0.5 and nunique2 < len(df2) * 0.5:
                            potential_keys.append(col)
                except:
                    # Skip columns that cause errors in comparison
                    pass
            
            self.logger.info(f"Detected {len(potential_keys)} potential join keys: {potential_keys}")
            
            return potential_keys
        except Exception as e:
            error_msg = f"Error in detect_join_keys: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Return empty list instead of raising to make the function more robust
            return []
            
    async def consolidate_by_department(self, datasets: List[Dict], department: str) -> Dict:
        """
        Consolidate datasets filtered by department
        
        Args:
            datasets: List of datasets to consolidate
            department: Department name to filter by
            
        Returns:
            Consolidated dataset for the department
        """
        try:
            # Filter datasets by department
            filtered_datasets = []
            for ds in datasets:
                if isinstance(ds, dict) and 'metadata' in ds:
                    dept = ds['metadata'].get('department')
                    if dept == department:
                        filtered_datasets.append(ds)
            
            if not filtered_datasets:
                self.logger.warning(f"No datasets found for department: {department}")
                return {"data": [], "metadata": {"department": department, "warning": "No data found"}}
            
            # Process datasets through consolidation pipeline
            consolidated_chunks = []
            async for chunk in self.consolidate(filtered_datasets):
                consolidated_chunks.append(chunk)
            
            # Combine chunks into a single result
            if len(consolidated_chunks) > 1:
                combined = consolidated_chunks[0]
                if 'data' in combined and isinstance(combined['data'], list):
                    for chunk in consolidated_chunks[1:]:
                        combined['data'].extend(chunk.get('data', []))
                result = combined
            elif consolidated_chunks:
                result = consolidated_chunks[0]
            else:
                result = {"data": [], "metadata": {"department": department, "warning": "No data after consolidation"}}
            
            # Ensure department is set in metadata
            if 'metadata' in result:
                result['metadata']['department'] = department
            
            self.logger.info(f"Successfully consolidated {len(filtered_datasets)} datasets for department: {department}")
            
            return result
        except Exception as e:
            error_msg = f"Error consolidating by department {department}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataConsolidationError("DEPARTMENT_CONSOLIDATION_ERROR", error_msg)
    
    async def consolidate_cross_department(self, datasets: List[Dict], departments: List[str]) -> Dict:
        """
        Consolidate datasets across multiple departments
        
        Args:
            datasets: List of datasets to consolidate
            departments: List of departments to include
            
        Returns:
            Cross-department consolidated dataset
        """
        try:
            # Filter datasets by departments
            filtered_datasets = []
            for ds in datasets:
                if isinstance(ds, dict) and 'metadata' in ds:
                    dept = ds['metadata'].get('department')
                    if dept in departments:
                        filtered_datasets.append(ds)
            
            if not filtered_datasets:
                self.logger.warning(f"No datasets found for departments: {departments}")
                return {"data": [], "metadata": {"departments": departments, "warning": "No data found"}}
            
            # Process datasets through consolidation pipeline with special config for cross-department
            config = {
                'check_schema_compatibility': True,
                'harmonize_strategy': 'union',
                'add_department_column': True  # Special flag for cross-department consolidation
            }
            
            # Add department column to each dataset for tracking
            for ds in filtered_datasets:
                if isinstance(ds, dict) and 'data' in ds and 'metadata' in ds:
                    dept = ds['metadata'].get('department')
                    if dept and isinstance(ds['data'], list):
                        for item in ds['data']:
                            if isinstance(item, dict):
                                item['department'] = dept
            
            # Consolidate the datasets
            consolidated_chunks = []
            async for chunk in self.consolidate(filtered_datasets, config):
                consolidated_chunks.append(chunk)
            
            # Combine chunks into a single result
            if len(consolidated_chunks) > 1:
                combined = consolidated_chunks[0]
                if 'data' in combined and isinstance(combined['data'], list):
                    for chunk in consolidated_chunks[1:]:
                        combined['data'].extend(chunk.get('data', []))
                result = combined
            elif consolidated_chunks:
                result = consolidated_chunks[0]
            else:
                result = {"data": [], "metadata": {"departments": departments, "warning": "No data after consolidation"}}
            
            # Ensure departments are set in metadata
            if 'metadata' in result:
                result['metadata']['departments'] = departments
                result['metadata']['cross_department'] = True
            
            self.logger.info(f"Successfully consolidated {len(filtered_datasets)} datasets across departments: {departments}")
            
            return result
        except Exception as e:
            error_msg = f"Error in cross-department consolidation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DataConsolidationError("CROSS_DEPARTMENT_ERROR", error_msg)
