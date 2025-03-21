import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime
import logging
from .warehouse_manager import WarehouseManager
from .config import settings

logger = logging.getLogger("databloom.cross_module_analysis")

class CrossModuleAnalysis:
    """Service for performing analysis across different data modules"""
    
    def __init__(self, warehouse_manager: WarehouseManager = None):
        """Initialize with a warehouse manager instance"""
        self.warehouse_manager = warehouse_manager or WarehouseManager()
        
    def join_datasets(self, 
                     primary_dataset_id: str, 
                     secondary_dataset_id: str,
                     join_columns: Dict[str, str],
                     user_context: Dict = None) -> pd.DataFrame:
        """
        Join two datasets from different modules
        
        Args:
            primary_dataset_id: ID of the primary dataset
            secondary_dataset_id: ID of the secondary dataset
            join_columns: Dictionary mapping column names between datasets
                          e.g. {"primary_column": "secondary_column"}
            user_context: User context for access control
            
        Returns:
            DataFrame with joined data
        """
        try:
            logger.debug(f"Attempting to join datasets {primary_dataset_id} and {secondary_dataset_id}")
            
            # Retrieve both datasets
            primary_df = self.warehouse_manager.retrieve(primary_dataset_id, user_context=user_context)
            secondary_df = self.warehouse_manager.retrieve(secondary_dataset_id, user_context=user_context)
            
            # Get the primary and secondary column names
            primary_col = list(join_columns.keys())[0]
            secondary_col = join_columns[primary_col]
            
            # Validate columns exist
            if primary_col not in primary_df.columns:
                raise ValueError(f"Primary column {primary_col} not found in dataset {primary_dataset_id}")
            if secondary_col not in secondary_df.columns:
                raise ValueError(f"Secondary column {secondary_col} not found in dataset {secondary_dataset_id}")
            
            # Rename secondary column to match primary for join
            secondary_df = secondary_df.rename(columns={secondary_col: primary_col})
            
            # Join the datasets
            joined_df = pd.merge(
                primary_df, 
                secondary_df,
                on=primary_col,
                how='inner',
                suffixes=('', '_secondary')
            )
            
            # Log join statistics
            logger.info(f"Successfully joined datasets {primary_dataset_id} and {secondary_dataset_id}. "
                       f"Result has {len(joined_df)} rows and {len(joined_df.columns)} columns.")
            
            return joined_df
        except Exception as e:
            logger.error(f"Error joining datasets: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
            raise
    
    def find_related_datasets(self, dataset_id: str, user_context: Dict = None) -> List[Dict]:
        """
        Find datasets that might be related to this one
        
        Args:
            dataset_id: ID of the dataset to find relations for
            user_context: User context for access control
            
        Returns:
            List of related dataset metadata
        """
        logger.debug(f"Finding related datasets for {dataset_id}")
        session = None
        try:
            # Get the dataset metadata
            Session = sessionmaker(bind=self.warehouse_manager.engine)
            session = Session()
            
            # Get catalog entry
            result = session.execute(
                self.warehouse_manager.data_catalog.select().where(
                    self.warehouse_manager.data_catalog.c.id == dataset_id
                )
            ).fetchone()
            
            if not result:
                logger.warning(f"Dataset with ID {dataset_id} not found")
                raise ValueError(f"Dataset with ID {dataset_id} not found")
                
            dataset_info = dict(result)
            module_type = dataset_info.get('module_type')
            department = dataset_info.get('department')
            
            # Get related module types from registry
            from .module_registry import DataModuleRegistry
            registry = DataModuleRegistry()
            module = registry.get_module(module_type)
            
            if not module:
                logger.info(f"No module definition found for {module_type}, searching by department {department}")
                # No module definition found, try to find datasets from same department
                related_datasets = self.warehouse_manager.list_datasets(
                    department=department,
                    user_context=user_context
                )
                # Filter out the current dataset
                related_datasets = [ds for ds in related_datasets if ds['id'] != dataset_id]
                logger.info(f"Found {len(related_datasets)} datasets in department {department}")
                return related_datasets
            
            # Get related module definitions
            related_modules = registry.get_related_modules(module_type)
            related_module_names = [m.name for m in related_modules]
            logger.debug(f"Found {len(related_modules)} related module types: {', '.join(related_module_names)}")
            
            # Find datasets of related module types
            all_related_datasets = []
            for related_module in related_module_names:
                module_datasets = self.warehouse_manager.list_datasets(
                    module_type=related_module,
                    user_context=user_context
                )
                all_related_datasets.extend(module_datasets)
            
            logger.info(f"Found {len(all_related_datasets)} related datasets across {len(related_module_names)} module types")
            return all_related_datasets
            
        except Exception as e:
            logger.error(f"Error finding related datasets: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
            raise
        finally:
            if session:
                session.close()

    def suggest_join_columns(self, 
                           primary_dataset_id: str, 
                           secondary_dataset_id: str,
                           user_context: Dict = None) -> List[Dict[str, str]]:
        """
        Suggest potential join columns between two datasets
        
        Args:
            primary_dataset_id: ID of the primary dataset
            secondary_dataset_id: ID of the secondary dataset
            user_context: User context for access control
            
        Returns:
            List of possible column pairs for joining
        """
        logger.debug(f"Finding potential join columns between {primary_dataset_id} and {secondary_dataset_id}")
        session = None
        try:
            # Get dataset schemas
            Session = sessionmaker(bind=self.warehouse_manager.engine)
            session = Session()
            
            # Get primary dataset schema
            primary_result = session.execute(
                self.warehouse_manager.data_catalog.select().where(
                    self.warehouse_manager.data_catalog.c.id == primary_dataset_id
                )
            ).fetchone()
            
            if not primary_result:
                logger.warning(f"Primary dataset {primary_dataset_id} not found")
                raise ValueError(f"Primary dataset {primary_dataset_id} not found")
                
            # Get secondary dataset schema
            secondary_result = session.execute(
                self.warehouse_manager.data_catalog.select().where(
                    self.warehouse_manager.data_catalog.c.id == secondary_dataset_id
                )
            ).fetchone()
            
            if not secondary_result:
                logger.warning(f"Secondary dataset {secondary_dataset_id} not found")
                raise ValueError(f"Secondary dataset {secondary_dataset_id} not found")
                
            primary_schema = dict(primary_result).get('schema', {})
            secondary_schema = dict(secondary_result).get('schema', {})
            
            if not primary_schema or not secondary_schema:
                logger.warning("One or both datasets have no schema information")
                raise ValueError("One or both datasets have no schema information")
            
            # Find potential join columns
            potential_joins = []
            
            # Check for exact column name matches
            for primary_col in primary_schema.keys():
                if primary_col in secondary_schema:
                    potential_joins.append({primary_col: primary_col})
                    logger.debug(f"Found exact column match: {primary_col}")
            
            # Check for common ID patterns
            common_id_suffixes = ['_id', 'id', '_key', '_code']
            for primary_col in primary_schema.keys():
                for secondary_col in secondary_schema.keys():
                    # Skip if already added
                    if {primary_col: secondary_col} in potential_joins:
                        continue
                        
                    # Check if both have common ID pattern
                    if any(primary_col.endswith(suffix) for suffix in common_id_suffixes) and \
                       any(secondary_col.endswith(suffix) for suffix in common_id_suffixes):
                        
                        # Check for semantic similarities
                        primary_base = primary_col.split('_')[0] 
                        secondary_base = secondary_col.split('_')[0]
                        
                        if primary_base == secondary_base:
                            potential_joins.append({primary_col: secondary_col})
                            logger.debug(f"Found ID pattern match: {primary_col} -> {secondary_col}")
            
            # Add special case for common joins
            # Customer data often joins with many datasets
            if 'customer_id' in primary_schema and 'customer_id' in secondary_schema:
                potential_joins.append({'customer_id': 'customer_id'})
                logger.debug("Found customer_id join")
                
            # Product data also frequently joins
            if 'product_id' in primary_schema and 'product_id' in secondary_schema:
                potential_joins.append({'product_id': 'product_id'})
                logger.debug("Found product_id join")
                
            logger.info(f"Found {len(potential_joins)} potential join column pairs")
            return potential_joins
        except Exception as e:
            logger.error(f"Error suggesting join columns: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
            raise
        finally:
            if session:
                session.close()

    def analyze_cross_metrics(self,
                            datasets: List[str],
                            metrics: List[Dict],
                            user_context: Dict = None) -> Dict:
        """
        Analyze metrics across multiple datasets
        
        Args:
            datasets: List of dataset IDs to analyze
            metrics: List of metric definitions
                     Example: [
                         {
                             "name": "revenue_by_campaign",
                             "primary_dataset": "sales",
                             "secondary_dataset": "marketing",
                             "primary_column": "revenue",
                             "secondary_column": "campaign_name",
                             "operation": "sum"
                         }
                     ]
            user_context: User context for access control
            
        Returns:
            Dictionary with analysis results
        """
        logger.debug(f"Analyzing cross metrics for {len(datasets)} datasets and {len(metrics)} metrics")
        all_results = {}
        
        # Process each metric
        for metric in metrics:
            try:
                metric_name = metric.get("name", "unnamed_metric")
                logger.debug(f"Processing metric: {metric_name}")
                
                # Get relevant datasets
                primary_dataset_id = next((ds for ds in datasets if metric["primary_dataset"] in ds), None)
                secondary_dataset_id = next((ds for ds in datasets if metric["secondary_dataset"] in ds), None)
                
                if not primary_dataset_id or not secondary_dataset_id:
                    error_msg = f"Dataset not found for metric {metric_name}"
                    logger.warning(error_msg)
                    all_results[metric_name] = {"error": error_msg}
                    continue
                
                # Determine join columns
                join_suggestions = self.suggest_join_columns(
                    primary_dataset_id, 
                    secondary_dataset_id,
                    user_context
                )
                
                if not join_suggestions:
                    error_msg = f"No join columns found for metric {metric_name}"
                    logger.warning(error_msg)
                    all_results[metric_name] = {"error": error_msg}
                    continue
                
                # Use first suggested join
                join_columns = join_suggestions[0]
                logger.debug(f"Using join columns: {join_columns}")
                
                # Join datasets
                try:
                    joined_df = self.join_datasets(
                        primary_dataset_id, 
                        secondary_dataset_id,
                        join_columns,
                        user_context
                    )
                    
                    # Validate required columns exist
                    if metric["primary_column"] not in joined_df.columns:
                        raise ValueError(f"Primary column {metric['primary_column']} not found in joined dataset")
                    if metric["secondary_column"] not in joined_df.columns:
                        raise ValueError(f"Secondary column {metric['secondary_column']} not found in joined dataset")
                    
                    # Apply operation
                    operation = metric["operation"].lower()
                    if operation == "sum":
                        result = joined_df.groupby(metric["secondary_column"])[metric["primary_column"]].sum()
                    elif operation == "avg":
                        result = joined_df.groupby(metric["secondary_column"])[metric["primary_column"]].mean()
                    elif operation == "count":
                        result = joined_df.groupby(metric["secondary_column"])[metric["primary_column"]].count()
                    elif operation == "min":
                        result = joined_df.groupby(metric["secondary_column"])[metric["primary_column"]].min()
                    elif operation == "max":
                        result = joined_df.groupby(metric["secondary_column"])[metric["primary_column"]].max()
                    else:
                        error_msg = f"Unsupported operation {operation} for metric {metric_name}"
                        logger.warning(error_msg)
                        all_results[metric_name] = {"error": error_msg}
                        continue
                    
                    result_dict = result.to_dict()
                    logger.info(f"Successfully calculated metric {metric_name} with {len(result_dict)} results")
                    all_results[metric_name] = result_dict
                    
                except Exception as e:
                    error_msg = f"Error processing metric {metric_name}: {str(e)}"
                    logger.error(error_msg, exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
                    all_results[metric_name] = {"error": error_msg}
                    
            except Exception as e:
                error_msg = f"Error processing metric {metric.get('name', 'unnamed_metric')}: {str(e)}"
                logger.error(error_msg, exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
                all_results[metric.get("name", "unnamed_metric")] = {"error": error_msg}
                
        return all_results
    
    def detect_correlations(self, datasets: List[str], user_context: Dict = None) -> Dict:
        """
        Detect correlations between key metrics across datasets
        
        Args:
            datasets: List of dataset IDs to analyze
            user_context: User context for access control
            
        Returns:
            Dictionary with correlation analysis
        """
        logger.debug(f"Detecting correlations between {len(datasets)} datasets")
        
        # Get datasets
        all_dfs = {}
        for dataset_id in datasets:
            try:
                logger.debug(f"Loading dataset {dataset_id}")
                df = self.warehouse_manager.retrieve(dataset_id, user_context=user_context)
                all_dfs[dataset_id] = df
                logger.debug(f"Successfully loaded dataset {dataset_id} with shape {df.shape}")
            except Exception as e:
                logger.warning(f"Failed to load dataset {dataset_id}: {str(e)}")
                continue
                
        if len(all_dfs) < 2:
            error_msg = "Not enough datasets to analyze correlations"
            logger.warning(error_msg)
            return {"error": error_msg}
            
        # Identify numeric columns in each dataset
        numeric_columns = {}
        for dataset_id, df in all_dfs.items():
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            numeric_columns[dataset_id] = numeric_cols
            logger.debug(f"Found {len(numeric_cols)} numeric columns in dataset {dataset_id}")
            
        # Create correlation matrix between datasets
        correlations = {}
        
        # Get dataset pairs
        from itertools import combinations
        dataset_pairs = list(combinations(all_dfs.items(), 2))
        logger.debug(f"Analyzing {len(dataset_pairs)} dataset pairs")
        
        for (dataset_id1, df1), (dataset_id2, df2) in dataset_pairs:
            pair_key = f"{dataset_id1}_{dataset_id2}"
            logger.debug(f"Processing dataset pair: {pair_key}")
            
            try:
                # Find common timestamps or date columns for alignment
                date_cols1 = [col for col in df1.columns if 'date' in col.lower() or 'time' in col.lower()]
                date_cols2 = [col for col in df2.columns if 'date' in col.lower() or 'time' in col.lower()]
                
                # If no date columns, try using index alignment
                if not date_cols1 or not date_cols2:
                    logger.debug(f"No date columns found for pair {pair_key}, using index alignment")
                    # Resample to ensure same length if needed
                    if len(df1) != len(df2):
                        min_len = min(len(df1), len(df2))
                        df1_sample = df1.sample(min_len) if len(df1) > min_len else df1
                        df2_sample = df2.sample(min_len) if len(df2) > min_len else df2
                        logger.debug(f"Resampled datasets to length {min_len}")
                    else:
                        df1_sample, df2_sample = df1, df2
                        
                    # Calculate correlations between numeric columns
                    corr_matrix = {}
                    for col1 in numeric_columns[dataset_id1]:
                        corr_matrix[col1] = {}
                        for col2 in numeric_columns[dataset_id2]:
                            try:
                                corr = df1_sample[col1].corr(df2_sample[col2])
                                if not pd.isna(corr):
                                    corr_matrix[col1][col2] = corr
                            except Exception as e:
                                logger.debug(f"Failed to calculate correlation between {col1} and {col2}: {str(e)}")
                                continue
                            
                    correlations[pair_key] = corr_matrix
                else:
                    logger.debug(f"Using date alignment for pair {pair_key}")
                    # If date columns exist, try to align on date
                    # Use first date column from each dataset
                    date_col1 = date_cols1[0]
                    date_col2 = date_cols2[0]
                    
                    # Convert to datetime if not already
                    try:
                        df1[date_col1] = pd.to_datetime(df1[date_col1])
                        df2[date_col2] = pd.to_datetime(df2[date_col2])
                        
                        # Set as index for both
                        df1_dated = df1.set_index(date_col1)
                        df2_dated = df2.set_index(date_col2)
                        
                        # Resample to daily if needed
                        df1_daily = df1_dated.resample('D').mean()
                        df2_daily = df2_dated.resample('D').mean()
                        
                        # Find common dates
                        common_dates = df1_daily.index.intersection(df2_daily.index)
                        
                        if len(common_dates) > 0:
                            logger.debug(f"Found {len(common_dates)} common dates for time series alignment")
                            # Filter to common dates
                            df1_aligned = df1_daily.loc[common_dates]
                            df2_aligned = df2_daily.loc[common_dates]
                            
                            # Calculate correlations
                            corr_matrix = {}
                            for col1 in df1_aligned.columns:
                                if col1 in numeric_columns[dataset_id1]:
                                    corr_matrix[col1] = {}
                                    for col2 in df2_aligned.columns:
                                        if col2 in numeric_columns[dataset_id2]:
                                            try:
                                                corr = df1_aligned[col1].corr(df2_aligned[col2])
                                                if not pd.isna(corr):
                                                    corr_matrix[col1][col2] = corr
                                            except Exception as e:
                                                logger.debug(f"Failed to calculate time-aligned correlation between {col1} and {col2}: {str(e)}")
                                                continue
                                            
                            correlations[pair_key] = corr_matrix
                        else:
                            logger.warning(f"No common dates found for time series alignment in pair {pair_key}")
                    except Exception as e:
                        logger.warning(f"Failed to perform time series alignment for pair {pair_key}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Error processing dataset pair {pair_key}: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
                continue
                    
        # Filter to significant correlations
        significant_correlations = {}
        correlation_threshold = settings.CORRELATION_SIGNIFICANCE_THRESHOLD
        logger.debug(f"Filtering correlations with threshold {correlation_threshold}")
        
        for pair, corr_matrix in correlations.items():
            significant_pair = {}
            for col1, col2_values in corr_matrix.items():
                significant_col = {col2: corr for col2, corr in col2_values.items() if abs(corr) > correlation_threshold}
                if significant_col:
                    significant_pair[col1] = significant_col
            if significant_pair:
                significant_correlations[pair] = significant_pair
                
        logger.info(f"Found {len(significant_correlations)} dataset pairs with significant correlations")
        return {
            "all_correlations": correlations,
            "significant_correlations": significant_correlations
        }
    
    def create_cross_department_dashboard(self, 
                                        departments: List[str], 
                                        metric_type: str,
                                        user_context: Dict = None) -> Dict:
        """
        Create a high-level cross-department dashboard
        
        Args:
            departments: List of departments to include
            metric_type: Type of metrics to include (financial, operational, etc.)
            user_context: User context for access control
            
        Returns:
            Dashboard data with cross-department metrics
        """
        logger.debug(f"Creating cross-department dashboard for {len(departments)} departments with metric type: {metric_type}")
        
        dashboard_data = {
            "departments": departments,
            "metric_type": metric_type,
            "creation_date": datetime.now().isoformat(),
            "metrics": {}
        }
        
        try:
            # Get datasets for each department
            department_datasets = {}
            for department in departments:
                try:
                    logger.debug(f"Fetching datasets for department: {department}")
                    dept_datasets = self.warehouse_manager.list_datasets(
                        department=department,
                        user_context=user_context
                    )
                    department_datasets[department] = dept_datasets
                    logger.debug(f"Found {len(dept_datasets)} datasets for department {department}")
                except Exception as e:
                    logger.warning(f"Failed to fetch datasets for department {department}: {str(e)}")
                    department_datasets[department] = []
            
            # Process metrics by type
            if metric_type == "financial":
                logger.debug("Processing financial metrics")
                # Financial metrics across departments
                for department, datasets in department_datasets.items():
                    try:
                        dept_metrics = {"revenue": 0, "cost": 0, "profit": 0}
                        
                        # Find financial datasets
                        financial_datasets = [
                            dataset for dataset in datasets 
                            if "Finance" in dataset.get("module_type", "") or 
                               "finance" in dataset.get("name", "").lower()
                        ]
                        logger.debug(f"Found {len(financial_datasets)} financial datasets for department {department}")
                        
                        for dataset in financial_datasets:
                            try:
                                df = self.warehouse_manager.retrieve(dataset["id"], user_context=user_context)
                                
                                # Check for revenue columns
                                revenue_cols = [col for col in df.columns if "revenue" in col.lower() or "income" in col.lower() or "sales" in col.lower()]
                                if revenue_cols:
                                    dept_metrics["revenue"] += df[revenue_cols[0]].sum()
                                    
                                # Check for cost columns
                                cost_cols = [col for col in df.columns if "cost" in col.lower() or "expense" in col.lower() or "spend" in col.lower()]
                                if cost_cols:
                                    dept_metrics["cost"] += df[cost_cols[0]].sum()
                                    
                                # Calculate profit
                                dept_metrics["profit"] = dept_metrics["revenue"] - dept_metrics["cost"]
                                
                                logger.debug(f"Processed financial metrics for dataset {dataset['id']}")
                            except Exception as e:
                                logger.warning(f"Failed to process dataset {dataset['id']}: {str(e)}")
                                continue
                                
                        dashboard_data["metrics"][department] = dept_metrics
                        logger.info(f"Completed financial metrics for department {department}")
                    except Exception as e:
                        logger.error(f"Failed to process department {department}: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
                        dashboard_data["metrics"][department] = {"error": str(e)}
                    
            elif metric_type == "operational":
                logger.debug("Processing operational metrics")
                # Operational metrics across departments
                for department, datasets in department_datasets.items():
                    try:
                        dept_metrics = {"productivity": 0, "efficiency": 0, "quality": 0}
                        
                        # Find operational datasets
                        operational_datasets = [
                            dataset for dataset in datasets 
                            if "Operations" in dataset.get("module_type", "") or 
                               "Project" in dataset.get("module_type", "")
                        ]
                        logger.debug(f"Found {len(operational_datasets)} operational datasets for department {department}")
                        
                        for dataset in operational_datasets:
                            try:
                                df = self.warehouse_manager.retrieve(dataset["id"], user_context=user_context)
                                
                                # Check for productivity columns
                                productivity_cols = [col for col in df.columns if "productivity" in col.lower() or "output" in col.lower()]
                                if productivity_cols:
                                    dept_metrics["productivity"] += df[productivity_cols[0]].mean()
                                    
                                # Check for efficiency columns
                                efficiency_cols = [col for col in df.columns if "efficiency" in col.lower() or "utilization" in col.lower()]
                                if efficiency_cols:
                                    dept_metrics["efficiency"] += df[efficiency_cols[0]].mean()
                                    
                                # Check for quality columns
                                quality_cols = [col for col in df.columns if "quality" in col.lower() or "defect" in col.lower() or "error" in col.lower()]
                                if quality_cols:
                                    dept_metrics["quality"] += df[quality_cols[0]].mean()
                                    
                                logger.debug(f"Processed operational metrics for dataset {dataset['id']}")
                            except Exception as e:
                                logger.warning(f"Failed to process dataset {dataset['id']}: {str(e)}")
                                continue
                                
                        dashboard_data["metrics"][department] = dept_metrics
                        logger.info(f"Completed operational metrics for department {department}")
                    except Exception as e:
                        logger.error(f"Failed to process department {department}: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
                        dashboard_data["metrics"][department] = {"error": str(e)}
                    
            elif metric_type == "customer":
                logger.debug("Processing customer metrics")
                # Customer metrics across departments
                for department, datasets in department_datasets.items():
                    try:
                        dept_metrics = {"satisfaction": 0, "retention": 0, "acquisition": 0}
                        
                        # Find customer-related datasets
                        customer_datasets = [
                            dataset for dataset in datasets 
                            if "Customer" in dataset.get("module_type", "") or 
                               "CRM" in dataset.get("module_type", "")
                        ]
                        logger.debug(f"Found {len(customer_datasets)} customer datasets for department {department}")
                        
                        for dataset in customer_datasets:
                            try:
                                df = self.warehouse_manager.retrieve(dataset["id"], user_context=user_context)
                                
                                # Check for satisfaction columns
                                satisfaction_cols = [col for col in df.columns if "satisfaction" in col.lower() or "nps" in col.lower() or "rating" in col.lower()]
                                if satisfaction_cols:
                                    dept_metrics["satisfaction"] += df[satisfaction_cols[0]].mean()
                                    
                                # Check for retention columns
                                retention_cols = [col for col in df.columns if "retention" in col.lower() or "churn" in col.lower()]
                                if retention_cols:
                                    dept_metrics["retention"] += df[retention_cols[0]].mean()
                                    
                                # Check for acquisition columns
                                acquisition_cols = [col for col in df.columns if "acquisition" in col.lower() or "new_customer" in col.lower()]
                                if acquisition_cols:
                                    dept_metrics["acquisition"] += df[acquisition_cols[0]].mean()
                                    
                                logger.debug(f"Processed customer metrics for dataset {dataset['id']}")
                            except Exception as e:
                                logger.warning(f"Failed to process dataset {dataset['id']}: {str(e)}")
                                continue
                                
                        dashboard_data["metrics"][department] = dept_metrics
                        logger.info(f"Completed customer metrics for department {department}")
                    except Exception as e:
                        logger.error(f"Failed to process department {department}: {str(e)}", exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
                        dashboard_data["metrics"][department] = {"error": str(e)}
            else:
                error_msg = f"Unsupported metric type: {metric_type}"
                logger.warning(error_msg)
                dashboard_data["error"] = error_msg
                
            return dashboard_data
            
        except Exception as e:
            error_msg = f"Error creating cross-department dashboard: {str(e)}"
            logger.error(error_msg, exc_info=settings.ERROR_INCLUDE_STACK_TRACE)
            dashboard_data["error"] = error_msg
            return dashboard_data
