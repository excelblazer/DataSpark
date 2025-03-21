import json
import yaml
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from functools import lru_cache
from .config import settings

logger = logging.getLogger("databloom.data_mapper")

class DataMapperError(Exception):
    """Base class for data mapper exceptions"""
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = f"[{datetime.now().isoformat()}] MapperError-{error_type}: {message}"
        super().__init__(self.message)

class DataMapper:
    """
    Centralized utility for data type mappings and metadata.
    Used by both AnalyticsRecommender and DataClassificationService.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the DataMapper with mappings from config file or defaults"""
        logger.debug("Initializing DataMapper")
        # Initialize with default mappings
        self.mappings = self._init_default_mappings()
        
        # Override with config file if provided
        if config_path:
            if not os.path.exists(config_path):
                logger.warning(f"Config path not found: {config_path}")
            else:
                self._load_config(config_path)
                logger.info(f"Loaded configuration from {config_path}")
    
    def _init_default_mappings(self) -> Dict:
        """Initialize default mappings"""
        return {
            # Data type to semantic type mappings
            "column_semantic_mapping": {
                'id': 'identifier',
                'date': 'datetime',
                'time': 'datetime',
                'year': 'datetime',
                'month': 'datetime',
                'day': 'datetime',
                'price': 'numerical',
                'cost': 'numerical',
                'revenue': 'numerical',
                'sales': 'numerical',
                'profit': 'numerical',
                'income': 'numerical',
                'expense': 'numerical',
                'quantity': 'numerical',
                'count': 'numerical',
                'age': 'numerical',
                'rating': 'numerical',
                'score': 'numerical',
                'latitude': 'geospatial',
                'longitude': 'geospatial',
                'country': 'geospatial',
                'city': 'geospatial',
                'state': 'geospatial',
                'zip': 'geospatial',
                'postal': 'geospatial',
                'name': 'categorical',
                'category': 'categorical',
                'type': 'categorical',
                'status': 'categorical',
                'gender': 'categorical',
                'segment': 'categorical',
                'comment': 'text',
                'review': 'text',
                'feedback': 'text',
                'description': 'text',
                'email': 'categorical',
                'phone': 'categorical'
            },
            
            # Semantic type to recommended analytics mappings
            "analytics_mapping": {
                'numerical': [
                    'average', 'sum', 'median', 'min', 'max', 'variance',
                    'standard_deviation', 'correlation', 'regression',
                    'time_series', 'forecasting', 'anomaly_detection'
                ],
                'categorical': [
                    'frequency', 'distribution', 'cross_tabulation',
                    'chi_square', 'sentiment_analysis', 'clustering'
                ],
                'datetime': [
                    'trend_analysis', 'seasonality', 'period_comparison',
                    'cohort_analysis', 'retention_analysis', 'funnel_analysis'
                ],
                'geospatial': [
                    'geo_clustering', 'proximity_analysis', 'heat_map',
                    'route_optimization'
                ],
                'text': [
                    'sentiment_analysis', 'keyword_extraction', 'topic_modeling',
                    'text_classification', 'entity_recognition'
                ]
            },
            
            # Data types to visualization mappings
            "visualization_mapping": {
                'numerical_vs_numerical': ['scatter', 'line', 'bubble'],
                'numerical_vs_categorical': ['bar', 'box', 'violin'],
                'categorical_vs_categorical': ['heatmap', 'mosaic', 'bar'],
                'numerical_distribution': ['histogram', 'density', 'box'],
                'categorical_distribution': ['pie', 'bar', 'donut'],
                'time_series': ['line', 'area', 'candlestick'],
                'geospatial': ['map', 'choropleth', 'scatter_geo']
            },
            
            # Department and module classification patterns
            # (to be populated from classification_service.py)
            "module_patterns": {},
            
            # Module schemas
            "module_schemas": {},
            "department_module_schemas": {}
        }
    
    def _load_config(self, config_path: str) -> None:
        """
        Load mappings from configuration file
        
        Args:
            config_path: Path to YAML or JSON config file
        """
        try:
            if config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
            
            # Update mappings with loaded config
            for key, value in config.items():
                if key in self.mappings:
                    self.mappings[key].update(value)
                else:
                    self.mappings[key] = value
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    
    @lru_cache(maxsize=1000)
    def get_semantic_type(self, column_name: str) -> str:
        """Get semantic type based on column name"""
        logger.debug(f"Getting semantic type for column: {column_name}")
        column_lower = column_name.lower()
        
        for key, semantic_type in self.mappings["column_semantic_mapping"].items():
            if key in column_lower:
                logger.debug(f"Found semantic type {semantic_type} for column {column_name}")
                return semantic_type
        
        logger.debug(f"No semantic type found for column {column_name}")
        return "unknown"
    
    @lru_cache(maxsize=100)
    def get_recommended_analytics(self, semantic_type: str) -> List[str]:
        """Get recommended analytics for a semantic type"""
        logger.debug(f"Getting analytics recommendations for type: {semantic_type}")
        recommendations = self.mappings["analytics_mapping"].get(semantic_type, [])
        logger.debug(f"Found {len(recommendations)} recommendations for {semantic_type}")
        return recommendations
    
    def get_visualization_types(self, relationship_type: str) -> List[str]:
        """
        Get recommended visualization types for a relationship type
        
        Args:
            relationship_type: Type of relationship between columns
            
        Returns:
            List of recommended visualization types
        """
        return self.mappings["visualization_mapping"].get(relationship_type, [])
    
    def get_module_patterns(self, category: Optional[str] = None) -> Dict:
        """
        Get module patterns for classification
        
        Args:
            category: Optional category to filter patterns
            
        Returns:
            Dictionary of module patterns
        """
        patterns = self.mappings.get("module_patterns", {})
        if category and category in patterns:
            return {category: patterns[category]}
        return patterns
    
    def save_config(self, config_path: str) -> None:
        """
        Save current mappings to a config file
        
        Args:
            config_path: Path to save the configuration
        """
        try:
            # Determine file format based on extension
            if config_path.endswith('.json'):
                with open(config_path, 'w') as f:
                    json.dump(self.mappings, f, indent=2)
            elif config_path.endswith(('.yml', '.yaml')):
                with open(config_path, 'w') as f:
                    yaml.dump(self.mappings, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
            
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {str(e)}")
            raise DataMapperError("CONFIG_SAVE_ERROR", str(e))
    
    def update_mapping(self, mapping_type: str, key: str, value: Any) -> None:
        """
        Update a specific mapping
        
        Args:
            mapping_type: Type of mapping to update
            key: Key to update
            value: New value
        """
        if mapping_type not in self.mappings:
            raise DataMapperError("INVALID_MAPPING_TYPE", f"Invalid mapping type: {mapping_type}")
        
        self.mappings[mapping_type][key] = value
        logger.info(f"Updated mapping {mapping_type}.{key}")
        
        # Clear cache when mappings are updated
        self.get_semantic_type.cache_clear()
        self.get_recommended_analytics.cache_clear()
    
    def map_to_module(self, dataset: Dict, module_type: str) -> Dict:
        """
        Map a dataset to a specific module schema
        
        Args:
            dataset: The dataset to map
            module_type: Target module type to map to
            
        Returns:
            Mapped dataset
        """
        try:
            logger.info(f"Mapping dataset to module: {module_type}")
            
            # Get module schema
            module_schema = self._get_module_schema(module_type)
            if not module_schema:
                logger.warning(f"No schema found for module type: {module_type}")
                # Return original dataset if no schema found
                return dataset
            
            # Extract data from dataset
            if isinstance(dataset, dict) and 'data' in dataset:
                data = dataset['data']
                metadata = dataset.get('metadata', {})
            else:
                data = dataset
                metadata = {}
            
            # Apply mapping and transformations
            mapped_data = self._apply_mapping(data, module_schema)
            
            # Update metadata
            mapped_metadata = {
                **metadata,
                'module_mapping': {
                    'module_type': module_type,
                    'mapped_at': datetime.now().isoformat(),
                    'mapping_version': module_schema.get('version', '1.0'),
                }
            }
            
            # Return mapped dataset
            return {
                'data': mapped_data,
                'metadata': mapped_metadata
            }
            
        except Exception as e:
            logger.error(f"Error mapping to module {module_type}: {str(e)}")
            raise DataMapperError("MODULE_MAPPING_ERROR", f"Error mapping to module {module_type}: {str(e)}")
    
    def _get_module_schema(self, module_type: str) -> Dict:
        """Get schema for a specific module type"""
        # Look for module schema in mappings
        module_schemas = self.mappings.get("module_schemas", {})
        
        # Try to get exact match first
        if module_type in module_schemas:
            return module_schemas[module_type]
        
        # Try case-insensitive match
        for schema_type, schema in module_schemas.items():
            if schema_type.lower() == module_type.lower():
                return schema
        
        # If no match in mappings, check if we have department-module-specific schema
        dept_module_schemas = self.mappings.get("department_module_schemas", {})
        for dept_module, schema in dept_module_schemas.items():
            if dept_module.lower().endswith(module_type.lower()):
                return schema
        
        # If not found, return empty schema
        logger.warning(f"No schema found for module type: {module_type}")
        return {}
    
    def _apply_mapping(self, data: List[Dict], module_schema: Dict) -> List[Dict]:
        """Apply mapping schema to data"""
        if not data or not module_schema:
            return data
        
        # Get field mappings
        field_mappings = module_schema.get('field_mappings', {})
        transformations = module_schema.get('transformations', {})
        
        # Apply mappings to each item
        mapped_data = []
        for item in data:
            mapped_item = self._map_item(item, field_mappings, transformations)
            mapped_data.append(mapped_item)
        
        return mapped_data
    
    def _map_item(self, item: Dict, field_mappings: Dict, transformations: Dict) -> Dict:
        """Map a single data item according to schema"""
        mapped_item = {}
        
        # Apply direct field mappings
        for target_field, source_field in field_mappings.items():
            if isinstance(source_field, str) and source_field in item:
                mapped_item[target_field] = item[source_field]
            elif isinstance(source_field, list):
                # Try each source field in order
                for src in source_field:
                    if src in item:
                        mapped_item[target_field] = item[src]
                        break
        
        # Apply transformations
        for field, transform in transformations.items():
            if field in mapped_item and transform == 'lowercase':
                if isinstance(mapped_item[field], str):
                    mapped_item[field] = mapped_item[field].lower()
            elif field in mapped_item and transform == 'uppercase':
                if isinstance(mapped_item[field], str):
                    mapped_item[field] = mapped_item[field].upper()
        
        return mapped_item
    
    def get_department_modules(self, department: str) -> List[str]:
        """
        Get available module types for a department
        
        Args:
            department: Department name
            
        Returns:
            List of module types available for the department
        """
        # Get module patterns by department
        patterns = self.mappings.get("module_patterns", {})
        dept_modules = patterns.get(department, {})
        
        # Get department-module schemas
        schema_keys = self.mappings.get("department_module_schemas", {}).keys()
        modules = []
        
        # Check patterns
        if dept_modules:
            modules.extend(dept_modules.keys())
        
        # Check schemas
        for key in schema_keys:
            if key.startswith(f"{department}_"):
                module = key.split("_", 1)[1]
                if module not in modules:
                    modules.append(module)
        
        return modules
