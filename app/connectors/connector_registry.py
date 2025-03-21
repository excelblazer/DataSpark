from typing import Dict, Type, Any
from .connectors.base_connector import BaseConnector
from .connectors.sql_connector import SQLConnector
from .connectors.rest_api_connector import RestAPIConnector
from .connectors.file_connector import FileConnector

class ConnectorRegistry:
    """Registry for data connectors"""
    
    def __init__(self):
        self._connectors: Dict[str, Type[BaseConnector]] = {}
        self._instances: Dict[str, Dict[str, BaseConnector]] = {}
        
        # Register built-in connectors
        self.register_connector('sql', SQLConnector)
        self.register_connector('rest_api', RestAPIConnector)
        self.register_connector('file', FileConnector)
    
    def register_connector(self, connector_type: str, connector_class: Type[BaseConnector]) -> None:
        """Register a new connector type"""
        self._connectors[connector_type] = connector_class
        self._instances[connector_type] = {}
    
    def create_connector(self, connector_type: str, connector_id: str, config: Dict[str, Any]) -> BaseConnector:
        """Create and configure a connector instance"""
        if connector_type not in self._connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
            
        connector = self._connectors[connector_type]()
        success = connector.connect(config)
        
        if not success:
            raise ConnectionError(f"Failed to connect using {connector_type} connector")
            
        self._instances[connector_type][connector_id] = connector
        return connector
    
    def get_connector(self, connector_type: str, connector_id: str) -> BaseConnector:
        """Get an existing connector instance"""
        if connector_type not in self._instances or connector_id not in self._instances[connector_type]:
            raise KeyError(f"No connector found for {connector_type}/{connector_id}")
            
        return self._instances[connector_type][connector_id]
