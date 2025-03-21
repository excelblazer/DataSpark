# app/connectors/base_connector.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseConnector(ABC):
    """Base class for all data connectors"""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from the source"""
        pass
    
    @abstractmethod
    def get_schema(self) -> List[Dict[str, Any]]:
        """Get schema information from the data source"""
        pass
        
    @abstractmethod
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Push data to the target system"""
        pass
