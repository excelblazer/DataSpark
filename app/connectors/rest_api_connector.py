import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from .base_connector import BaseConnector

class RestAPIConnector(BaseConnector):
    """Connector for REST APIs"""
    
    def __init__(self):
        self.base_url = None
        self.headers = None
        self.auth = None
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Configure connection to REST API"""
        try:
            self.base_url = config['base_url']
            self.headers = config.get('headers', {})
            
            if 'auth' in config:
                if config['auth']['type'] == 'basic':
                    self.auth = (config['auth']['username'], config['auth']['password'])
                elif config['auth']['type'] == 'token':
                    self.headers['Authorization'] = f"Bearer {config['auth']['token']}"
                    
            # Test connection
            response = requests.get(
                f"{self.base_url.rstrip('/')}/status", 
                headers=self.headers,
                auth=self.auth
            )
            return response.status_code < 400
        except Exception as e:
            print(f"API connection error: {str(e)}")
            return False
    
    def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from REST API"""
        try:
            endpoint = query.get('endpoint', '')
            method = query.get('method', 'GET')
            params = query.get('params', {})
            body = query.get('body', {})
            
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=self.headers, auth=self.auth)
            elif method.upper() == 'POST':
                response = requests.post(url, params=params, json=body, headers=self.headers, auth=self.auth)
            else:
                return {'error': f"Method {method} not supported"}
                
            if response.status_code >= 400:
                return {'error': f"API Error: {response.status_code}"}
                
            json_data = response.json()
            
            # Extract data based on path if provided
            if 'data_path' in query:
                for key in query['data_path'].split('.'):
                    json_data = json_data.get(key, {})
                    
            # Convert to dataframe if needed
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
                return {'data': df.to_dict('records'), 'schema': list(df.columns)}
            
            return {'data': json_data}
        except Exception as e:
            return {'error': str(e)}
    
    def get_schema(self) -> List[Dict[str, Any]]:
        """Get API schema information (if available)"""
        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}/schema", 
                headers=self.headers,
                auth=self.auth
            )
            if response.status_code < 400:
                return response.json()
            return [{'error': f"Could not fetch schema: {response.status_code}"}]
        except Exception as e:
            return [{'error': str(e)}]
            
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Push data to API endpoint"""
        try:
            if 'endpoint' not in data or 'data' not in data:
                return False
                
            url = f"{self.base_url.rstrip('/')}/{data['endpoint'].lstrip('/')}"
            response = requests.post(
                url,
                json=data['data'],
                headers=self.headers,
                auth=self.auth
            )
            
            return response.status_code < 400
        except Exception as e:
            print(f"Error pushing data: {str(e)}")
            return False
