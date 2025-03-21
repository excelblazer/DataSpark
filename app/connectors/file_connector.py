import pandas as pd
import json
import os
from typing import Dict, Any, List, Optional
from .base_connector import BaseConnector

class FileConnector(BaseConnector):
    """Connector for file-based data sources"""
    
    def __init__(self):
        self.file_path = None
        self.file_type = None
        
    def connect(self, config: Dict[str, Any]) -> bool:
        """Set up file connection"""
        try:
            self.file_path = config['file_path']
            if not os.path.exists(self.file_path):
                return False
                
            _, ext = os.path.splitext(self.file_path)
            self.file_type = ext.lower().replace('.', '')
            
            return True
        except Exception as e:
            print(f"File connector error: {str(e)}")
            return False
    
    def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Read data from file"""
        try:
            if self.file_type == 'csv':
                df = pd.read_csv(self.file_path, **query.get('options', {}))
            elif self.file_type in ['xls', 'xlsx']:
                df = pd.read_excel(self.file_path, **query.get('options', {}))
            elif self.file_type == 'json':
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif self.file_type == 'parquet':
                df = pd.read_parquet(self.file_path, **query.get('options', {}))
            else:
                return {'error': f"Unsupported file type: {self.file_type}"}
                
            return {'data': df.to_dict('records'), 'schema': list(df.columns)}
        except Exception as e:
            return {'error': str(e)}
    
    def get_schema(self) -> List[Dict[str, Any]]:
        """Get file schema information"""
        try:
            if self.file_type == 'csv':
                df = pd.read_csv(self.file_path, nrows=0)
            elif self.file_type in ['xls', 'xlsx']:
                df = pd.read_excel(self.file_path, nrows=0)
            elif self.file_type == 'json':
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame([data[0]])
                else:
                    return [{'error': 'Empty or invalid JSON file'}]
            elif self.file_type == 'parquet':
                df = pd.read_parquet(self.file_path, nrows=0)
            else:
                return [{'error': f"Unsupported file type: {self.file_type}"}]
                
            return [{
                'name': os.path.basename(self.file_path),
                'columns': [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]
            }]
        except Exception as e:
            return [{'error': str(e)}]
            
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Write data to file"""
        try:
            if 'data' not in data:
                return False
                
            df = pd.DataFrame(data['data'])
            output_path = data.get('output_path', self.file_path)
            file_type = data.get('file_type', self.file_type)
            
            if file_type == 'csv':
                df.to_csv(output_path, index=False, **data.get('options', {}))
            elif file_type in ['xls', 'xlsx']:
                df.to_excel(output_path, index=False, **data.get('options', {}))
            elif file_type == 'json':
                df.to_json(output_path, orient='records', **data.get('options', {}))
            elif file_type == 'parquet':
                df.to_parquet(output_path, index=False, **data.get('options', {}))
            else:
                return False
                
            return True
        except Exception as e:
            print(f"Error writing data: {str(e)}")
            return False
