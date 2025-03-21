import pandas as pd
import sqlalchemy
from typing import Dict, Any, List, Optional
from .base_connector import BaseConnector

class SQLConnector(BaseConnector):
    """Connector for SQL databases"""
    
    def __init__(self):
        self.engine = None
        self.connection = None
        self.dialect_mapping = {
            'postgresql': 'postgresql',
            'mysql': 'mysql+pymysql',
            'mssql': 'mssql+pyodbc',
            'oracle': 'oracle+cx_oracle',
            'sqlite': 'sqlite'
        }
    
    def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to SQL database"""
        try:
            dialect = self.dialect_mapping.get(config['dialect'], 'postgresql')
            conn_str = f"{dialect}://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
            
            if config.get('ssl'):
                self.engine = sqlalchemy.create_engine(
                    conn_str,
                    connect_args={"ssl": config['ssl']}
                )
            else:
                self.engine = sqlalchemy.create_engine(conn_str)
                
            self.connection = self.engine.connect()
            return True
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False
    
    def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data using SQL query"""
        try:
            if 'sql' in query:
                df = pd.read_sql(query['sql'], self.connection)
            elif 'table' in query:
                df = pd.read_sql_table(query['table'], self.connection)
            else:
                raise ValueError("Either 'sql' or 'table' must be provided")
                
            return {'data': df.to_dict('records'), 'schema': list(df.columns)}
        except Exception as e:
            return {'error': str(e)}
    
    def get_schema(self) -> List[Dict[str, Any]]:
        """Get database schema information"""
        if not self.connection:
            return [{'error': 'Not connected'}]
            
        inspector = sqlalchemy.inspect(self.engine)
        tables = []
        
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable']
                })
            
            tables.append({
                'name': table_name,
                'columns': columns
            })
            
        return tables
        
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Push data to database table"""
        try:
            if 'table' not in data or 'data' not in data:
                return False
                
            df = pd.DataFrame(data['data'])
            df.to_sql(
                data['table'], 
                self.connection, 
                if_exists=data.get('if_exists', 'append'),
                index=False
            )
            return True
        except Exception as e:
            print(f"Error pushing data: {str(e)}")
            return False
