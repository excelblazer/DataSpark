import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

class DataProcessor:
    def __init__(self):
        self.supported_visualizations = ['line', 'bar', 'scatter', 'pie', 'heatmap']
        
    def process_raw_data(self, data: Dict) -> pd.DataFrame:
        """Process raw data into analyzable format"""
        try:
            df = pd.DataFrame(data)
            df = self._clean_data(df)
            df = self._transform_data(df)
            return df
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise Exception(f"Error processing data: {str(e)}")
    
    def generate_report(self, report_config: Dict) -> Dict:
        """Generate automated report based on configuration"""
        try:
            results = {
                'summary': self._generate_summary(report_config),
                'visualizations': self._create_visualizations(report_config),
                'insights': self._extract_insights(report_config),
                'recommendations': self._generate_recommendations(report_config)
            }
            return results
        except Exception as e:
            raise Exception(f"Error generating report: {str(e)}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers"""
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        return df
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply necessary transformations to data"""
        # Convert date columns to datetime
        date_columns = df.select_dtypes(include=['object']).columns
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                logging.info(f"Transformed column {col} to datetime.")
            except:
                logging.warning(f"Failed to convert column {col} to datetime.")
                continue
        
        return df
    
    def _generate_summary(self, config: Dict) -> Dict:
        """Generate summary statistics"""
        df = self.process_raw_data(config['data'])
        summary = {
            'record_count': len(df),
            'metrics': df.describe().to_dict(),
            'last_updated': datetime.now().isoformat()
        }
        return summary
    
    def _create_visualizations(self, config: Dict) -> List[Dict]:
        """Create visualizations based on config"""
        df = self.process_raw_data(config['data'])
        visualizations = []
        
        for viz_config in config['visualizations']:
            if viz_config['type'] not in self.supported_visualizations:
                logging.warning(f"Unsupported visualization type: {viz_config['type']}")
                continue
                
            fig = self._create_single_visualization(
                df, 
                viz_config['type'],
                viz_config.get('x'),
                viz_config.get('y'),
                viz_config.get('title')
            )
            visualizations.append({
                'type': viz_config['type'],
                'data': fig.to_json()
            })
            logging.info(f"Created visualization: {viz_config['type']} with title: {viz_config.get('title')}")
        
        return visualizations
    
    def _create_single_visualization(
        self, 
        df: pd.DataFrame,
        viz_type: str,
        x_col: str,
        y_col: str,
        title: str
    ) -> go.Figure:
        """Create a single visualization"""
        if viz_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif viz_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif viz_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif viz_type == 'pie':
            fig = px.pie(df, values=y_col, names=x_col, title=title)
        elif viz_type == 'heatmap':
            fig = px.imshow(df.pivot_table(values=y_col, index=x_col), title=title)
        else:
            logging.error(f"Unsupported visualization type: {viz_type}")
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        logging.info(f"Created visualization of type: {viz_type} with title: {title}")
        return fig
    
    def _extract_insights(self, config: Dict) -> List[str]:
        """Extract key insights from the data"""
        df = self.process_raw_data(config['data'])
        insights = []
        
        # Basic statistical insights
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            insights.append(f"Average {col}: {mean:.2f} (±{std:.2f})")
            logging.info(f"Extracted insight: Average {col}: {mean:.2f} (±{std:.2f})")
        
        # Trend insights
        if len(df) > 1:
            for col in df.select_dtypes(include=[np.number]).columns:
                trend = np.polyfit(range(len(df)), df[col], 1)[0]
                direction = "increasing" if trend > 0 else "decreasing"
                insights.append(f"{col} shows an overall {direction} trend")
                logging.info(f"Extracted trend insight: {col} shows an overall {direction} trend")
        
        return insights
    
    def _generate_recommendations(self, config: Dict) -> List[str]:
        """Generate data-driven recommendations"""
        df = self.process_raw_data(config['data'])
        recommendations = []
        
        # Basic recommendations based on data quality
        missing_pct = df.isnull().mean() * 100
        for col, pct in missing_pct.items():
            if pct > 10:
                recommendations.append(
                    f"Consider improving data collection for {col} "
                    f"({pct:.1f}% missing data)"
                )
                logging.info(f"Recommendation: Improve data collection for {col} ({pct:.1f}% missing data)")
        
        # Performance recommendations
        for col in df.select_dtypes(include=[np.number]).columns:
            recent_avg = df[col].tail(10).mean()
            overall_avg = df[col].mean()
            if recent_avg < overall_avg * 0.8:
                recommendations.append(
                    f"Investigate recent drop in {col} performance"
                )
                logging.info(f"Recommendation: Investigate drop in {col} performance")
        
        return recommendations

    def generate_alerts(self, data, anomalies, thresholds=None):
        """
        Generate alerts based on anomalies and thresholds
        
        Args:
            data: Input data
            anomalies: Anomaly detection results
            thresholds: Dictionary of alert thresholds
            
        Returns:
            List of alerts
        """
        if thresholds is None:
            thresholds = {
                'anomaly_percentage': 5.0,
                'missing_data': 10.0,
                'duplicates': 2.0
            }
        alerts = []
        
        # Check for anomalies
        anomaly_indices = [i for i, val in enumerate(anomalies) if val == -1]
        anomaly_percentage = (len(anomaly_indices) / len(anomalies)) * 100 if anomalies else 0
        
        if anomaly_percentage > thresholds['anomaly_percentage']:
            alerts.append({
                'level': 'warning',
                'type': 'anomaly',
                'message': f'Detected {anomaly_percentage:.1f}% anomalous data points',
                'details': {
                    'indices': anomaly_indices,
                    'threshold': thresholds['anomaly_percentage']
                }
            })
        
        # Check for missing data
        df = pd.DataFrame(data)
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
        
        if missing_percentage > thresholds['missing_data']:
            alerts.append({
                'level': 'error',
                'type': 'missing_data',
                'message': f'Dataset contains {missing_percentage:.1f}% missing values',
                'details': {
                    'missing_count': int(missing_count),
                    'missing_by_column': df.isnull().sum().to_dict(),
                    'threshold': thresholds['missing_data']
                }
            })
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / df.shape[0]) * 100
        
        if duplicate_percentage > thresholds['duplicates']:
            alerts.append({
                'level': 'warning',
                'type': 'duplicates',
                'message': f'Dataset contains {duplicate_percentage:.1f}% duplicate rows',
                'details': {
                    'duplicate_count': int(duplicate_count),
                    'threshold': thresholds['duplicates']
                }
            })
        
        return alerts
