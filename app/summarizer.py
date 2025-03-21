import pandas as pd
import numpy as np
from typing import Dict, List, Union
from datetime import datetime

class Summarizer:
    def __init__(self):
        self.summary_types = ["executive", "financial", "operational", "trend"]
    
    def summarize(self, data: Union[Dict, pd.DataFrame], summary_type: str = "executive") -> Dict:
        """
        Generate actionable summary based on data
        
        Args:
            data: Data to summarize (dict or DataFrame)
            summary_type: Type of summary to generate
            
        Returns:
            Dictionary containing summary information
        """
        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
            
        if summary_type == "executive":
            return self._executive_summary(df)
        elif summary_type == "financial":
            return self._financial_summary(df)
        elif summary_type == "operational":
            return self._operational_summary(df)
        elif summary_type == "trend":
            return self._trend_summary(df)
        else:
            return self._executive_summary(df)
    
    def _executive_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive executive summary with recommendations"""
        # Identify numeric and date columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # Calculate key metrics
        metrics = {}
        insights = []
        recommendations = []
        
        # Calculate basic statistics
        for col in numeric_cols:
            metrics[f"{col}_avg"] = df[col].mean()
            metrics[f"{col}_min"] = df[col].min()
            metrics[f"{col}_max"] = df[col].max()
            
            # Add insights based on data patterns
            if df[col].std() / df[col].mean() > 0.5:
                insights.append(f"High variability detected in {col} (CV: {df[col].std() / df[col].mean():.2f})")
            
            # Check for trends if date column is available
            if date_cols and len(date_cols) > 0:
                primary_date = date_cols[0]
                sorted_df = df.sort_values(by=primary_date)
                first_val = sorted_df[col].iloc[0] if not sorted_df.empty else 0
                last_val = sorted_df[col].iloc[-1] if not sorted_df.empty else 0
                
                if last_val > first_val * 1.1:
                    insights.append(f"{col} has increased by {(last_val - first_val) / first_val * 100:.1f}% over the period")
                elif first_val > last_val * 1.1:
                    insights.append(f"{col} has decreased by {(first_val - last_val) / first_val * 100:.1f}% over the period")
                    recommendations.append(f"Investigate the decline in {col} values")
        
        # Data quality recommendations
        missing_pct = df.isnull().mean() * 100
        for col, pct in missing_pct.items():
            if pct > 0:
                insights.append(f"{col} has {pct:.1f}% missing values")
                if pct > 10:
                    recommendations.append(f"Address data quality issues with {col} ({pct:.1f}% missing data)")
        
        # General recommendations
        if len(numeric_cols) > 0 and len(date_cols) > 0:
            recommendations.append("Consider time series forecasting for future projections")
        
        if len(numeric_cols) >= 2:
            recommendations.append("Analyze correlations between key metrics for additional insights")
        
        return {
            "summary_type": "executive",
            "data_points": len(df),
            "time_period": f"{df[date_cols[0]].min()} to {df[date_cols[0]].max()}" if date_cols else "N/A",
            "key_metrics": metrics,
            "insights": insights,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }
    
    def _financial_summary(self, df: pd.DataFrame) -> Dict:
        """Generate financial-focused summary"""
        # Implementation for financial summary
        # Look for financial columns like revenue, cost, profit, etc.
        financial_indicators = {}
        financial_columns = [col for col in df.columns if col.lower() in [
            'revenue', 'sales', 'cost', 'expense', 'profit', 'margin', 'income', 
            'budget', 'actual', 'forecast', 'cash', 'assets', 'liabilities'
        ]]
        
        # Calculate financial metrics and KPIs
        # Implementation would go here
        
        return {
            "summary_type": "financial",
            "financial_indicators": financial_indicators,
            "insights": [],
            "recommendations": []
        }
    
    def _operational_summary(self, df: pd.DataFrame) -> Dict:
        """Generate operations-focused summary"""
        # Implementation for operational summary
        return {
            "summary_type": "operational",
            "operational_indicators": {},
            "insights": [],
            "recommendations": []
        }
    
    def _trend_summary(self, df: pd.DataFrame) -> Dict:
        """Generate trend analysis summary"""
        # Implementation for trend summary
        return {
            "summary_type": "trend",
            "trend_indicators": {},
            "insights": [],
            "recommendations": []
        }
