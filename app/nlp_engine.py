import spacy
import re
from typing import Dict, List, Optional
import pandas as pd

class NaturalLanguageProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.time_patterns = {
            "last quarter": {"period": "quarter", "offset": -1},
            "this quarter": {"period": "quarter", "offset": 0},
            "last month": {"period": "month", "offset": -1},
            "ytd": {"period": "ytd", "offset": 0},
            "last year": {"period": "year", "offset": -1}
        }
        self.metric_keywords = {
            "revenue": "revenue",
            "sales": "revenue",
            "profit": "profit",
            "margin": "margin",
            "cost": "cost",
            "expenses": "expenses",
            "users": "users",
            "customers": "customers"
        }
        self.viz_keywords = {
            "trend": "line",
            "mix": "pie",
            "breakdown": "bar",
            "comparison": "bar",
            "distribution": "histogram",
            "correlation": "scatter",
            "map": "geo"
        }
        
    def interpret(self, query: str) -> Dict:
        """
        Convert natural language query to structured analytics request
        """
        query = query.lower()
        doc = self.nlp(query)
        
        # Extract time period
        time_period = self._extract_time_period(query)
        
        # Extract metrics
        metrics = self._extract_metrics(doc)
        
        # Extract visualization type
        viz_type = self._extract_visualization_type(query)
        
        # Extract grouping dimensions
        dimensions = self._extract_dimensions(doc)
        
        # Extract filters
        filters = self._extract_filters(doc)
        
        # Determine if comparison is needed
        comparison = "compare" in query or "vs" in query or "versus" in query
        
        # Build the query plan
        query_plan = {
            "time_period": time_period,
            "metrics": metrics,
            "dimensions": dimensions,
            "filters": filters,
            "visualization": viz_type,
            "comparison": comparison,
            "original_query": query
        }
        
        return query_plan
    
    def _extract_time_period(self, query: str) -> Dict:
        """Extract time period information from query"""
        for pattern, info in self.time_patterns.items():
            if pattern in query:
                return info
        # Default to last month if no time period specified
        return {"period": "month", "offset": -1}
    
    def _extract_metrics(self, doc) -> List[str]:
        """Extract metrics from the query"""
        metrics = []
        for token in doc:
            if token.text in self.metric_keywords:
                metrics.append(self.metric_keywords[token.text])
        
        # If no metrics found, default to revenue
        return metrics if metrics else ["revenue"]
    
    def _extract_visualization_type(self, query: str) -> str:
        """Determine visualization type based on query"""
        for keyword, viz in self.viz_keywords.items():
            if keyword in query:
                return viz
        # Default to line chart if no visualization keyword found
        return "line"
    
    def _extract_dimensions(self, doc) -> List[str]:
        """Extract dimensions to group by"""
        dimensions = []
        for token in doc:
            if token.pos_ == "NOUN" and token.text not in self.metric_keywords:
                if token.text in ["product", "region", "customer", "category", "segment", "channel"]:
                    dimensions.append(token.text)
        
        return dimensions
    
    def _extract_filters(self, doc) -> Dict[str, str]:
        """Extract filters from query"""
        filters = {}
        # Simple pattern matching for filters
        # e.g., "for product X" or "in region Y"
        for token in doc:
            if token.text in ["for", "in", "by", "with"]:
                if token.i + 2 < len(doc) and doc[token.i + 1].pos_ == "NOUN":
                    filter_key = doc[token.i + 1].text
                    filter_value = doc[token.i + 2].text
                    filters[filter_key] = filter_value
        
        return filters
    
    def generate_sql(self, interpretation: Dict, tables: Dict) -> str:
        """
        Generate SQL query from interpreted request
        """
        select_clause = []
        for metric in interpretation["metrics"]:
            select_clause.append(f"SUM({metric}) as {metric}")
        
        # Add dimensions to select clause
        for dim in interpretation["dimensions"]:
            select_clause.append(dim)
        
        # Determine appropriate table
        table_name = self._determine_table(interpretation, tables)
        
        # Build WHERE clause for time period
        time_clause = self._build_time_clause(interpretation["time_period"])
        
        # Build WHERE clause for filters
        filter_clauses = []
        for key, value in interpretation["filters"].items():
            filter_clauses.append(f"{key} = '{value}'")
        
        where_clause = " AND ".join([time_clause] + filter_clauses) if filter_clauses else time_clause
        
        # Build GROUP BY clause
        group_by = ""
        if interpretation["dimensions"]:
            group_by = f"GROUP BY {', '.join(interpretation['dimensions'])}"
        
        # Assemble the full query
        sql = f"""
        SELECT {', '.join(select_clause)}
        FROM {table_name}
        WHERE {where_clause}
        {group_by}
        """
        
        return sql
    
    def _determine_table(self, interpretation: Dict, tables: Dict) -> str:
        """Determine which table to use based on the metrics and dimensions"""
        # This is a simplified implementation
        # In reality, you would match metrics and dimensions to table schemas
        for metric in interpretation["metrics"]:
            for table_name, table_info in tables.items():
                if metric in table_info["columns"]:
                    return table_name
        
        # Default to a general table if no match found
        return "fact_sales"
    
    def _build_time_clause(self, time_period: Dict) -> str:
        """Build time filter SQL based on time period"""
        period = time_period["period"]
        offset = time_period["offset"]
        
        if period == "quarter":
            if offset == -1:
                return "date_trunc('quarter', date) = date_trunc('quarter', current_date) - interval '3 months'"
            else:
                return "date_trunc('quarter', date) = date_trunc('quarter', current_date)"
        elif period == "month":
            if offset == -1:
                return "date_trunc('month', date) = date_trunc('month', current_date) - interval '1 month'"
            else:
                return "date_trunc('month', date) = date_trunc('month', current_date)"
        elif period == "year":
            if offset == -1:
                return "date_trunc('year', date) = date_trunc('year', current_date) - interval '1 year'"
            else:
                return "date_trunc('year', date) = date_trunc('year', current_date)"
        elif period == "ytd":
            return "date_part('year', date) = date_part('year', current_date) AND date <= current_date"
        
        # Default to last 30 days
        return "date >= current_date - interval '30 days'"
