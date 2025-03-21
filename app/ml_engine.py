import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class MLEngine:
    def __init__(self):
        self.scaler = StandardScaler()

    def _detect_seasonality(self, df: pd.DataFrame) -> Dict:
        """Detect seasonal patterns"""
        try:
            decomposition = pd.DataFrame(df).decompose(period=30)
            return {
                'seasonal': decomposition.seasonal.tolist(),
                'period': 30
            }
        except Exception as e:
            return {'seasonal': False, 'error': str(e)}
        
    def analyze_trends(self, data: Dict) -> Dict:
        """Analyze trends in time series data"""
        df = pd.DataFrame(data['values'])
        
        # Perform trend analysis
        results = {
            'trend_direction': self._detect_trend(df),
            'seasonality': self._detect_seasonality(df),
            'outliers': self._detect_outliers(df),
            'patterns': self._identify_patterns(df)
        }
        return results
    
    def generate_forecast(self, data: Dict) -> Dict:
        """Generate forecasts using Prophet"""
        df = pd.DataFrame(data['values'])
        df.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(df)
        
        future = model.make_future_dataframe(periods=data.get('horizon', 30))
        forecast = model.predict(future)
        
        return {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
            'components': model.plot_components(forecast).to_dict()
        }
    
    def _detect_trend(self, df: pd.DataFrame) -> str:
        """Detect overall trend direction"""
        x = np.arange(len(df))
        y = df.values.flatten()
        z = np.polyfit(x, y, 1)
        if z[0] > 0:
            return 'increasing'
        elif z[0] < 0:
            return 'decreasing'
        return 'stable'
    
    def _detect_seasonality(self, df: pd.DataFrame) -> Dict:
        """Detect seasonal patterns"""
        try:
            decomposition = pd.DataFrame(df).decompose(period=30)
            return {
                'seasonal': decomposition.seasonal.tolist(),
                'period': 30
            }
        except:
            return {'seasonal': False}
    
    def _detect_outliers(self, df: pd.DataFrame) -> List:
        """Detect outliers using IQR method"""
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))]
        return outliers.index.tolist()
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify common patterns in the data"""
        patterns = {
            'stationary': self._check_stationarity(df),
            'clusters': self._identify_clusters(df)
        }
        return patterns
    
    def _check_stationarity(self, df: pd.DataFrame) -> bool:
        """Check if time series is stationary"""
        result = adfuller(df.values.flatten())
        return result[1] < 0.05
    
    def _identify_clusters(self, df: pd.DataFrame) -> Dict:
        """Identify clusters in the data"""
        if len(df) < 3:
            return {'clusters': 0}
            
        scaled_data = self.scaler.fit_transform(df)
        kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
    
        return {
            'clusters': len(set(clusters)),
            'labels': clusters.tolist()
        }
    def detect_anomalies(self, data: Dict) -> List:
        """Detect anomalies using Isolation Forest"""
        df = pd.DataFrame(data['values'])
        isolation_forest = IsolationForest(contamination=0.05)
        isolation_forest.fit(df)
        anomalies = isolation_forest.predict(df)
        return anomalies.tolist()
    
    def forecast_with_lstm(self, data: Dict) -> Dict:
        """Generate forecasts using LSTM"""
        df = pd.DataFrame(data['values'])
        df.columns = ['ds', 'y']
        
        # Prepare the data for LSTM
        df['y'] = self.scaler.fit_transform(df[['y']])
        X = []
        y = []
        n_future = data.get('horizon', 30)
        n_past = 14  # Number of past observations
        
        for i in range(n_past, len(df) - n_future):
            X.append(df['y'][i - n_past:i])
            y.append(df['y'][i:i + n_future])
        
        X = np.array(X)
        y = np.array(y)
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(n_past, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(n_future))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
        
        # Make predictions
        forecast = model.predict(X[-1].reshape(1, n_past, 1))
        forecast = self.scaler.inverse_transform(forecast)
        return {
            'forecast': forecast.tolist()
        }
        # Add this to app/ml_engine.py

def create_budget_forecast(self, data: Dict, config: Dict) -> Dict:
    """
    Generate budget forecasts based on operational and financial data
    
    Args:
        data: Historical data for forecasting
        config: Configuration with forecasting parameters
        
    Returns:
        Dict containing budget forecasts
    """
    # Extract configuration parameters
    time_periods = config.get('periods', 12)
    target_variables = config.get('target_variables', [])
    growth_assumptions = config.get('growth_assumptions', {})
    seasonality = config.get('seasonality', True)
    include_historical = config.get('include_historical', True)
    
    # Convert input data to DataFrame
    df = pd.DataFrame(data['values'])
    
    # Identify date column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)
        except:
            pass
    
    # Process each target variable
    forecast_results = {}
    
    for variable in target_variables:
        if variable not in df.columns:
            continue
            
        # Apply specific growth assumption if provided
        growth_rate = growth_assumptions.get(variable, 0.05)  # Default 5% growth
        
        # Call appropriate forecasting method based on data
        if len(df) >= 12 and date_cols:  # Enough data for Prophet
            try:
                forecast = self._forecast_with_prophet(
                    df, 
                    date_col, 
                    variable, 
                    periods=time_periods, 
                    seasonality=seasonality,
                    growth_prior=growth_rate
                )
                forecast_results[variable] = forecast
            except Exception as e:
                # Fallback to simpler method
                forecast = self._forecast_with_trend(
                    df, 
                    date_col, 
                    variable, 
                    periods=time_periods, 
                    growth_rate=growth_rate
                )
                forecast_results[variable] = forecast
        else:
            # Use simpler trend-based forecast for limited data
            forecast = self._forecast_with_trend(
                df, 
                date_col, 
                variable, 
                periods=time_periods, 
                growth_rate=growth_rate
            )
            forecast_results[variable] = forecast
    
    # Combine forecasts into a comprehensive budget
    budget_forecast = self._create_consolidated_budget(forecast_results, include_historical)
    
    return {
        'budget_forecast': budget_forecast,
        'forecast_period': f"{time_periods} periods",
        'generated_at': datetime.now().isoformat(),
        'assumptions': {
            'growth_rates': growth_assumptions,
            'seasonality': seasonality
        }
    }

def _forecast_with_prophet(self, df, date_col, target_col, periods=12, seasonality=True, growth_prior=0.05):
    """Forecast using Prophet with specified parameters"""
    # Prepare data for Prophet
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    
    # Configure Prophet model
    model = Prophet(
        yearly_seasonality=seasonality,
        weekly_seasonality=seasonality,
        daily_seasonality=False,
        growth='linear'
    )
    
    # Set growth prior if specified
    if growth_prior:
        model.growth.prior_scale = growth_prior
    
    # Fit model and generate forecast
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    # Extract relevant columns and rename
    result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    result_df = result_df.rename(columns={
        'ds': date_col, 
        'yhat': f'{target_col}_forecast',
        'yhat_lower': f'{target_col}_lower',
        'yhat_upper': f'{target_col}_upper'
    })
    
    return result_df

def _forecast_with_trend(self, df, date_col, target_col, periods=12, growth_rate=0.05):
    """Generate simple trend-based forecast"""
    if date_col in df.columns and target_col in df.columns:
        # Get the last value
        last_value = df[target_col].iloc[-1] if not df.empty else 0
        last_date = df[date_col].iloc[-1] if not df.empty else datetime.now()
        
        # Generate future dates
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
        
        # Generate forecast values with growth
        forecast_values = [last_value * (1 + growth_rate) ** (i+1) for i in range(periods)]
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            date_col: future_dates,
            f'{target_col}_forecast': forecast_values,
            f'{target_col}_lower': [v * 0.9 for v in forecast_values],
            f'{target_col}_upper': [v * 1.1 for v in forecast_values]
        })
        
        return result_df
    else:
        # Handle missing columns
        return pd.DataFrame()

def _create_consolidated_budget(self, forecast_results, include_historical=True):
    """Combine individual forecasts into a comprehensive budget"""
    # Combine all forecasts based on date
    combined_df = None
    date_col = None
    
    for variable, forecast_df in forecast_results.items():
        # Identify date column
        date_cols = [col for col in forecast_df.columns if pd.api.types.is_datetime64_any_dtype(forecast_df[col])]
        if date_cols:
            date_col = date_cols[0]
            
            if combined_df is None:
                combined_df = forecast_df[[date_col]].copy()
                
            # Add variable columns
            for col in forecast_df.columns:
                if col != date_col:
                    combined_df[col] = forecast_df[col]
        
    if combined_df is None:
        return pd.DataFrame()
        
    # Calculate derived metrics if possible
    if 'revenue_forecast' in combined_df.columns and 'cost_forecast' in combined_df.columns:
        combined_df['profit_forecast'] = combined_df['revenue_forecast'] - combined_df['cost_forecast']
        
        if 'revenue_forecast' in combined_df.columns and combined_df['revenue_forecast'].sum() > 0:
            combined_df['margin_forecast'] = (combined_df['profit_forecast'] / combined_df['revenue_forecast'] * 100).round(2)
    
    return combined_df
