import pandas as pd
import numpy as np
from typing import Dict, List, Union
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

class ScenarioSimulator:
    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine
        self.scenario_types = ["optimistic", "pessimistic", "baseline", "custom"]
    
    def simulate(self, 
                data: Union[Dict, pd.DataFrame], 
                target_variables: List[str], 
                scenario_type: str = "baseline",
                time_periods: int = 12,
                custom_adjustments: Dict = None) -> Dict:
        """
        Simulate scenarios based on historical data
        
        Args:
            data: Historical data as DataFrame or dict
            target_variables: List of variables to forecast
            scenario_type: Type of scenario (optimistic, pessimistic, baseline, custom)
            time_periods: Number of periods to forecast
            custom_adjustments: Custom percentage adjustments for custom scenario
            
        Returns:
            Dictionary containing scenario simulation results
        """
        # Convert to DataFrame if needed
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # Ensure date column is identified and converted to datetime
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if not date_cols:
            # Try to identify and convert a date column
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols = [col]
                        break
                    except:
                        continue
        
        if not date_cols:
            raise ValueError("No date column found for time-based forecasting")
            
        date_col = date_cols[0]
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        # Create simulation results for each target variable
        simulation_results = {}
        
        for variable in target_variables:
            if variable not in df.columns:
                continue
                
            # Determine scenario adjustments
            adjustments = self._get_scenario_adjustments(
                scenario_type, 
                custom_adjustments.get(variable, 0) if custom_adjustments else 0
            )
            
            # Generate baseline forecast
            baseline_forecast = self._generate_forecast(df, date_col, variable, time_periods)
            
            # Apply adjustments based on scenario
            simulation_results[variable] = {
                "baseline": baseline_forecast,
                "scenario_forecast": self._apply_scenario_adjustments(baseline_forecast, adjustments)
            }
            
        return {
            "scenario_type": scenario_type,
            "simulation_date": datetime.now().isoformat(),
            "time_periods": time_periods,
            "results": simulation_results
        }
    
    def _generate_forecast(self, df: pd.DataFrame, date_col: str, variable: str, periods: int) -> pd.DataFrame:
        """Generate baseline forecast using time series methods"""
        # Check if we have enough data for forecasting
        if len(df) < 3:
            raise ValueError(f"Not enough historical data for forecasting {variable}")
            
        # Extract time series
        time_series = df[[date_col, variable]].set_index(date_col)
        
        # Handle missing values
        time_series = time_series.fillna(method='ffill').fillna(method='bfill')
        
        # Determine frequency
        freq = pd.infer_freq(time_series.index)
        if freq is None:
            # Default to monthly if can't be determined
            freq = 'M'
            
        # Try to use ARIMA for forecasting
        try:
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Generate future dates
            last_date = time_series.index[-1]
            if freq == 'M':
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
            elif freq == 'D':
                future_dates = [last_date + pd.DateOffset(days=i+1) for i in range(periods)]
            elif freq == 'Q':
                future_dates = [last_date + pd.DateOffset(months=(i+1)*3) for i in range(periods)]
            elif freq == 'Y' or freq == 'A':
                future_dates = [last_date + pd.DateOffset(years=i+1) for i in range(periods)]
            else:
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
                
            # Generate forecast
            forecast = model_fit.forecast(steps=periods)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                date_col: future_dates,
                variable: forecast.values,
                'lower_ci': forecast.values * 0.9,  # Simplified CI
                'upper_ci': forecast.values * 1.1   # Simplified CI
            })
            
            return forecast_df
            
        except Exception as e:
            # Fallback to simple moving average if ARIMA fails
            ma_periods = min(3, len(time_series))
            avg_value = time_series[variable].rolling(window=ma_periods).mean().iloc[-1]
            
            # Generate future dates
            last_date = time_series.index[-1]
            if freq == 'M':
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
            elif freq == 'D':
                future_dates = [last_date + pd.DateOffset(days=i+1) for i in range(periods)]
            else:
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
                
            # Create forecast DataFrame with constant value
            forecast_df = pd.DataFrame({
                date_col: future_dates,
                variable: [avg_value] * periods,
                'lower_ci': [avg_value * 0.9] * periods,  # Simplified CI
                'upper_ci': [avg_value * 1.1] * periods   # Simplified CI
            })
            
            return forecast_df
    
    def _get_scenario_adjustments(self, scenario_type: str, custom_adjustment: float = 0) -> Dict:
        """Get adjustment parameters based on scenario type"""
        if scenario_type == "optimistic":
            return {
                "growth_factor": 1.2,  # 20% more growth than baseline
                "volatility": 0.8      # 20% less volatility than baseline
            }
        elif scenario_type == "pessimistic":
            return {
                "growth_factor": 0.8,  # 20% less growth than baseline
                "volatility": 1.2      # 20% more volatility than baseline
            }
        elif scenario_type == "custom":
            # Convert percentage to factor (e.g., 10% becomes 1.1, -10% becomes 0.9)
            adjustment_factor = 1 + (custom_adjustment / 100)
            return {
                "growth_factor": adjustment_factor,
                "volatility": 1.0      # Default volatility
            }
        else:  # baseline
            return {
                "growth_factor": 1.0,
                "volatility": 1.0
            }
    
    def _apply_scenario_adjustments(self, forecast_df: pd.DataFrame, adjustments: Dict) -> pd.DataFrame:
        """Apply scenario adjustments to baseline forecast"""
        # Create a copy of the forecast
        adjusted_df = forecast_df.copy()
        
        # Get the variable name (excluding date, lower_ci, upper_ci)
        variable = [col for col in forecast_df.columns if col not in ['date', 'lower_ci', 'upper_ci']][0]
        
        # Apply growth factor adjustment
        growth_factor = adjustments.get("growth_factor", 1.0)
        adjusted_df[variable] = forecast_df[variable] * growth_factor
        
        # Apply volatility adjustment to confidence intervals
        volatility = adjustments.get("volatility", 1.0)
        adjusted_df['lower_ci'] = forecast_df['lower_ci'] * growth_factor * (1 - (1 - 0.9) * volatility)
        adjusted_df['upper_ci'] = forecast_df['upper_ci'] * growth_factor * (1 + (1.1 - 1) * volatility)
        
        return adjusted_df
