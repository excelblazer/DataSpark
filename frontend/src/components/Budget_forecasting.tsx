import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  TextField,
  Typography,
} from '@mui/material';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';
import Plot from 'react-plotly.js';

const BudgetForecast: React.FC = () => {
  const [forecastPeriods, setForecastPeriods] = useState<number>(12);
  const [targetVariables, setTargetVariables] = useState<string[]>(['revenue', 'cost']);
  const [seasonality, setSeasonality] = useState<boolean>(true);
  const [growthRate, setGrowthRate] = useState<{[key: string]: number}>({
    revenue: 0.05,
    cost: 0.03
  });
  const [selectedDataSource, setSelectedDataSource] = useState<string>('');
  const [dataSourceOptions] = useState<{id: string, name: string}[]>([
    { id: 'financial_data', name: 'Financial Data' },
    { id: 'operational_data', name: 'Operational Data' },
    { id: 'sales_data', name: 'Sales Data' }
  ]);

  const createBudgetMutation = useMutation({
    mutationFn: (forecastRequest: any) => {
      return axios.post('/api/analysis/budget_forecast', forecastRequest);
    }
  });

  const handleCreateBudget = () => {
    const request = {
      data: {
        source: selectedDataSource,
        values: [] // In a real app, you'd fetch this based on selectedDataSource
      },
      config: {
        periods: forecastPeriods,
        target_variables: targetVariables,
        growth_assumptions: growthRate,
        seasonality: seasonality,
        include_historical: true
      }
    };
    
    createBudgetMutation.mutate(request);
  };

  const handleGrowthRateChange = (variable: string, value: number) => {
    setGrowthRate(prev => ({
      ...prev,
      [variable]: value
    }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Budget Forecasting
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Forecast Configuration
              </Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Data Source</InputLabel>
                <Select
                  value={selectedDataSource}
                  onChange={(e) => setSelectedDataSource(e.target.value)}
                >
                  {dataSourceOptions.map(option => (
                    <MenuItem key={option.id} value={option.id}>
                      {option.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <TextField
                fullWidth
                margin="normal"
                label="Forecast Periods"
                type="number"
                value={forecastPeriods}
                onChange={(e) => setForecastPeriods(Number(e.target.value))}
              />
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Seasonality</InputLabel>
                <Select
                  value={seasonality.toString()}
                  onChange={(e) => setSeasonality(e.target.value === 'true')}
                >
                  <MenuItem value="true">Yes</MenuItem>
                  <MenuItem value="false">No</MenuItem>
                </Select>
              </FormControl>
              
              <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
                Growth Assumptions
              </Typography>
              
              {targetVariables.map(variable => (
                <TextField
                  key={variable}
                  fullWidth
                  margin="normal"
                  label={`${variable.charAt(0).toUpperCase() + variable.slice(1)} Growth Rate`}
                  type="number"
                  InputProps={{
                    inputProps: { step: 0.01, min: -0.5, max: 0.5 },
                    endAdornment: '%'
                  }}
                  value={growthRate[variable] * 100}
                  onChange={(e) => handleGrowthRateChange(variable, Number(e.target.value) / 100)}
                />
              ))}
              
              <Button
                variant="contained"
                color="primary"
                fullWidth
                sx={{ mt: 2 }}
                onClick={handleCreateBudget}
                disabled={createBudgetMutation.isLoading || !selectedDataSource}
              >
                {createBudgetMutation.isLoading ? <CircularProgress size={24} /> : 'Generate Budget Forecast'}
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, height: '100%' }}>
            {createBudgetMutation.isLoading ? (
              <Box display="flex" justifyContent="center" alignItems="center" height="300px">
                <CircularProgress />
              </Box>
            ) : createBudgetMutation.isError ? (
              <Typography color="error">
                Error generating forecast: {createBudgetMutation.error.message}
              </Typography>
            ) : createBudgetMutation.data ? (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Budget Forecast Results
                </Typography>
                
                <Plot
                  data={[
                    {
                      x: createBudgetMutation.data.data?.budget_forecast?.map((item: any) => item.date) || [],
                      y: createBudgetMutation.data.data?.budget_forecast?.map((item: any) => item.revenue_forecast) || [],
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Revenue Forecast',
                      line: { color: '#1976d2' }
                    },
                    {
                      x: createBudgetMutation.data.data?.budget_forecast?.map((item: any) => item.date) || [],
                      y: createBudgetMutation.data.data?.budget_forecast?.map((item: any) => item.cost_forecast) || [],
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Cost Forecast',
                      line: { color: '#dc004e' }
                    },
                    {
                      x: createBudgetMutation.data.data?.budget_forecast?.map((item: any) => item.date) || [],
                      y: createBudgetMutation.data.data?.budget_forecast?.map((item: any) => item.profit_forecast) || [],
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Profit Forecast',
                      line: { color: '#4caf50' }
                    }
                  ]}
                  layout={{
                    autosize: true,
                    height: 400,
                    margin: { l: 50, r: 50, t: 30, b: 50 },
                    legend: { orientation: 'h', y: -0.2 }
                  }}
                  useResizeHandler
                  style={{ width: '100%' }}
                />
                
                <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
                  Assumptions:
                </Typography>
                <Typography variant="body2">
                  Forecast period: {createBudgetMutation.data.data?.forecast_period}
                </Typography>
                <Typography variant="body2">
                  Generated at: {createBudgetMutation.data.data?.generated_at}
                </Typography>
              </Box>
            ) : (
              <Box display="flex" justifyContent="center" alignItems="center" height="300px">
                <Typography variant="body1">
                  Configure and generate a budget forecast to see results
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default BudgetForecast;
