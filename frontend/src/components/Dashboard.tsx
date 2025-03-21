import React, { useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
} from '@mui/material';
import Plot from 'react-plotly.js';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

const Dashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState('7d');
  const [layout, setLayout] = useState([]);
  const [widgets, setWidgets] = useState([]);

    const addWidget = () => {
    const newWidget = {
      i: `widget-${widgets.length}`,
      x: 0,
      y: 0,
      w: 4,
      h: 4,
      content: <div>New Widget</div>,
    };
    setWidgets([...widgets, newWidget]);
    setLayout([...layout, newWidget]);
  };

  const removeWidget = (i) => {
    setWidgets(widgets.filter((widget) => widget.i !== i));
    setLayout(layout.filter((item) => item.i !== i));
  };

  const { data: metricsData, isLoading: metricsLoading } = useQuery(
    ['metrics', timeRange],
    async () => {
      const response = await axios.get('/api/metrics/summary');
      return response.data;
    }
  );

  const { data: trendData, isLoading: trendLoading } = useQuery(
    ['trends', timeRange],
    async () => {
      const response = await axios.post('/api/analysis/trend', {
        timeRange,
      });
      return response.data;
    }
  );

  const renderMetricCard = (title: string, value: number, change: number) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography color="textSecondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4">{value}</Typography>
        <Typography
          color={change >= 0 ? 'success.main' : 'error.main'}
          variant="body2"
        >
          {change >= 0 ? '+' : ''}{change}% vs last period
        </Typography>
      </CardContent>
    </Card>
  );

  if (metricsLoading || trendLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="50vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
        <div>
      <Button onClick={addWidget}>Add Widget</Button>
      <ResponsiveGridLayout
        className="layout"
        layouts={{ lg: layout }}
        breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
      >
        {widgets.map((widget) => (
          <div key={widget.i} data-grid={widget}>
            <Paper>
              {widget.content}
              <Button onClick={() => removeWidget(widget.i)}>Remove</Button>
            </Paper>
          </div>
        ))}
      </ResponsiveGridLayout>
    </div>
    
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Analytics Dashboard</Typography>
        <Box>
          {['7d', '30d', '90d'].map((range) => (
            <Button
              key={range}
              variant={timeRange === range ? 'contained' : 'outlined'}
              onClick={() => setTimeRange(range)}
              sx={{ ml: 1 }}
            >
              {range}
            </Button>
          ))}
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Metric Cards */}
        <Grid item xs={12} md={4}>
          {renderMetricCard('Total Revenue', 150000, 12.5)}
        </Grid>
        <Grid item xs={12} md={4}>
          {renderMetricCard('Active Users', 2500, -2.1)}
        </Grid>
        <Grid item xs={12} md={4}>
          {renderMetricCard('Conversion Rate', 3.2, 0.8)}
        </Grid>

        {/* Trend Analysis */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Trend Analysis
            </Typography>
            <Plot
              data={[
                {
                  x: trendData?.dates || [],
                  y: trendData?.values || [],
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Trend',
                },
              ]}
              layout={{
                autosize: true,
                height: 400,
                margin: { l: 50, r: 50, t: 30, b: 50 },
              }}
              useResizeHandler
              style={{ width: '100%' }}
            />
          </Paper>
        </Grid>

        {/* Predictive Analytics */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Predictions
            </Typography>
            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Forecast',
                  line: { dash: 'dot' },
                },
              ]}
              layout={{
                autosize: true,
                height: 300,
                margin: { l: 50, r: 50, t: 30, b: 50 },
              }}
              useResizeHandler
              style={{ width: '100%' }}
            />
          </Paper>
        </Grid>

        {/* Pattern Detection */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Pattern Detection
            </Typography>
            <Plot
              data={[
                {
                  type: 'heatmap',
                  z: [[1, 20, 30], [20, 1, 60], [30, 60, 1]],
                },
              ]}
              layout={{
                autosize: true,
                height: 300,
                margin: { l: 50, r: 50, t: 30, b: 50 },
              }}
              useResizeHandler
              style={{ width: '100%' }}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
