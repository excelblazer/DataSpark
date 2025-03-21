# DataBloom

DataBloom is a comprehensive analytics and data processing platform designed to provide insightful analytics, data cleaning, cross-module analysis, scenario simulation, and more. The platform leverages advanced machine learning and statistical models to generate actionable insights from various data sources.

## Core Capabilities

- **Data Processing & Cleaning**: Memory-efficient chunk processing for large datasets with robust error handling
- **Analytics & Insights**: Automated trend detection, anomaly identification, and predictive analytics
- **Cross-Module Analysis**: Connects siloed data across departments for integrated insights
- **Data Classification & Mapping**: Auto-categorizes datasets and maps relationships between data elements
- **Scenario Simulation**: Model different business scenarios with what-if analysis
- **Secure Data Management**: Encrypted data storage with granular access controls
- **Interactive Dashboards**: Real-time analytics visualizations with Plotly

## Architecture Components

### Data Processing Pipeline

- **DataCleaner**: Processes datasets in memory-efficient chunks with customizable thresholds for outlier detection, imputation strategies, and normalization
- **DataConsolidator**: Merges multiple data sources with type consistency validation and conflict resolution
- **DataMapper**: Maps semantic relationships between data elements with caching for performance optimization
- **MLEngine**: Provides machine learning capabilities for trend analysis, forecasting, and anomaly detection

### Analysis Services

- **AnalyticsRecommender**: Suggests relevant analytics operations based on data characteristics
- **ClassificationService**: Automatically classifies datasets by department and data type
- **CrossModuleAnalysis**: Identifies relationships between datasets across organizational boundaries
- **ScenarioSimulator**: Creates what-if scenarios for budget forecasting and business planning

### Infrastructure Components

- **WarehouseManager**: Encrypted data storage with metadata management
- **ExportManager**: Multi-format export capabilities (CSV, Excel, PDF, etc.)
- **ModuleRegistry**: Central registry of data modules and their definitions
- **NLPEngine**: Natural language processing for query understanding and data summarization

## Features

- Interactive dashboards with real-time analytics
- Machine learning-based trend detection and pattern analysis
- Predictive analytics using Prophet and scikit-learn
- Automated report generation
- Data source integration (ERP, SaaS, file uploads)
- Customizable visualizations using Plotly
- Modern React-based frontend with Material-UI
- Advanced data classification
- Budget forecasting
  
### Core Functionalities
- **Analytics Recommender**: Suggests analytics operations based on data characteristics.
- **Classification Services**: Automatically classifies datasets into appropriate categories.
- **Cross Module Analysis Services**: Analyzes and correlates data across different modules.
- **Data Consolidation**: Merges and cleans data from multiple sources into a single dataset.
- **Export Manager**: Exports data and visualizations in multiple formats (CSV, Excel, PDF, PPT, etc.).
- **Module Registry**: Maintains a registry of data modules and their definitions.
- **Scenario Simulator**: Simulates various business scenarios (optimistic, pessimistic, baseline, custom).
- **Summarizer**: Generates comprehensive summaries of datasets.
- **Data Cleaner**: Cleans, normalizes, and handles missing data.

### Frontend Components
- **Anomaly Detection**: Identifies anomalies in the data.
- **Budget Forecasting**: Forecasts budget based on historical and operational data.

## Technology Stack

### Backend

- **Framework**: FastAPI for high-performance API endpoints
- **Data Processing**: Pandas & NumPy for efficient data manipulation
- **Machine Learning**: Scikit-learn for analytics, Prophet for time series forecasting
- **Data Visualization**: Plotly for interactive charts and dashboards
- **Database**: SQLAlchemy ORM with support for multiple database backends
- **Authentication**: JWT-based authentication with role-based access control
- **Encryption**: Cryptography library for secure data storage

### Frontend

- **Framework**: React with TypeScript
- **UI Components**: Material-UI for modern, responsive interface
- **State Management**: React Context API
- **Data Visualization**: Plotly.js for interactive charts

## Project Structure
```
DataBloom/
├── app/                      # Backend Python application
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── analytics_recommender.py  # Analytics recommendations
│   ├── classification_service.py # Data classification
│   ├── config.py             # Configuration and settings
│   ├── connectors/           # Data source connectors
│   │   ├── base_connector.py # Base class for connectors
│   │   ├── file_connector.py # File-based data import
│   │   ├── rest_api_connector.py # API-based data import
│   │   └── sql_connector.py  # Database connectors
│   ├── cross_module_analysis.py # Cross-module analytics
│   ├── data_cleaner.py       # Data cleaning with chunk processing
│   ├── data_consolidation.py # Data consolidation services
│   ├── data_mapper.py        # Semantic data mapping
│   ├── data_processor.py     # Data processing pipeline
│   ├── database.py           # Database configuration
│   ├── export_manager.py     # Export functionality
│   ├── ml_engine.py          # Machine learning capabilities
│   ├── models.py             # Data models
│   ├── module_registry.py    # Module registry
│   ├── nlp_engine.py         # Natural language processing
│   ├── scenario_simulator.py # Scenario simulation
│   ├── summarizer.py         # Data summarization
│   └── warehouse_manager.py  # Data warehouse management
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── Dashboard.tsx # Main dashboard
│   │   │   ├── Budget_forecasting.tsx # Budget forecasting
│   │   │   ├── Anomaly_detection/     # Anomaly detection components
│   │   │   └── Layout.tsx    # Application layout
│   │   └── App.tsx           # Main React application
│   ├── public/               # Static assets
│   └── package.json          # Frontend dependencies
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_api.py           # API tests
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher
- MongoDB (optional, for data warehouse)
- PostgreSQL (optional, for persistent storage)

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configuration
# Be sure to change the SECRET_KEY and ENCRYPTION_KEY in production!
```

4. Start the backend server:
```bash
cd app
uvicorn main:app --reload
```

### Frontend Setup

1. Install frontend dependencies:
```bash
cd frontend
npm install
```

2. Start the frontend development server:
```bash
npm start
```

## Performance Optimizations

- **Chunk Processing**: Large datasets are processed in configurable chunks to minimize memory usage
- **Caching**: Semantic type detection and analytics recommendations use LRU cache for improved performance
- **Asynchronous Operations**: Time-consuming data operations run asynchronously with progress tracking
- **Query Optimization**: Database queries are optimized for performance with proper indexing
- **Parallel Processing**: Where appropriate, operations are parallelized for improved throughput

## Security Features

- **JWT Authentication**: Secure token-based authentication with configurable expiration
- **Data Encryption**: Warehouse data is encrypted using Fernet symmetric encryption
- **SSL Support**: Database connections can be secured with SSL
- **Environment Configuration**: Sensitive information is stored in environment variables
- **Error Handling**: Comprehensive error handling and logging
- **CORS Protection**: Configurable CORS settings to restrict access
- **Dependency Scanning**: Regular updates to dependencies to address security vulnerabilities

## API Endpoints

DataBloom provides a comprehensive API for integration with other systems:

### Authentication
- `POST /token`: Obtain JWT authentication token

### Data Management
- `POST /api/data/upload`: Upload data files
- `POST /api/data/clean`: Clean and normalize data
- `GET /api/warehouse/datasets`: List available datasets
- `GET /api/warehouse/data/{id}`: Retrieve dataset by ID

### Analytics
- `POST /api/analysis/trend`: Perform trend analysis
- `POST /api/analysis/predict`: Generate predictions
- `POST /api/analysis/recommend`: Recommend analytics operations
- `POST /api/analysis/anomalies`: Detect anomalies
- `POST /api/analysis/cross-module`: Perform cross-module analysis

### Reporting
- `POST /api/dashboard/create`: Create new dashboard
- `POST /api/report/generate`: Generate automated report
- `POST /api/budget/forecast`: Create budget forecasts

## Testing

Run the test suite to verify the application:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app

# Performance testing
pytest tests/test_data_consolidation.py -v --log-cli-level=INFO --durations=0

# Memory profiling
mprof run --python python -m app.main --consolidation-test
```

## API Versioning

The API is versioned to ensure backward compatibility:

- `/api/v1/...` - Current stable API
- Legacy endpoints are maintained but marked as deprecated

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
