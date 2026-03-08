# churn_prediction_system
churn_prediction_system/
│
├── data/
│   ├── raw/                # Raw, unprocessed data (CSV, JSON, etc.)
│   ├── processed/          # Cleaned and feature-engineered datasets
│   └── external/           # Any external data sources (APIs, third-party)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_baseline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py     # ETL scripts for cleaning raw data
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train_model.py      # Training & evaluation
│   │   └── predict_model.py    # Inference script
│   │
│   ├── api/
│   │   ├── app.py              # FastAPI main application
│   │   └── routes.py           # API endpoints
│   │
│   ├── utils/
│   │   ├── config.py           # Configs (paths, parameters)
│   │   └── helpers.py          # Helper functions
│   │
│   └── visualization/
│       └── dashboard.py        # Streamlit or plotting scripts
│
├── tests/
│   ├── test_data.py            # Unit tests for data processing
│   ├── test_models.py          # Unit tests for model code
│   └── test_api.py             # Unit tests for API endpoints
│
├── docker/
│   ├── Dockerfile              # Dockerfile for API service
│   └── docker-compose.yml      # Optional, for running multiple services
│
├── mlruns/                     # MLflow experiment tracking folder (auto)
│
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
└── setup.py                    # Optional, if packaging as a module