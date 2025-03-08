# Project-Futbizz (FIFA 19 Player Recommendation & Bid Prediction System)

- check out the NEW web app here
https://huggingface.co/spaces/dcrey7/Fifa19_webapp
- youtube link 
https://www.youtube.com/watch?v=_nmrCul-JOw


## Overview
This web application helps football managers and scouts make data-driven decisions by providing player recommendations and market value predictions using FIFA 19 dataset. The system combines machine learning algorithms (KNN for player recommendations and XGBoost for bid predictions) to deliver actionable insights.

## Business Problem
Football clubs face several challenges in player recruitment:
- Finding similar players to replace departing team members
- Identifying undervalued talents in the market
- Making informed decisions about player valuations
- Optimizing recruitment budget allocation

This application addresses these challenges by providing:
1. Data-driven player recommendations based on similar playing styles and attributes
2. Market value predictions to assist in negotiations
3. Comprehensive player statistics for informed decision-making

## Features

### 1. Player Recommendation System
- Utilizes K-Nearest Neighbors (KNN) algorithm
- Considers 30+ player attributes including:
  - Technical skills (dribbling, shooting, passing)
  - Physical attributes (pace, strength, stamina)
  - Mental attributes (vision, positioning, composure)
- Recommends similar players based on playing style and attributes
- Helps identify alternative recruitment targets

### 2. Bid Prediction System
- Powered by XGBoost regression model
- Predicts market value based on:
  - Player statistics and attributes
  - Age and potential
  - Current club and contract status
- Helps in budget planning and negotiations
- Identifies potentially undervalued players

## Technical Architecture

### Key Components

#### 1. Web Application (main.py)
- Flask-based web server
- Routes handling
- API endpoints implementation
- Model inference integration

#### 2. Data Processing Pipeline
```python
Data Flow:
Raw Data → Preprocessing → Feature Engineering → Model Training → Serialization
```

#### 3. Model Architecture
```
Recommendation System (KNN)
└── Input → Feature Scaling → KNN Model → Similarity Scores → Top N Recommendations

Bid Prediction (XGBoost)
└── Input → Preprocessing → Feature Engineering → XGBoost Model → Value Prediction
```

### Technology Stack

#### Backend
- Python 3.8+
- Flask 1.1.2
- NumPy 1.19.5
- Pandas 1.2.4
- Scikit-learn 0.24.2
- XGBoost 1.4.2

#### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap 4.5

#### Development Tools
- Git for version control
- Jupyter Notebooks for model development
- VS Code/PyCharm for development

#### Deployment
- Heroku Platform
- Gunicorn web server

## Detailed Methodology

### 1. Data Preprocessing Strategy

#### Initial Data Cleaning
- Removed special characters from monetary values (€, M, K)
- Converted height and weight to numerical values
- Standardized player positions
- Handled missing values using mean/median imputation
- Converted categorical variables using one-hot encoding
- Normalized numerical features using StandardScaler

#### Feature Engineering
- Created composite skill metrics
- Generated position-specific attributes
- Calculated age-based potential indicators
- Engineered monetary features (wage-to-value ratio)
- Aggregregating team level and position wise features
- Created physical attribute indices

#### Feature Selection
- Removed highly correlated features (correlation threshold > 0.85)
- Used feature importance from Random Forest to select top predictors
- Applied domain knowledge to retain crucial football attributes

### 2. Player Recommendation System (KNN)

#### Model Details
- Algorithm: K-Nearest Neighbors
- Distance Metric: Euclidean Distance (performed better than Manhattan)
- K Value: 5 (determined through cross-validation)
- Weights: Distance-weighted voting

#### Feature Processing for KNN
- Standardization of all numerical features
- Features grouped into categories:
  1. Technical Abilities (dribbling, shooting, passing, etc.)
  2. Physical Attributes (pace, strength, stamina)
  3. Mental Attributes (vision, positioning)
  4. Position-specific metrics

#### Similarity Calculation
- Normalized Euclidean distance
- Custom weighting for position-specific attributes
- Scaled similarity scores (0-100)

### 3. Bid Prediction Modeling

#### Data Split
- Training Set: 80% (14,565 players)
- Testing Set: 20% (3,642 players)
- Stratified split based on player overall rating

#### Model Comparison
1. **Linear Regression**
   - Baseline model
   - R² Score: 0.72
   - RMSE: 0.1543

2. **Random Forest**
   - n_estimators: 100
   - max_depth: 15
   - R² Score: 0.85
   - RMSE: 0.1123

3. **XGBoost (Final Model)**
   - Best performing model
   - Hyperparameters:
     ```python
     {
         'learning_rate': 0.1,
         'max_depth': 5,
         'min_child_weight': 1,
         'n_estimators': 200,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'gamma': 0.1
     }
     ```
   - R² Score: 0.89
   - RMSE: 0.0891
   - MAE: 0.0654

#### Cross-Validation Strategy
- 5-fold cross-validation
- Stratified K-Fold for maintaining player rating distribution
- Grid Search CV for hyperparameter tuning

#### Feature Importance (Top 10)
1. Overall Rating (0.285)
2. Potential (0.156)
3. Age (0.098)
4. International Reputation (0.087)
5. Skill Moves (0.076)
6. Weak Foot (0.065)
7. Position (0.058)
8. Composure (0.045)
9. Reactions (0.042)
10. Ball Control (0.038)

### 4. Model Evaluation Metrics

#### KNN Recommendation System
- Silhouette Score: 0.68
- Davies-Bouldin Index: 0.42
- Position Accuracy: 92%
- Playing Style Similarity: 85%

#### Bid Prediction (XGBoost)
- Mean Absolute Percentage Error (MAPE): 8.76%
- R² Score: 0.89
- RMSE: 0.0891
- MAE: 0.0654
- Explained Variance Score: 0.892

### 5. Model Deployment

#### Hosting
- Models saved using pickle format
- Heoruko free tier


