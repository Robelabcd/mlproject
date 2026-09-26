## end to end machine learning project

 # Student Performance Prediction: End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Elastic%20Beanstalk-FF9900?logo=amazonaws&logoColor=white)

An end-to-end machine learning project that predicts a student's **math score** from demographic information, test preparation, and their reading and writing scores. It covers the full workflow: exploratory data analysis, model experimentation, a modular and reusable training pipeline, a Flask web app for real-time predictions, and deployment configuration for AWS Elastic Beanstalk.

<!-- Add a screenshot of the prediction form here -->
<!-- ![Prediction app](docs/app_screenshot.png) -->

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)

## Problem Statement

This project looks at how student performance on exams relates to factors like gender, ethnicity, parental level of education, lunch type, and test preparation. The goal is a regression model that predicts a student's math score from those factors plus their reading and writing scores.

## Dataset

- **Source:** [Students Performance in Exams (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size:** 1,000 rows, 8 columns
- **Target:** `math_score`

| Feature                        | Type        | Description                                        |
|--------------------------------|-------------|----------------------------------------------------|
| `gender`                       | Categorical | Male / female                                      |
| `race_ethnicity`               | Categorical | Group A–E                                          |
| `parental_level_of_education`  | Categorical | Highest education level of the student's parents   |
| `lunch`                        | Categorical | Standard or free/reduced                           |
| `test_preparation_course`      | Categorical | Completed or none                                  |
| `reading_score`                | Numerical   | Reading exam score (0–100)                         |
| `writing_score`                | Numerical   | Writing exam score (0–100)                         |

## Project Architecture

```
            ┌──────────────────┐     ┌───────────────────────┐     ┌──────────────────┐
stud.csv ──▶│  Data Ingestion  │────▶│  Data Transformation  │────▶│  Model Trainer   │
            │  80/20 split     │     │  impute, encode, scale│     │  GridSearchCV    │
            └──────────────────┘     └───────────────────────┘     └──────────────────┘
                     │                          │                           │
                     ▼                          ▼                           ▼
              artifacts/*.csv         artifacts/preprocessor.pkl     artifacts/model.pkl
                                                │                           │
                                                └────────────┬──────────────┘
                                                             ▼
                                   User input ──▶  Predict Pipeline  ──▶  Flask web app
```

## Project Structure

```
mlproject/
├── app.py                          # Flask app (entry point: app:application)
├── setup.py                        # Packages src/ as an installable module
├── requirement.txt                 # Python dependencies
├── .ebextensions/
│   └── python.config               # AWS Elastic Beanstalk WSGI config
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/stud.csv               # Raw dataset
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads raw data, creates train/test split
│   │   ├── data_transformation.py  # Builds and saves the preprocessing pipeline
│   │   └── model_trainer.py        # Trains, tunes, and selects the best model
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py     # Loads artifacts and serves predictions
│   ├── exception.py                # Custom exception with file and line details
│   ├── logger.py                   # Timestamped file logging
│   └── utils.py                    # Save/load objects, model evaluation
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form
└── artifacts/                      # Generated data splits, preprocessor, model
```

## Modeling Approach

**1. Data ingestion.** The raw CSV is loaded, saved to `artifacts/data.csv`, and split 80/20 into train and test sets (`random_state=42`).

**2. Data transformation.** A scikit-learn `ColumnTransformer` handles each feature type separately:

- **Numerical** (`reading_score`, `writing_score`): median imputation, then `StandardScaler`.
- **Categorical** (the five demographic features): most-frequent imputation, then `OneHotEncoder`, then `StandardScaler(with_mean=False)`.

The fitted preprocessor is saved as `preprocessor.pkl`, so the exact same transformations are applied at inference time.

**3. Model training and selection.** Seven regressors are tuned with `GridSearchCV` (3-fold CV) and compared by R² on the held-out test set:

Linear Regression · Decision Tree · Random Forest · Gradient Boosting · XGBoost · CatBoost · AdaBoost

The best-scoring model is saved to `model.pkl`. A model is only accepted if its test R² is at least 0.6.

**4. Engineering practices.**

- Modular, config-driven components built with `@dataclass` configs.
- A custom exception class that reports the file name and line number of every error.
- Timestamped log files for every run.
- The project is packaged with `setup.py` so `src` can be imported anywhere.

## Results

Test-set R² from the model comparison in `notebook/2. MODEL TRAINING.ipynb`:

| Model                   | Test R² |
|-------------------------|:-------:|
| Ridge                   | 0.8806  |
| **Linear Regression**   | **0.8804** |
| Random Forest           | 0.8521  |
| CatBoost                | 0.8516  |
| AdaBoost                | 0.8434  |
| XGBoost                 | 0.8278  |
| Lasso                   | 0.8253  |
| K-Neighbors             | 0.7837  |
| Decision Tree           | 0.7407  |

The linear models generalize best. The tree-based models reach near-perfect R² on the training set (0.99+ for Decision Tree and XGBoost) but score much lower on the test set, which shows clear overfitting. The strong linear relationship between reading, writing, and math scores explains why simple models win here. The training pipeline also selected **Linear Regression** as the deployed model.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` and (optionally) `conda` or `venv`

### Installation

```bash
git clone https://github.com/Robelabcd/mlproject.git
cd mlproject

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies and the src package
pip install -r requirement.txt
pip install -e .
```

## Usage

### Train the model

Run the full pipeline (ingestion, transformation, training) from the project root:

```bash
python -m src.components.data_ingestion
```

This regenerates everything in `artifacts/` and prints the best model's test R².

### Run the web app

```bash
python app.py
```

Open [http://localhost:5000/predictdata](http://localhost:5000/predictdata), fill in the student's details, and click **Predict** to get the estimated math score.

## Deployment

The project includes configuration for **AWS Elastic Beanstalk**. The `.ebextensions/python.config` file points the WSGI server at the Flask object:

```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: app:application
```

To deploy, create a Python Elastic Beanstalk environment and connect it to this repository, either directly or through AWS CodePipeline.

## Future Improvements

- Add cross-validated metrics (RMSE, MAE) alongside R² in the training pipeline.
- Add unit tests for each pipeline component.
- Add input validation on the prediction form.
- Containerize the app with Docker and set up CI/CD with GitHub Actions.
- Implement `train_pipeline.py` as a single training entry point.

## Author

**Robel**, M.S. Computer Science, George Washington University

Feel free to open an issue or reach out with questions or suggestions.
