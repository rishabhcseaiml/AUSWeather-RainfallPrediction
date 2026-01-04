# AUSWeather-RainfallPrediction


## Project Overview
This project builds a machine learning classifier to predict whether it will rain tomorrow in the Melbourne area, Australia. The model uses historical weather data and implements a complete ML pipeline with preprocessing, hyperparameter tuning, and model evaluation.

## Dataset
- **Source**: Australian Government's Bureau of Meteorology
- **Features**: 22 weather-related variables (temperature, humidity, wind, pressure, etc.)
- **Locations**: Melbourne area (Melbourne, MelbourneAirport, Watsonia)
- **Records**: 7,557 observations after preprocessing
- **Target**: RainToday (Yes/No - binary classification)

## Project Structure

├── notebooks/
│ └── FinalProject_AUSWeather.ipynb
├── data/
│ └── weatherAUS-2.csv
├── scripts/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ └── model_training.py
├── results/
│ ├── model_performance.txt
│ └── feature_importances.csv
├── requirements.txt
└── README.md
