import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =========================
# Load Dataset
# =========================
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)

# =========================
# Data Cleaning
# =========================
df.dropna(inplace=True)

df.rename(columns={
    'RainToday': 'RainYesterday',
    'RainTomorrow': 'RainToday'
}, inplace=True)

df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]

# =========================
# Feature Engineering
# =========================
def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)
df.drop(columns=['Date'], inplace=True)

# =========================
# Split Features & Target
# =========================
X = df.drop(columns='RainToday')
y = df['RainToday']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# Preprocessing
# =========================
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =========================
# RANDOM FOREST MODEL
# =========================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)

rf_grid.fit(X_train, y_train)

print("Best Random Forest Parameters:", rf_grid.best_params_)
print("Best CV Accuracy:", rf_grid.best_score_)

rf_test_score = rf_grid.score(X_test, y_test)
print("Test Accuracy:", rf_test_score)

y_pred_rf = rf_grid.predict(X_test)

print("\nRandom Forest Classification Report")
print(classification_report(y_test, y_pred_rf))

# =========================
# Random Forest Confusion Matrix
# =========================
conf_matrix = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

# =========================
# Feature Importance
# =========================
feature_names = numeric_features + list(
    rf_grid.best_estimator_['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)

importances = rf_grid.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_features = importance_df.head(20)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.gca().invert_yaxis()
plt.title("Top 20 Important Features")
plt.xlabel("Importance")
plt.show()

# =========================
# LOGISTIC REGRESSION MODEL
# =========================
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

lr_param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)

lr_grid.fit(X_train, y_train)

y_pred_lr = lr_grid.predict(X_test)

print("\nLogistic Regression Classification Report")
print(classification_report(y_test, y_pred_lr))

# =========================
# Logistic Regression Confusion Matrix
# =========================
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure()
sns.heatmap(conf_matrix_lr, annot=True, cmap='Blues', fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
