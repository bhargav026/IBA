'''
Imagine you have a dataset where you have different features like Age , Gender , Height , Weight , BMI , and Blood Pressure and 
you have to classify the people into different classes like Normal , Overweight , Obesity , Underweight , and Extreme Obesity by using any 4 different classification algorithms.
 Now you have to build a model which can classify people into different classes.
https://www.kaggle.com/datasets/ankurbajaj9/obesity-levels This is the Dataset You can use this dataset for this question.
'''


import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data, meta = arff.loadarff('ObesityDataSet_raw_and_data_sinthetic.arff')
data = pd.DataFrame(data)
data = data.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Extract the feature matrix X and the target variable y
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Custom transformer for label encoding
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X, y=None):
        X_encoded = X.copy()
        for col, le in self.label_encoders.items():
            X_encoded[col] = le.transform(X[col])
        return X_encoded

# Define the column transformer for encoding categorical variables and scaling numerical variables
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float', 'int']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('label_encoder', LabelEncoderTransformer(), categorical_cols.tolist()),
        ('scaler', StandardScaler(), numerical_cols.tolist())
    ])

# Define the classification models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}
results = {}
for model_name, model in models.items():
    # Create the pipeline with preprocessing and the classification model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    results[model_name] = classification_report(y_test, y_pred,output_dict=True)
    # Print the results
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print()

best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"Best Model: {best_model}")
print("Best Model Classification Report:")
print(results[best_model])


#creating the best model
pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', models[best_model])
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuaracy: ', accuracy)
