# # Using calculated class weights and oversampling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from source.classes.FeatureSelector import FeatureSelector

selector = FeatureSelector()

total_features = [
            'requirements', 'benefits', 'type_of_contract', 'type_of_position',
            'required_education', 'nature_of_company', 'nature_of_job',
            'company_profile_and_description', 'salary'
        ]

# 1. Select the features
df = selector.make_df_from_features(total_features)

X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to over-sample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Compute manual class weights (giving more weight to the minority class)
class_weights = {0: 1., 1: len(y_train) / (2 * np.sum(y_train == 1))}

# 6. Train HistGradientBoostingClassifier with manually calculated class weights
hgb = HistGradientBoostingClassifier(class_weight=class_weights, random_state=42)
hgb.fit(X_train_resampled, y_train_resampled)

# 7. Make predictions
y_pred = hgb.predict(X_test)

# 8. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.99      0.96      0.98      3395
#            1       0.56      0.86      0.68       181
#
#     accuracy                           0.96      3576
#    macro avg       0.78      0.91      0.83      3576
# weighted avg       0.97      0.96      0.96      3576
#
#
# Process finished with exit code 0