from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from source.classes.FeatureSelector import FeatureSelector
import joblib

selector = FeatureSelector()

total_features = [
    'requirements', 'benefits', 'type_of_contract', 'type_of_position',
    'required_education', 'nature_of_company', 'nature_of_job',
    'company_profile_and_description', 'title', 'function', 'description', 'company_profile'
]


# 1. Select the features
df = selector.make_df_from_features(total_features)

X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

X.fillna(0, inplace=True)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Apply SMOTE to over-sample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 4. class weights (emphasize fraud class)
class_weights = {0: 1., 1: 2}

# 5. Train HistGradientBoostingClassifier with hyperparameters from a random search
hgb = HistGradientBoostingClassifier(
    class_weight=class_weights,
    learning_rate=0.05,             # Lower learning rate for stability
    max_iter=500,                   # More boosting iterations (default is 100)
    max_leaf_nodes=31,              # Controls tree complexity
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    scoring='loss',                 # Use loss minimization for early stopping
    random_state=42
)

hgb.fit(X_train_resampled, y_train_resampled)

y_pred = hgb.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))


# Save the trained model to disk
joblib.dump(hgb, '../models/hgb_classifier_v1.joblib')

# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.99      1.00      1.00      3403
#            1       0.95      0.85      0.90       173
#
#     accuracy                           0.99      3576
#    macro avg       0.97      0.92      0.95      3576
# weighted avg       0.99      0.99      0.99      3576
