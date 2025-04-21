# high precision and then high recall solution
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from source.classes.FeatureSelector import FeatureSelector

# ---------------------- Setup ----------------------
selector = FeatureSelector()

all_features = [
    'requirements', 'benefits', 'type_of_contract', 'type_of_position',
    'required_education', 'nature_of_company', 'nature_of_job',
    'company_profile_and_description', 'salary','title', 'function', 'company_profile', 'description'
]

meta_features = ['requirements', 'company_profile_and_description', 'benefits', 'title', 'function', 'company_profile', 'description']

# ---------------------- Base Models (High Precision) ----------------------
base_models = {}
train_meta_probs = []
train_meta_labels = None

for feature in meta_features:
    df = selector.make_df_from_features([feature])
    df.fillna(0, inplace=True)

    X = df.drop(columns='fraudulent')
    y = df['fraudulent']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train high-precision base model
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    # Save model
    base_models[feature] = model

    # Collect meta features for second layer
    probs = model.predict_proba(X_train)[:, 1]
    train_meta_probs.append(probs)

    # Store labels only once
    if train_meta_labels is None:
        train_meta_labels = y_train
        train_meta_index = X_train.index

# Stack into meta-feature matrix
meta_X_train = np.stack(train_meta_probs, axis=1)
meta_y_train = train_meta_labels

# ---------------------- Identify Likely Fraud Cases (High Precision Passes) ----------------------
# Flag cases where any base model gives high confidence of fraud (> 0.5, can be tuned). reducing threshold increases recall and reduces the precision
threshold = 0.3
flagged_idx = np.any(meta_X_train > threshold, axis=1)

X_meta_flagged = meta_X_train[flagged_idx]
y_meta_flagged = meta_y_train.iloc[flagged_idx]

# ---------------------- Final Model (High Recall) ----------------------
# Use full features for final model training
df_all = selector.make_df_from_features(all_features)
df_all.fillna(0, inplace=True)
X_all = df_all.drop(columns='fraudulent')
y_all = df_all['fraudulent']

# Split according to flagged index (using same training split)
X_all_train = X_all.loc[train_meta_index]
X_all_flagged = X_all_train.iloc[flagged_idx]
y_all_flagged = y_all.loc[X_all_flagged.index]

# Train high-recall model
smote = SMOTE(random_state=42)
X_final_resampled, y_final_resampled = smote.fit_resample(X_all_flagged, y_all_flagged)

recall_model = HistGradientBoostingClassifier(
    class_weight={0: 1., 1: len(y_all_flagged) / (2 * sum(y_all_flagged == 1))}, random_state=42)

recall_model.fit(X_final_resampled, y_final_resampled)

# ---------------------- Test Phase ----------------------
# Step 1: Base model probabilities
test_meta_probs = []
test_labels = None
test_meta_index = None

for feature in meta_features:
    df_test = selector.make_df_from_features([feature])
    df_test.fillna(0, inplace=True)

    X_test = df_test.drop(columns='fraudulent')
    y_test = df_test['fraudulent']

    X_train_, X_test_f, y_train_, y_test_f = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    probs = base_models[feature].predict_proba(X_test_f)[:, 1]
    test_meta_probs.append(probs)

    if test_labels is None:
        test_labels = y_test_f
        test_meta_index = X_test_f.index

meta_X_test = np.stack(test_meta_probs, axis=1)

# Step 2: Flag test samples with high fraud probability
test_flagged_idx = np.any(meta_X_test > threshold, axis=1)

# Step 3: Gather corresponding full features
df_all_test = selector.make_df_from_features(all_features)
df_all_test.fillna(0, inplace=True)

X_all_test = df_all_test.drop(columns='fraudulent')
y_all_test = df_all_test['fraudulent']

X_test_all_split = X_all_test.loc[test_meta_index]
X_test_flagged = X_test_all_split.iloc[test_flagged_idx]
y_test_flagged = y_all_test.loc[X_test_flagged.index]

# Step 4: Final prediction with recall model
final_preds = recall_model.predict(X_test_flagged)

# ---------------------- Evaluation ----------------------
print(f"=== Final Output (High Precision Base → High Recall Final) with threshold of {threshold} ===")
print(classification_report(y_test_flagged, final_preds))

# === Final Output (High Precision Base → High Recall Final) with threshold of 0.3 ===
#               precision    recall  f1-score   support
#
#            0       0.99      0.98      0.99      2845
#            1       0.77      0.87      0.81       181
#
#     accuracy                           0.98      3026
#    macro avg       0.88      0.93      0.90      3026
# weighted avg       0.98      0.98      0.98      3026

