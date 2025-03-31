# # Without perfoming any oversampling and using balanced class weights

# In[280]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# 1. Load dataset (assuming 'target' is the column to predict)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train HistGradientBoostingClassifier (handles NaNs automatically)
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
hgb.fit(X_train, y_train)

# 4. Make predictions
y_pred = hgb.predict(X_test)

# 5. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:


# # Using balanced class weights

# In[284]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

# 1. Load dataset (assuming 'fraudulent' is the column to predict)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Train HistGradientBoostingClassifier with class_weight='balanced'
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
hgb.fit(X_train_resampled, y_train_resampled)

# 6. Make predictions
y_pred = hgb.predict(X_test)

# 7. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:


# # Oversampling and using balanced class weights

# In[286]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # SMOTE for oversampling

# 1. Load dataset (assuming 'fraudulent' is the target variable)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Train RandomForestClassifier with class_weight='balanced' to handle class imbalance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_resampled, y_train_resampled)

# 6. Make predictions
y_pred = rf.predict(X_test)

# 7. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:


# # Using calculated class weights and oversampling

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

# 1. Load dataset (assuming 'fraudulent' is the column to predict)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to oversample the minority class in the training data
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

# In[ ]:




