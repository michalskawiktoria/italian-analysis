# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load dataset
data = pd.read_csv("dataset.csv")
ateco = pd.read_excel("Ateco.xls")

# Example: merge with ATECO mapping (assuming 'client_id' and 'ateco_code')
data = data.merge(ateco, on="ateco_code", how="left")

# 3. Feature engineering (simplified RFM example)
data['Recency'] = (pd.to_datetime("today") - pd.to_datetime(data['last_purchase_date'])).dt.days
data['Frequency'] = data.groupby('client_id')['purchase_id'].transform('count')
data['Monetary'] = data.groupby('client_id')['revenue'].transform('sum')

# Label high-value customers (top 20% by revenue)
threshold = data['Monetary'].quantile(0.80)
data['HighValue'] = (data['Monetary'] >= threshold).astype(int)

# Select features for SVM
features = data[['Recency', 'Frequency', 'Monetary']].drop_duplicates()
labels = data[['HighValue']].drop_duplicates()

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train SVM classifier
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train_scaled, y_train.values.ravel())

# 7. Predictions
y_pred = svm_model.predict(X_test_scaled)

# 8. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
