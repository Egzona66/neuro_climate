import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Load processed data
processed_data_file = "processed_data/mri_features.npz"  # Path to processed data
print("Loading processed data...")
data = np.load(processed_data_file)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']

# 2. Train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest classifier
print("Training Random Forest classifier...")
clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

# 4. Evaluate the model
print("Evaluating the model...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. Save results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
feature_importances = clf.feature_importances_
np.savetxt(os.path.join(results_dir, "feature_importances.csv"), feature_importances, delimiter=",")
print(f"Results saved in {results_dir}.")
