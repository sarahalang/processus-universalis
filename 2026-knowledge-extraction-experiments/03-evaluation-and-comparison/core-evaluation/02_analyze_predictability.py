import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import re

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "../../data/labeled_segments.json")

# Mappings from Capstone Analysis
A_TO_E = {
    'a1': 'E16', 'a2': 'E37', 'a3': 'E38', 'a4': 'E44',
    'a5': 'E17', 'a6': 'E19', 'a7': 'E39', 'a8': 'E34',
    'a9': 'E2', 'a12': 'E45', 'a13': 'E42',
    'a15': 'E32b', 'a16': 'E27',
    'a21': 'E3', 'a22': 'E35', 'a25': 'E22', 'a26': 'E11'
}

E_TO_GROUP = {
    'E2': 'I', 'E3': 'I', 'E11': 'I', 'E22': 'I', 'E35': 'I',
    'E16': 'II', 'E17': 'II', 'E19': 'II', 'E27': 'II', 'E32b': 'II',
    'E34': 'III', 'E37': 'III', 'E38': 'III', 'E39': 'III',
    'E42': 'III', 'E44': 'III', 'E45': 'III',
}

def get_group(tid):
    m = re.search(r'a(\d+)', tid)
    if m:
        a_key = 'a' + m.group(1)
        ename = A_TO_E.get(a_key)
        return E_TO_GROUP.get(ename)
    return None

def analyze_predictability():
    if not os.path.exists(INPUT_JSON):
        print("Error: labeled_segments.json not found.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matrix = pd.DataFrame(data).groupby(['text_id', 'cluster_label']).size().unstack(fill_value=0)
    valid_tids = [tid for tid in matrix.index if get_group(tid) is not None]
    X = matrix.loc[valid_tids]
    y = np.array([get_group(tid) for tid in valid_tids])
    
    print(f"Analyzing {len(X)} docs across {len(X.columns)} features.")

    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (Linear)": make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
    }

    for clf_name, clf in classifiers.items():
        loo = LeaveOneOut()
        y_true, y_pred, results = [], [], []
        
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tid = valid_tids[test_index[0]]
            
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]
            
            y_pred.append(pred)
            y_true.append(y_test[0])
            results.append({"Document": tid, "True": y_test[0], "Predicted": pred, "Match": "✓" if pred == y_test[0] else "✗"})
            
        print(f"\n--- {clf_name} Accuracy: {accuracy_score(y_true, y_pred)*100:.1f}% ---")
        print(pd.DataFrame(results).to_string(index=False))

    # Feature Importance (Random Forest only)
    rf = classifiers["Random Forest"]
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTOP DIAGNOSTIC CONCEPTS (Random Forest):")
    print(importances.head(10).to_string())

if __name__ == "__main__":
    analyze_predictability()
