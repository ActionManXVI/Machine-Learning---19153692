# store_classification_enhanced.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from DataPreprocessing import DataPreprocessor

# Set Working Directory
os.chdir(os.path.dirname(__file__))

def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def get_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return dict(zip(np.unique(y), class_weights))

def create_ensemble(X_train, y_train, class_weights):
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
    
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble

def visualize_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.mean([member.feature_importances_ for member in model.estimators_ if hasattr(member, 'feature_importances_')], axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Store Classification')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preprocessor = DataPreprocessor('hm_all_stores.csv')
    df = preprocessor.preprocess()
    X, y = preprocessor.get_store_classification_data()
    
    # Handle class imbalance
    X_resampled, y_resampled = balance_dataset(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get class weights
    class_weights = get_class_weights(y_train)
    
    # Create and train ensemble
    ensemble = create_ensemble(X_train_scaled, y_train, class_weights)
    
    # Make predictions
    y_pred = ensemble.predict(X_test_scaled)
    
    # Print classification report
    print("Ensemble Model - Store Classification Results:")
    print(classification_report(y_test, y_pred, target_names=preprocessor.get_store_class_names()))
    
    # Visualize feature importance
    visualize_feature_importance(ensemble, X.columns)

    # Individual model performance (for comparison)
    for name, model in ensemble.named_estimators_.items():
        y_pred_individual = model.predict(X_test_scaled)
        print(f"\n{name.upper()} - Store Classification Results:")
        print(classification_report(y_test, y_pred_individual, target_names=preprocessor.get_store_class_names()))