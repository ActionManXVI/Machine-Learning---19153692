import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from DataPreprocessing import DataPreprocessor

# Set the working directory
os.chdir(os.path.dirname(__file__))

def predict_country(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_country = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_country.fit(X_train, y_train)
    
    y_pred = rf_country.predict(X_test)
    
    return y_test, y_pred

if __name__ == "__main__":
    preprocessor = DataPreprocessor('hm_all_stores.csv')
    df = preprocessor.preprocess()
    X, y = preprocessor.get_country_prediction_data()
    
    y_test, y_pred = predict_country(X, y)
    
    print("Country Prediction Results:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Retrieve unique class names from y_test
    unique_labels = sorted(set(y_test))
    
    # Get country names that match the labels in y_test
    country_names = preprocessor.get_country_names()
    relevant_names = [name for i, name in enumerate(country_names) if i in unique_labels]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=relevant_names))
