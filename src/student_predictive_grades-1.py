import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Global variables
df = None
model = None
label_encoders = {}
model_type = None

def load_dataset():
    """Load CSV or Excel dataset"""
    global df
    file_path = filedialog.askopenfilename(
        title="Select Dataset File",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if file_path:
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            messagebox.showinfo("Success", f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

def encode_categorical_data(X, y=None, is_training=True):
    """Handle categorical data with label encoding"""
    global label_encoders
    X_encoded = X.copy()
    
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            if is_training:
                le = LabelEncoder()
                X_encoded[col] = X_encoded[col].fillna('unknown')
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    X_encoded[col] = X_encoded[col].fillna('unknown').astype(str)
                    encoded_values = []
                    for val in X_encoded[col]:
                        if val in le.classes_:
                            encoded_values.append(le.transform([val])[0])
                        else:
                            encoded_values.append(le.transform(['unknown'])[0] if 'unknown' in le.classes_ else 0)
                    X_encoded[col] = encoded_values
    
    X_encoded = X_encoded.fillna(0)
    
    if y is not None and y.dtype == 'object' and is_training:
        target_le = LabelEncoder()
        y_encoded = target_le.fit_transform(y.astype(str))
        label_encoders['_target_'] = target_le
        return X_encoded, y_encoded
    
    return X_encoded, y

def train_model():
    """Train selected machine learning model"""
    global df, model, label_encoders, model_type
    
    if df is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return
    
    features_text = features_entry.get().strip()
    target_text = target_entry.get().strip()
    algorithm = algorithm_var.get()
    
    if not features_text or not target_text:
        messagebox.showerror("Error", "Please specify features and target!")
        return
    
    try:
        label_encoders = {}
        features = [f.strip() for f in features_text.split(',')]
        target = target_text.strip()
        
        # Check columns exist
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols or target not in df.columns:
            messagebox.showerror("Error", f"Missing columns: {missing_cols + [target] if target not in df.columns else missing_cols}")
            return
        
        # Prepare data
        X, y = df[features], df[target]
        mask = y.notna()
        X, y = X[mask], y[mask]
        
        # Determine model type
        is_classification = y.dtype == 'object' or len(y.unique()) < 10
        model_type = 'classification' if is_classification else 'regression'
        
        # Encode and split data
        X_encoded, y_encoded = encode_categorical_data(X, y, is_training=True)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
        
        # Train model based on algorithm and type
        if is_classification:
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
            }
            model = models[algorithm]
            model.fit(X_train, y_train)
            
            accuracy = accuracy_score(y_test, model.predict(X_test))
            cv_scores = cross_val_score(model, X_encoded, y_encoded, cv=5, scoring='accuracy')
            message = f"{algorithm} Classification\nAccuracy: {accuracy:.3f}\nCV: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})"
        else:
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Linear Regression": LinearRegression()
            }
            model = models[algorithm]
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores = cross_val_score(model, X_encoded, y_encoded, cv=5, scoring='r2')
            message = f"{algorithm} Regression\nR²: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}\nCV R²: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})"
        
        messagebox.showinfo("Model Trained", message)
        
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

def make_predictions():
    """Make predictions and show results with feature importance"""
    global df, model, model_type
    
    if df is None or model is None:
        messagebox.showerror("Error", "Please load data and train model first!")
        return
    
    features_text = features_entry.get().strip()
    if not features_text:
        messagebox.showerror("Error", "Please specify features!")
        return
    
    try:
        features = [f.strip() for f in features_text.split(',')]
        X = df[features]
        X_encoded, _ = encode_categorical_data(X, is_training=False)
        predictions = model.predict(X_encoded)
        
        # Decode if categorical target
        if '_target_' in label_encoders:
            predictions = label_encoders['_target_'].inverse_transform(predictions.astype(int))
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"=== RESULTS ({algorithm_var.get()}) ===\n")
        result_text.insert(tk.END, f"Samples: {len(predictions)}, Type: {model_type.title()}\n\n")
        
        # Sample predictions
        result_text.insert(tk.END, "Sample Predictions:\n")
        for i in range(min(10, len(predictions))):
            if model_type == 'classification':
                result_text.insert(tk.END, f"  {i+1}: {predictions[i]}\n")
            else:
                result_text.insert(tk.END, f"  {i+1}: {predictions[i]:.2f}\n")
        if len(predictions) > 10:
            result_text.insert(tk.END, f"  ... and {len(predictions) - 10} more\n")
        
        # Statistics
        result_text.insert(tk.END, f"\nStatistics:\n")
        if model_type == 'classification':
            unique, counts = np.unique(predictions, return_counts=True)
            for pred, count in zip(unique, counts):
                result_text.insert(tk.END, f"  {pred}: {count} ({count/len(predictions)*100:.1f}%)\n")
        else:
            result_text.insert(tk.END, f"  Mean: {np.mean(predictions):.2f}\n")
            result_text.insert(tk.END, f"  Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}\n")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            result_text.insert(tk.END, f"\nFeature Importance:\n")
            importance_data = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
            for feature, importance in importance_data:
                result_text.insert(tk.END, f"  {feature}: {importance:.3f}\n")
            
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# GUI Setup
root = tk.Tk()
root.title("ML Student Performance Predictor")
root.geometry("700x750")

tk.Button(root, text="Load Dataset", command=load_dataset, font=("Arial", 12)).pack(pady=10)

tk.Label(root, text="Algorithm:").pack()
algorithm_var = tk.StringVar(value="Random Forest")
ttk.Combobox(root, textvariable=algorithm_var, 
             values=["Random Forest", "Decision Tree", "Linear Regression", "Logistic Regression"],
             state="readonly", width=25).pack(pady=5)

tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root, width=60)
features_entry.pack(pady=5)

tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root, width=60)
target_entry.pack(pady=5)

tk.Button(root, text="Train Model", command=train_model, font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="Make Predictions", command=make_predictions, font=("Arial", 12)).pack(pady=5)

result_text = tk.Text(root, height=22, width=80)
result_text.pack(pady=10)

root.mainloop()