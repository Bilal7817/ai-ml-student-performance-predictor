import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Global variables
df = None
model = None
label_encoders = {}
model_type = None  # Track if current model is classifier or regressor

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
    
    # Encode categorical features
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
                    # Handle unseen categories safely
                    encoded_values = []
                    for val in X_encoded[col]:
                        if val in le.classes_:
                            encoded_values.append(le.transform([val])[0])
                        else:
                            # Use the encoding for 'unknown' if available, otherwise use 0
                            if 'unknown' in le.classes_:
                                encoded_values.append(le.transform(['unknown'])[0])
                            else:
                                encoded_values.append(0)
                    X_encoded[col] = encoded_values
    
    # Handle missing values in numeric columns
    X_encoded = X_encoded.fillna(0)
    
    # Encode target if categorical and during training
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
        # Reset encoders for new training
        label_encoders = {}
        
        # Parse input
        features = [f.strip() for f in features_text.split(',')]
        target = target_text.strip()
        
        # Check columns exist
        missing_cols = [f for f in features if f not in df.columns]
        if missing_cols:
            messagebox.showerror("Error", f"Missing columns: {missing_cols}")
            return
        
        if target not in df.columns:
            messagebox.showerror("Error", f"Target '{target}' not found!")
            return
        
        # Prepare data
        X = df[features]
        y = df[target]
        
        # Remove rows with missing target values
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Determine if classification or regression
        is_classification = y.dtype == 'object' or len(y.unique()) < 10
        model_type = 'classification' if is_classification else 'regression'
        
        # Encode categorical data
        X_encoded, y_encoded = encode_categorical_data(X, y, is_training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
        
        # Select and train model based on algorithm choice and problem type
        if is_classification:
            if algorithm == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif algorithm == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            messagebox.showinfo("Success", f"{algorithm} Classification trained!\nAccuracy: {accuracy:.3f}")
            
        else:  # Regression
            if algorithm == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            elif algorithm == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            elif algorithm == "Linear Regression":
                model = LinearRegression()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            messagebox.showinfo("Success", f"{algorithm} Regression trained!\nR² Score: {r2:.3f}\nRMSE: {rmse:.2f}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

def make_predictions():
    """Make predictions with trained model"""
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
        
        # Encode data using existing encoders
        X_encoded, _ = encode_categorical_data(X, is_training=False)
        
        # Make predictions
        predictions = model.predict(X_encoded)
        
        # Decode predictions if target was categorical
        if '_target_' in label_encoders:
            target_le = label_encoders['_target_']
            predictions_decoded = target_le.inverse_transform(predictions.astype(int))
        else:
            predictions_decoded = predictions
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions for {len(predictions)} samples:\n")
        result_text.insert(tk.END, f"Model Type: {model_type.title()}\n")
        result_text.insert(tk.END, f"Algorithm: {algorithm_var.get()}\n\n")
        
        # Show first 15 predictions
        for i in range(min(15, len(predictions))):
            if model_type == 'classification':
                result_text.insert(tk.END, f"Sample {i+1}: {predictions_decoded[i]}\n")
            else:
                result_text.insert(tk.END, f"Sample {i+1}: {predictions[i]:.2f}\n")
        
        if len(predictions) > 15:
            result_text.insert(tk.END, f"... and {len(predictions) - 15} more\n")
        
        # Add summary statistics
        result_text.insert(tk.END, f"\nSummary:\n")
        if model_type == 'classification':
            unique_preds, counts = np.unique(predictions_decoded, return_counts=True)
            for pred, count in zip(unique_preds, counts):
                percentage = (count / len(predictions)) * 100
                result_text.insert(tk.END, f"{pred}: {count} ({percentage:.1f}%)\n")
        else:
            result_text.insert(tk.END, f"Mean: {np.mean(predictions):.2f}\n")
            result_text.insert(tk.END, f"Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}\n")
            
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# GUI Setup
root = tk.Tk()
root.title("ML Student Performance Predictor")
root.geometry("700x750")

# Load dataset
tk.Button(root, text="Load Dataset", command=load_dataset, font=("Arial", 12)).pack(pady=10)

# Algorithm selection
tk.Label(root, text="Select Algorithm:", font=("Arial", 10)).pack()
algorithm_var = tk.StringVar(value="Random Forest")
algorithm_combo = ttk.Combobox(root, textvariable=algorithm_var, 
                              values=["Random Forest", "Decision Tree", "Linear Regression", "Logistic Regression"],
                              state="readonly", width=25)
algorithm_combo.pack(pady=5)

# Features input
tk.Label(root, text="Features (comma-separated):", font=("Arial", 10)).pack()
features_entry = tk.Entry(root, width=60)
features_entry.pack(pady=5)
tk.Label(root, text="Example: age,gender,study_hours_per_day", font=("Arial", 8), fg="gray").pack()

# Target input
tk.Label(root, text="Target:", font=("Arial", 10)).pack()
target_entry = tk.Entry(root, width=60)
target_entry.pack(pady=5)
tk.Label(root, text="Example: exam_score", font=("Arial", 8), fg="gray").pack()

# Action buttons
tk.Button(root, text="Train Model", command=train_model, font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="Make Predictions", command=make_predictions, font=("Arial", 12)).pack(pady=5)

# Results display
tk.Label(root, text="Results:", font=("Arial", 10)).pack()
result_text = tk.Text(root, height=22, width=80)
result_text.pack(pady=10)

root.mainloop()