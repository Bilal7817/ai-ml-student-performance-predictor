import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Global variables
df = None
model = None
label_encoders = {}

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
                    # Handle unseen categories
                    for i, val in enumerate(X_encoded[col]):
                        if val in le.classes_:
                            X_encoded.iloc[i, X_encoded.columns.get_loc(col)] = le.transform([val])[0]
                        else:
                            X_encoded.iloc[i, X_encoded.columns.get_loc(col)] = le.transform(['unknown'])[0]
    
    # Handle missing values
    X_encoded = X_encoded.fillna(0)
    
    # Encode target if categorical
    if y is not None and y.dtype == 'object':
        if is_training:
            target_le = LabelEncoder()
            y_encoded = target_le.fit_transform(y.astype(str))
            label_encoders['_target_'] = target_le
            return X_encoded, y_encoded
        else:
            if '_target_' in label_encoders:
                return X_encoded, label_encoders['_target_'].transform(y.astype(str))
    
    return X_encoded, y

def train_model():
    """Train RandomForest model"""
    global df, model, label_encoders
    
    if df is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return
    
    features_text = features_entry.get().strip()
    target_text = target_entry.get().strip()
    
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
        
        print(f"Removed {(~mask).sum()} rows with missing target values")
        print(f"Training with {len(X)} samples")
        
        # Encode categorical data
        X_encoded, y_encoded = encode_categorical_data(X, y, is_training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
        
        # Train model - choose classifier or regressor based on target type
        if y.dtype == 'object' or len(y.unique()) < 10:  # Categorical or few unique values
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            messagebox.showinfo("Success", f"Classification Model trained!\nAccuracy: {accuracy:.3f}")
        else:  # Continuous values - use regression
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            messagebox.showinfo("Success", f"Regression Model trained!\nR² Score: {r2:.3f}\nRMSE: {rmse:.2f}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

def make_predictions():
    """Make predictions with trained model"""
    global df, model
    
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
        
        # Decode if target was categorical
        if '_target_' in label_encoders:
            predictions = label_encoders['_target_'].inverse_transform(predictions)
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions for {len(predictions)} samples:\n\n")
        
        # Show first 15 predictions
        for i in range(min(15, len(predictions))):
            result_text.insert(tk.END, f"Sample {i+1}: {predictions[i]}\n")
        
        if len(predictions) > 15:
            result_text.insert(tk.END, f"... and {len(predictions) - 15} more")
            
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# GUI Setup
root = tk.Tk()
root.title("Student Performance Predictor")
root.geometry("600x650")

# Widgets
tk.Button(root, text="Load Dataset", command=load_dataset, font=("Arial", 12)).pack(pady=10)

tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root, width=50)
features_entry.pack(pady=5)

tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root, width=50)
target_entry.pack(pady=5)

tk.Button(root, text="Train Model", command=train_model, font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="Make Predictions", command=make_predictions, font=("Arial", 12)).pack(pady=5)

result_text = tk.Text(root, height=20, width=70)
result_text.pack(pady=10)

root.mainloop()