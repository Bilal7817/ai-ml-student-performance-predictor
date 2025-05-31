import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Global variables to store dataset and model
df = None
model = None
label_encoders = {}

def load_dataset():
    """Load CSV or Excel dataset through file dialog"""
    global df
    # Mac-compatible file dialog
    file_path = filedialog.askopenfilename(
        title="Select Dataset File",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Excel files", "*.xls"), ("All files", "*.*")]
    )
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            
            # Debug: Print available columns
            print("Available columns:", df.columns.tolist())
            print("First few rows:")
            print(df.head())
            
            messagebox.showinfo("Success", f"Dataset loaded successfully!\nRows: {len(df)}, Columns: {len(df.columns)}")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    return None

def train_model():
    """Train machine learning model with user-specified features and target"""
    global df, model
    
    # Check if dataset is loaded
    if df is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return None
    
    # Get features and target from user input
    features_text = features_entry.get().strip()
    target_text = target_entry.get().strip()
    
    if not features_text or not target_text:
        messagebox.showerror("Error", "Please specify both features and target!")
        return None
    
    try:
        # Parse features (comma-separated)
        features = [f.strip() for f in features_text.split(',')]
        target = target_text.strip()
        
        # Check if columns exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            messagebox.showerror("Error", f"Features not found in dataset: {missing_features}")
            return None
        
        if target not in df.columns:
            messagebox.showerror("Error", f"Target '{target}' not found in dataset!")
            return None
        
        # Prepare data
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        messagebox.showinfo("Model Trained", f"Model trained successfully!\nAccuracy: {accuracy:.2f}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")
    return None

def make_predictions():
    """Make predictions using the trained model"""
    global df, model
    
    # Check if dataset and model exist
    if df is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return
    
    if model is None:
        messagebox.showerror("Error", "Please train a model first!")
        return
    
    # Get features from user input
    features_text = features_entry.get().strip()
    if not features_text:
        messagebox.showerror("Error", "Please specify features!")
        return
    
    try:
        # Parse features
        features = [f.strip() for f in features_text.split(',')]
        
        # Make predictions on the entire dataset
        X_new = df[features]
        predictions = model.predict(X_new)
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions for {len(predictions)} samples:\n\n")
        
        # Show first 20 predictions
        for i in range(min(20, len(predictions))):
            result_text.insert(tk.END, f"Sample {i+1}: {predictions[i]}\n")
        
        if len(predictions) > 20:
            result_text.insert(tk.END, f"\n... and {len(predictions) - 20} more predictions")
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# Create main GUI window
root = tk.Tk()
root.title("Student Predictive Grades")
root.geometry("600x700")

# Load dataset button
load_button = tk.Button(root, text="Load Dataset", command=load_dataset, font=("Arial", 12))
load_button.pack(pady=10)

# Features input
tk.Label(root, text="Features (comma-separated):", font=("Arial", 10)).pack()
features_entry = tk.Entry(root, width=50)
features_entry.pack(pady=5)

# Target input
tk.Label(root, text="Target:", font=("Arial", 10)).pack()
target_entry = tk.Entry(root, width=50)
target_entry.pack(pady=5)

# Train model button
train_button = tk.Button(root, text="Train Model", command=train_model, font=("Arial", 12))
train_button.pack(pady=10)

# Make predictions button
predict_button = tk.Button(root, text="Make Predictions", command=make_predictions, font=("Arial", 12))
predict_button.pack(pady=10)

# Results display
tk.Label(root, text="Prediction Results:", font=("Arial", 10)).pack()
result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

# Start GUI
root.mainloop()