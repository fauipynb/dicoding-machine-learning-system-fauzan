import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def prepare_data():
    # 1. Load Data
    raw_path = 'data/raw/mobile_price.csv'
    if not os.path.exists(raw_path):
        # Fallback for CI if data not yet downloaded (though it should be in repo)
        url = "https://raw.githubusercontent.com/arpita-maji/Mobile-Price-Classification/master/mobile_price_range_data.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(raw_path)

    # 2. Split
    X = df.drop('price_range', axis=1)
    y = df['price_range']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Save
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('src', exist_ok=True) # For scaler
    
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    joblib.dump(scaler, 'src/scaler.joblib')
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
