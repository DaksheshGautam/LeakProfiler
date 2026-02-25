__version__ = "0.2"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def run_leakguard(file_path, target_column):        #Main function to run the LeakGuard tool.
    
    report = {}
    
    # 1. Load Data
    df = pd.read_csv(file_path)
    report['dataset_shape'] = df.shape
    
    # 2. Separate Target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 3. Run Detectors
    report['identifier_risk'] = detect_identifiers(X)
    report['duplicates'] = detect_duplicates(df)
    report['high_correlation'] = detect_high_correlation(X, y)
    report['high_importance'] = detect_feature_importance_leakage(X, y)
    report['temporal_leakage'] = detect_temporal_leakage(df, target_column)
    
    # 4. Generate Report
    print_report(report)

def detect_identifiers(df, threshold=0.95):         #Detects columns that are likely identifiers based on uniqueness.
    
    identifier_cols = []
    for col in df.columns:
        if df[col].nunique() / len(df) > threshold:
            identifier_cols.append(col)
    return identifier_cols

def detect_duplicates(df):                  #Detects duplicate rows in the dataframe.
    return df.duplicated().sum()

def detect_high_correlation(X, y, threshold=0.8):           #Detects features with high correlation to the target.

    y_numeric = y                               # Ensure y is numeric for correlation calculation
    if y.dtype == 'object':
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)

    
    df_corr = X.copy()                          # Combine features and target for correlation matrix  
    
    for col in df_corr.columns:                 # Preprocess data: handle categorical features and NaNs
        if df_corr[col].dtype == 'object':
            df_corr[col] = df_corr[col].astype('category').cat.codes
        if df_corr[col].isnull().any():
            df_corr[col] = df_corr[col].fillna(df_corr[col].median())

    df_corr['target'] = y_numeric

    
    numeric_df_corr = df_corr.select_dtypes(include=np.number)          # Select only numeric columns for correlation calculation

    
    correlations = numeric_df_corr.corr()['target'].abs().sort_values(ascending=False)      # Calculate correlations


    
    high_corr_features = correlations[correlations > threshold]          # Filter high correlations (excluding target itself)
    high_corr_features = high_corr_features.drop('target', errors='ignore')

    return high_corr_features.index.tolist()


def detect_feature_importance_leakage(X, y, threshold=0.30):            #Detects leakage using feature importance from a RandomForest model.
    
    
    X_processed = X.copy()                      # Preprocess data: handle categorical features and NaNs
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            X_processed[col] = X_processed[col].astype('category').cat.codes
        
        if X_processed[col].isnull().any():                 # Handle missing values
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    
    y_processed = y                             
    if y_processed.dtype == 'object':
        le = LabelEncoder()                             # Handle categorical target
        y_processed = le.fit_transform(y_processed)
            
    # Train a lightweight RandomForest model
    # Use a small portion of data to keep it fast
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    
    importances = pd.Series(model.feature_importances_, index=X_processed.columns)          # Get feature importances
    
    high_importance_features = importances[importances > threshold].index.tolist()          # Filter high importance features
    
    return high_importance_features

def detect_temporal_leakage(df, target_col, threshold=0.4):
    
    #Detects potential temporal leakage risks by checking multiple signals:
    #1. Target Autocorrelation (when sorted by date)
    #2. Regular Time Spacing (e.g., hourly, daily)
    #3. Timestamp Uniqueness

    temporal_warnings = []
    
    # Identify potential datetime columns
    # 1. Existing datetime dtypes
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    
    # 2. Object columns that look like dates (heuristic based on name)
    candidate_cols = [c for c in df.select_dtypes(include=['object']).columns 
                      if any(x in c.lower() for x in ['date', 'time', 'year', 'month', 'day'])]
    
    for col in candidate_cols:
        try:
            # Check first few non-null values to see if they parse
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Use coerce to handle potential mixed formats or noise
                parsed = pd.to_datetime(sample, errors='coerce')
                # If more than 50% parse successfully, treat as date
                if parsed.notna().sum() > len(sample) * 0.5:
                    date_cols.append(col)
        except:
            pass
            
    # Check Autocorrelation
    for col in set(date_cols):
        # Create a temporary dataframe to sort without affecting original
        temp_df = df[[col, target_col]].copy()
        
        # Drop NaNs
        temp_df = temp_df.dropna()

        # Parse date column to ensure it's datetime
        try:
            temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
            temp_df = temp_df.dropna(subset=[col])
        except:
            continue

        if len(temp_df) < 10: 
            continue

        # Ensure target is numeric for correlation
        if temp_df[target_col].dtype == 'object':
            try:
                temp_df[target_col] = pd.to_numeric(temp_df[target_col])
            except:
                le = LabelEncoder()
                temp_df[target_col] = le.fit_transform(temp_df[target_col].astype(str))
        
        # Sort by date
        temp_df = temp_df.sort_values(by=col)
        
        # --- Signal 1: Autocorrelation ---
        autocorr = temp_df[target_col].autocorr(lag=1)
        if pd.isna(autocorr):
            autocorr = 0.0
            
        # --- Signal 2: Regular Spacing ---
        is_regular = False
        time_diffs = temp_df[col].diff().dropna()
        if len(time_diffs) > 0:
            # Check if the most frequent time delta constitutes > 80% of the data
            mode_freq = time_diffs.value_counts(normalize=True).iloc[0]
            if mode_freq > 0.8:
                is_regular = True

        # --- Signal 3: Uniqueness ---
        n_unique = temp_df[col].nunique()
        uniqueness_ratio = n_unique / len(temp_df)
        is_unique = uniqueness_ratio > 0.95

        # --- Decision Logic ---
        detected_signals = []
        
        if abs(autocorr) > threshold:
            detected_signals.append(f"High Target Autocorrelation ({autocorr:.2f})")
        elif abs(autocorr) > 0.1:
            detected_signals.append(f"Moderate Target Autocorrelation ({autocorr:.2f})")
            
        if is_regular:
            detected_signals.append("Regular Time Spacing")
            
        if is_unique:
            detected_signals.append("High Timestamp Uniqueness")
            
        # Flag if we have strong autocorrelation OR multiple temporal signals
        if (abs(autocorr) > threshold) or (len(detected_signals) >= 2):
            temporal_warnings.append(
                f"Temporal Leakage Risk in '{col}': {', '.join(detected_signals)}. "
                "Data appears to be a time-series; use TimeSeriesSplit."
            )
    print(col, autocorr, uniqueness_ratio, mode_freq)
    return temporal_warnings


def print_report(report):               #Prints the final LeakGuard report.
   
    print("LeakGuard Report v{__version__}: ")
    print(f"Dataset shape: {report['dataset_shape']}")
    
    if report['identifier_risk']:
        print("Identifier Risk:")
        print(report['identifier_risk'])
        
    if report['duplicates'] > 0:
        print(f"Duplicates:{report['duplicates']}")
        
    if report['high_correlation']:
        print("High Correlation with Target:")
        print(report['high_correlation'])
        
    if report['high_importance']:
        print("High Importance Features (Potential Leakage):")
        print(report['high_importance'])
        
    if report.get('temporal_leakage'):
        print("Temporal Leakage Risks:")
        for warning in report['temporal_leakage']:
            print(f"- {warning}")
        
    print("End of Report.")