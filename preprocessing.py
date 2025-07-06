import pandas as pd
from sklearn.preprocessing import RobustScaler

# Save column names from training to reuse later
MODEL_FEATURES_PATH = 'model/features_used.csv'

def preprocess_input(df):
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag',
                'land', 'logged_in', 'is_guest_login', 'level', 'outcome']

    # Step 1: Drop non-numeric columns and scale numeric data
    df_num = df.drop(cat_cols, axis=1, errors='ignore')
    num_cols = df_num.columns
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=num_cols)

    # Step 2: Drop old and merge scaled back
    df = df.drop(num_cols, axis=1)
    df[num_cols] = df_scaled

    # Step 3: Label encode outcome column if present
    if 'outcome' in df.columns:
        df['outcome'] = df['outcome'].apply(lambda x: 0 if x == 'normal' else 1)

    # Step 4: One-hot encode categorical
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], drop_first=False)

    # Step 5: Align with model features
    features_used = pd.read_csv(MODEL_FEATURES_PATH)
    expected_cols = features_used['feature'].tolist()

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0  # add missing

    df = df[expected_cols]  # reorder
    return df
