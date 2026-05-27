# preprocess_data.py
import pandas as pd
import numpy as np
import os

# Load original CSV
raw_path = "original.csv"

if not os.path.exists(raw_path):
    raise FileNotFoundError(f"{raw_path} not found!")

df = pd.read_csv(raw_path).replace('-', np.nan)

# Drop irrelevant columns
df = df.drop(columns=['district_rank', 'island_rank'], errors='ignore')

# --- Stream imputation ---
df_stream_not_null = df[df['stream'].notna()].copy()
subject_to_stream_map = {}
subject_columns = ['sub1', 'sub2', 'sub3']

all_subjects = pd.concat([df_stream_not_null[col].dropna() for col in subject_columns]).unique()

for subject in all_subjects:
    relevant_rows = df_stream_not_null[
        (df_stream_not_null['sub1'] == subject) |
        (df_stream_not_null['sub2'] == subject) |
        (df_stream_not_null['sub3'] == subject)
    ]
    if not relevant_rows.empty:
        stream_modes = relevant_rows['stream'].dropna().mode()
        if not stream_modes.empty:
            subject_to_stream_map[subject] = stream_modes[0]

def impute_stream(row):
    if pd.isna(row['stream']):
        for sub_col in ['sub1', 'sub2', 'sub3']:
            subject = row[sub_col]
            if pd.notna(subject) and subject in subject_to_stream_map:
                return subject_to_stream_map[subject]
    return row['stream']

df['stream'] = df.apply(impute_stream, axis=1)

# --- Cleaning and calculated columns ---
df['al_year'] = pd.to_numeric(df['al_year'], errors='coerce')
df['birth_year'] = pd.to_numeric(df['birth_year'], errors='coerce')
df['age'] = df['al_year'] - df['birth_year']
df['Zscore'] = pd.to_numeric(df['Zscore'], errors='coerce')
df['cgt_r'] = pd.to_numeric(df['cgt_r'], errors='coerce')

# University eligibility
df['eligible_for_university_entrance'] = df.apply(
    lambda row: 'No' if 'F' in [str(row['sub1_r']), str(row['sub2_r']), str(row['sub3_r'])] else 'Yes',
    axis=1
)

# Clean strings
df['gender'] = df['gender'].astype(str).str.lower().str.strip()
df['stream'] = df['stream'].astype(str).str.upper().str.strip()

# Drop rows with missing critical info
df = df.dropna(subset=['stream', 'gender', 'sub1_r', 'sub2_r', 'sub3_r'])
df = df[df['stream'] != 'NON']
df = df[(df['age'] > 15) & (df['age'] < 30)]

# Save preprocessed CSV
df.to_csv("cleaned_al_data.csv", index=False)
print("✅ Preprocessed data saved as cleaned_al_data.csv")