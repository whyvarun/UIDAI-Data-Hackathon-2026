import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.preprocessing import MinMaxScaler

DATA_FOLDER_PATH = r"C:/Users/VD/Downloads/UIDAI/data"

def load_and_clean_data(path):
    print("\n[STEP 1] Loading and Merging Data...")
    
    files_map = {
        'enrol': 'api_data_aadhar_enrolment_*.csv',
        'demo': 'api_data_aadhar_demographic_*.csv',
        'bio': 'api_data_aadhar_biometric_*.csv'
    }
    
    dfs = {}
    for label, pattern in files_map.items():
        full_pattern = os.path.join(path, pattern)
        files = glob.glob(full_pattern)
        
        if not files:
            print(f"   [WARNING] No files found for {label}")
            dfs[label] = pd.DataFrame()
            continue
            
        df_list = [pd.read_csv(f) for f in files]
        combined = pd.concat(df_list, ignore_index=True)
        
        combined.columns = combined.columns.str.strip().str.lower()
        dfs[label] = combined
        print(f"   [OK] Loaded {label.upper()}: {combined.shape[0]} rows")

    for key in dfs:
        if not dfs[key].empty:
            dfs[key]['date'] = pd.to_datetime(dfs[key]['date'], format='%d-%m-%Y', errors='coerce')
            for col in ['state', 'district']:
                if col in dfs[key].columns:
                    dfs[key][col] = dfs[key][col].astype(str).str.strip().str.upper()

    print("   [INFO] Performing Time-Series Merge...")
    merge_cols = ['date', 'state', 'district', 'pincode']
    
    master = dfs['enrol']
    if not dfs['demo'].empty:
        master = pd.merge(master, dfs['demo'], on=merge_cols, how='outer', suffixes=('_enrol', '_demo'))
    if not dfs['bio'].empty:
        master = pd.merge(master, dfs['bio'], on=merge_cols, how='outer', suffixes=('', '_bio'))
    
    master = master.fillna(0)
    print(f"   [SUCCESS] Master Dataset Ready: {master.shape}")
    return master

def create_strategic_features(df):
    print("\n[STEP 2] Engineering Strategic Indicators...")
    
    df['Migration_Intensity'] = df['demo_age_17_'] / (df['age_18_greater'] + 1)
    
    df['Child_Compliance_Score'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
    
    df['Total_Load'] = (df['age_0_5'] + df['age_5_17'] + df['age_18_greater'] + 
                        df['demo_age_17_'] + df['bio_age_17_'])
    
    return df

def generate_hackathon_visuals(df):
    print("\n[STEP 3] Generating Visual Insights...")
    sns.set_theme(style="whitegrid", context="talk")
    
    state_stats = df.groupby('state')[['Migration_Intensity', 'Child_Compliance_Score', 'Total_Load']].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total_Load'], bins=50, kde=True, color='teal')
    plt.title('Univariate: Distribution of Daily Transaction Volume')
    plt.xlabel('Transactions per Pincode per Day')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('1_Univariate_Distribution.png')
    print("   [SAVED] 1_Univariate_Distribution.png")

    top_mig = state_stats.sort_values('Migration_Intensity', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_mig, x='Migration_Intensity', y='state', palette='viridis')
    plt.title('Bivariate: Top "Migration Magnet" States\n(High Updates Relative to New Enrolments)')
    plt.xlabel('Migration Intensity Score')
    plt.tight_layout()
    plt.savefig('2_Bivariate_Migration.png')
    print("   [SAVED] 2_Bivariate_Migration.png")

    corr_matrix = df[['age_18_greater', 'demo_age_17_', 'bio_age_17_', 'Migration_Intensity']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Trivariate: Correlation Matrix of Aadhaar Events')
    plt.tight_layout()
    plt.savefig('3_Trivariate_Correlation.png')
    print("   [SAVED] 3_Trivariate_Correlation.png")

if __name__ == "__main__":
    if os.path.exists(DATA_FOLDER_PATH):
        master_df = load_and_clean_data(DATA_FOLDER_PATH)
        
        if not master_df.empty:
            master_df = create_strategic_features(master_df)
            
            generate_hackathon_visuals(master_df)
            
            print("\n[STEP 4] Saving Impact Report...")
            report = master_df.groupby('state')[['Migration_Intensity', 'Child_Compliance_Score']].mean()
            report.to_csv('Final_Impact_Report.csv')
            print("   [SUCCESS] Analysis Complete. Files saved in project folder.")
        else:
            print("[ERROR] Dataframes are empty.")
    else:
        print(f"[ERROR] Path does not exist: {DATA_FOLDER_PATH}")