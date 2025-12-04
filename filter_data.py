import pandas as pd

def assign_stage(dia):
    """
    Assign growth stage based on DBH (DIA) in cm
    Stage 1: <5
    Stage 2: 5-10
    Stage 3: 10.1-17.5
    Stage 4: 17.6-27.5
    Stage 5: 27.6-42.5
    Stage 6: >42.5
    """
    if pd.isna(dia):
        return None
    elif dia < 5/2.205:
        return 1
    elif dia <= 10/2.205:
        return 2
    elif dia <= 17.5/2.205:
        return 3
    elif dia <= 27.5/2.205:
        return 4
    elif dia <= 42.5/2.205:
        return 5
    else:
        return 6

def clean_forest_data(input_file, output_file):
    """
    Filter forest inventory data to keep only rows that meet criteria:
    - INVYR = 2024
    - SPCD = 261
    Then classify trees into stages 1-6 based on DBH (DIA)
    Keep only essential columns
    """
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original number of rows: {len(df)}")
    
    # Apply filters
    print("\nApplying filters...")
    
    # Count rows before filtering
    initial_count = len(df)
    
    # Filter 1: Keep only INVYR = 2024
    df = df[df['INVYR'] == 2024]
    print(f"  After INVYR = 2024 filter: {len(df)} rows (removed {initial_count - len(df)})")
    
    # Filter 2: Keep only SPCD = 216
    count_before = len(df)
    df = df[df['SPCD'] == 261]
    print(f"  After SPCD = 261 filter: {len(df)} rows (removed {count_before - len(df)})")
    
    print(f"\nTotal rows removed: {initial_count - len(df)}")
    print(f"Final number of rows: {len(df)}")
    
    if len(df) == 0:
        print("\n⚠ WARNING: No rows match the filter criteria!")
        print("The output file will be empty.")
    else:
        # Add stage classification based on DIA (DBH)
        df['STAGE'] = df['DIA'].apply(assign_stage)
        
        # Keep only specified columns
        columns_to_keep = ['CN', 'PLOT', 'SUBP', 'TREE', 'STATECD', 'INVYR', 'SPCD', 'DIA', 'HT', 'ACTUALHT', 'STAGE']
        df = df[columns_to_keep]
        
        print(f"\nKept columns: {', '.join(columns_to_keep)}")
        print(f"Final data shape: {df.shape}")
        
        print(f"\nStage classification breakdown:")
        stage_counts = df['STAGE'].value_counts().sort_index()
        for stage, count in stage_counts.items():
            if pd.notna(stage):
                print(f"  Stage {int(stage)}: {count} trees")
        
        if df['STAGE'].isna().sum() > 0:
            print(f"  No stage (missing DIA): {df['STAGE'].isna().sum()} trees")
        
        print(f"\nDIA (DBH) statistics:")
        print(df['DIA'].describe())
    
    # Save cleaned data
    print(f"\nSaving filtered data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("✓ Data filtering completed successfully!")
    return df

# Main execution
if __name__ == "__main__":
    # Specify your input and output file names
    input_file = "./Tree_Data/VT_TREE.csv"  # Change this to your input file name
    output_file = "./Filter_Data/VT_TREE_FILTER.csv"  # Change this to your desired output file name
    
    # Run the cleaning function
    filtered_df = clean_forest_data(input_file, output_file)
    
    # Optional: Display first few rows of filtered data
    if len(filtered_df) > 0:
        print("\nFirst few rows of filtered data:")
        print(filtered_df.head(10))