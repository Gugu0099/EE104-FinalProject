import pandas as pd

def clean_seedling_data(input_file, output_file):
    """
    Filter forest seedling data to keep only rows that meet criteria:
    - INVYR = 2024
    - SPCD = 261
    Mark all seedlings as STAGE 1
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
    
    # Filter 2: Keep only SPCD = 261
    count_before = len(df)
    df = df[df['SPCD'] == 261]
    print(f"  After SPCD = 261 filter: {len(df)} rows (removed {count_before - len(df)})")
    
    print(f"\nTotal rows removed: {initial_count - len(df)}")
    print(f"Final number of rows: {len(df)}")
    
    if len(df) == 0:
        print("\n⚠ WARNING: No rows match the filter criteria!")
        print("The output file will be empty.")
    else:
        # Mark all seedlings as STAGE 1
        df['STAGE'] = 1
        
        # Keep only specified columns (seedlings don't have TREE, DIA, HT, ACTUALHT)
        columns_to_keep = ['CN', 'PLOT', 'SUBP', 'STATECD', 'INVYR', 'SPCD', 'TREECOUNT', 'STAGE']
        df = df[columns_to_keep]
        
        print(f"\nKept columns: {', '.join(columns_to_keep)}")
        print(f"Final data shape: {df.shape}")
        
        print(f"\nAll seedlings marked as STAGE 1")
        print(f"Total seedling count (sum of TREECOUNT): {df['TREECOUNT'].sum()}")
        
        print(f"\nTREECOUNT statistics:")
        print(df['TREECOUNT'].describe())
    
    # Save cleaned data
    print(f"\nSaving filtered data to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("✓ Data filtering completed successfully!")
    return df

# Main execution
if __name__ == "__main__":
    # Specify your input and output file names
    input_file = "./Tree_Data/RI_SEEDLING.csv"
    output_file = "./Filter_Data/RI_SEEDLING_FILTER.csv"
    
    # Run the cleaning function
    filtered_df = clean_seedling_data(input_file, output_file)
    
    # Optional: Display first few rows of filtered data
    if len(filtered_df) > 0:
        print("\nFirst few rows of filtered data:")
        print(filtered_df.head(10))