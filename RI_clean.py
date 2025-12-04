import pandas as pd
import numpy as np

def assign_stage(dia_inches):
    """Assign growth stage based on DBH"""
    if pd.isna(dia_inches):
        return None
    
    dia_cm = dia_inches * 2.54
    
    if dia_cm < 5:
        return 1
    elif dia_cm <= 10:
        return 2
    elif dia_cm <= 17.5:
        return 3
    elif dia_cm <= 27.5:
        return 4
    elif dia_cm <= 42.5:
        return 5
    else:
        return 6

def process_ri_data(raw_tree_file, output_file):
    """
    Complete pipeline for RI: clean raw data and create combined file
    """
    
    print("="*60)
    print("Processing Rhode Island Tree Data")
    print("="*60)
    
    # Step 1: Read and filter
    print(f"\nReading data from {raw_tree_file}...")
    df = pd.read_csv(raw_tree_file)
    df.columns = df.columns.str.strip()
    
    initial_count = len(df)
    print(f"Initial rows: {initial_count}")
    
    # Filter by year and species
    df = df[df['INVYR'] == 2024]
    print(f"After INVYR = 2024 filter: {len(df)} rows")
    
    df = df[df['SPCD'] == 261]
    print(f"After SPCD = 261 filter: {len(df)} rows")
    
    if len(df) == 0:
        print("\n⚠ WARNING: No data matches filters!")
        return None
    
    # Step 2: Add STAGE column
    print("\nAdding STAGE column based on DIA...")
    df['STAGE'] = df['DIA'].apply(assign_stage)
    
    # Step 3: Add TREECOUNT column
    df['TREECOUNT'] = 1.0
    
    # Step 4: Select and order columns
    columns_order = ['CN', 'PLOT', 'SUBP', 'TREE', 'STATECD', 'INVYR', 'SPCD', 
                     'DIA', 'HT', 'ACTUALHT', 'TREECOUNT', 'STAGE']
    
    df = df[columns_order]
    df = df.sort_values(['PLOT', 'SUBP']).reset_index(drop=True)
    
    # Step 5: Display results
    print(f"\nStage distribution:")
    for stage in sorted(df['STAGE'].dropna().unique()):
        count = int(df[df['STAGE'] == stage]['TREECOUNT'].sum())
        dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][int(stage)-1]
        print(f"  Stage {int(stage)} ({dbh_range} cm): {count} individuals")
    
    print(f"\nTotal: {int(df['TREECOUNT'].sum())} individuals")
    
    # Step 6: Save
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("✓ Processing completed successfully!")
    
    return df


if __name__ == "__main__":
    # Process RI data (no seedlings)
    df = process_ri_data(
        raw_tree_file="./Tree_Data/RI_TREE.csv",  # Your raw input file
        output_file="./Filter_Data/RI_COMBINED.csv"  # Final output
    )
    
    if df is not None:
        print("\nFirst 10 rows:")
        print(df.head(10))