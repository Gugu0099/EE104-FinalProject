import pandas as pd
import numpy as np
import os

def combine_seedling_and_tree_data(seedling_file, tree_file, output_file):
    """
    Combine seedling and tree data into one unified file.
    If seedling file doesn't exist, just process tree file alone.
    
    Parameters:
    seedling_file: Path to seedling CSV file (can be None or non-existent)
    tree_file: Path to tree CSV file
    output_file: Path for combined output CSV file
    """
    
    # Check if seedling file exists
    has_seedlings = seedling_file and os.path.exists(seedling_file)
    
    if has_seedlings:
        print("Reading seedling data...")
        seedling_df = pd.read_csv(seedling_file)
        seedling_df.columns = seedling_df.columns.str.strip()
        print(f"  Seedlings: {len(seedling_df)} rows")
    else:
        print("⚠ No seedling file found - processing tree data only")
        seedling_df = None
    
    print("Reading tree data...")
    tree_df = pd.read_csv(tree_file)
    tree_df.columns = tree_df.columns.str.strip()
    print(f"  Trees: {len(tree_df)} rows")
    
    # Add TREECOUNT to tree data (each tree = 1 individual)
    tree_df['TREECOUNT'] = 1.0
    
    # Define column order
    columns_order = ['CN', 'PLOT', 'SUBP', 'TREE', 'STATECD', 'INVYR', 'SPCD', 
                     'DIA', 'HT', 'ACTUALHT', 'TREECOUNT', 'STAGE']
    
    if has_seedlings:
        # Add missing columns to seedling data
        seedling_df['TREE'] = np.nan
        seedling_df['DIA'] = np.nan
        seedling_df['HT'] = np.nan
        seedling_df['ACTUALHT'] = np.nan
        
        # Ensure columns are in the same order
        seedling_df = seedling_df[columns_order]
        tree_df = tree_df[columns_order]
        
        # Combine the dataframes
        combined_df = pd.concat([seedling_df, tree_df], ignore_index=True)
    else:
        # Just use tree data
        tree_df = tree_df[columns_order]
        combined_df = tree_df.copy()
    
    # Sort by PLOT and SUBP for organization
    combined_df = combined_df.sort_values(['PLOT', 'SUBP']).reset_index(drop=True)
    
    print(f"\nCombined data:")
    print(f"  Total rows: {len(combined_df)}")
    print(f"\nStage distribution:")
    
    for stage in sorted(combined_df['STAGE'].dropna().unique()):
        stage_data = combined_df[combined_df['STAGE'] == stage]
        
        # Count total individuals (sum TREECOUNT)
        total_individuals = stage_data['TREECOUNT'].sum()
        records = len(stage_data)
        
        print(f"  Stage {int(stage)}: {int(total_individuals)} individuals ({records} records)")
    
    # Save combined data
    print(f"\nSaving combined data to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print("✓ Data combination completed successfully!")
    
    return combined_df


def load_combined_population(csv_file):
    """
    Load population from combined CSV file and count individuals in each stage
    Properly handles TREECOUNT for seedlings
    
    Parameters:
    csv_file: Path to combined CSV file
    
    Returns:
    Array with count of individuals in each stage [stage1, stage2, ..., stage6]
    """
    df = pd.read_csv(csv_file)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Count individuals in each stage
    initial_pop = np.zeros(6)
    
    for stage in range(1, 7):
        stage_df = df[df['STAGE'] == stage]
        
        # Sum TREECOUNT for all records in this stage
        count = stage_df['TREECOUNT'].sum()
        initial_pop[stage-1] = count
    
    print("\nInitial Population from Combined Data:")
    print("-" * 50)
    for i, count in enumerate(initial_pop):
        dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][i]
        print(f"Stage {i+1} (DBH {dbh_range} cm): {int(count)} individuals")
    print(f"Total: {int(np.sum(initial_pop))} individuals")
    print("-" * 50)
    
    return initial_pop


# Main execution
if __name__ == "__main__": 
    # Example 2: State WITHOUT seedlings (e.g., RI)
    print("\n\n" + "="*60)
    print("Processing Rhode Island (no seedlings)")
    print("="*60)
    seedling_file = "./Filter_Data/_SEEDLING_FILTER.csv"  # Doesn't exist
    tree_file = "./Filter_Data/RI_TREE_FILTER.csv"
    output_file = "./Filter_Data/RI_COMBINED.csv"
    
    combined_df = combine_seedling_and_tree_data(seedling_file, tree_file, output_file)
    
    # Display sample
    print("\nFirst 10 rows of RI combined data:")
    print(combined_df.head(10))