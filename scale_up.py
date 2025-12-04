import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import os
from framework import HemlockPVA

def load_and_scale_population(csv_file: str, target_population: int = 1_000_000_000) -> Dict:
    """
    Load population from all-state CSV and scale proportionally to target
    
    Formula for each stage:
    scaled_pop[stage] = (original_pop[stage] / total_original) × target_population
    
    Parameters:
    csv_file: Path to combined all-state CSV file
    target_population: Target total population (default: 1 billion)
    
    Returns:
    Dictionary with original and scaled populations
    """
    
    print("="*70)
    print(f"SCALING POPULATION TO {target_population:,}")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # Count individuals in each stage (original)
    original_pop = np.zeros(6)
    
    for stage in range(1, 7):
        stage_df = df[df['STAGE'] == stage]
        count = stage_df['TREECOUNT'].sum()
        original_pop[stage-1] = count
    
    total_original = np.sum(original_pop)
    
    print(f"\nOriginal Population:")
    print("-"*70)
    for i, count in enumerate(original_pop):
        dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][i]
        proportion = count / total_original if total_original > 0 else 0
        print(f"Stage {i+1} ({dbh_range:>12} cm): {int(count):>8,} ({proportion:>6.2%})")
    print(f"{'Total':>30}: {int(total_original):>8,}")
    
    # Calculate scaling factor
    scaling_factor = target_population / total_original
    
    # Scale each stage using YOUR formula:
    # scaled_pop[stage] = (original_pop[stage] / total_original) × 1,000,000,000
    scaled_pop = np.zeros(6)
    
    print(f"\nScaling Calculation:")
    print("-"*70)
    print(f"Formula: (Stage Pop / Total Pop) × {target_population:,}")
    print(f"Scaling factor: {scaling_factor:,.2f}x")
    print("-"*70)
    
    for i in range(6):
        proportion = original_pop[i] / total_original
        scaled_pop[i] = proportion * target_population
        
        dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][i]
        print(f"Stage {i+1}: ({int(original_pop[i]):>8,} / {int(total_original):>8,}) × {target_population:,}")
        print(f"       = {proportion:.6f} × {target_population:,}")
        print(f"       = {int(scaled_pop[i]):>15,}\n")
    
    total_scaled = np.sum(scaled_pop)
    
    print(f"Scaled Population Summary:")
    print("-"*70)
    for i, count in enumerate(scaled_pop):
        dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][i]
        proportion = count / total_scaled if total_scaled > 0 else 0
        print(f"Stage {i+1} ({dbh_range:>12} cm): {int(count):>15,} ({proportion:>6.2%})")
    print(f"{'Total':>30}: {int(total_scaled):>15,}")
    
    return {
        'original_pop': original_pop,
        'scaled_pop': scaled_pop,
        'scaling_factor': scaling_factor,  # ADD THIS LINE
        'total_original': total_original,
        'total_scaled': total_scaled
    }

def run_scaled_hemlock_analysis(csv_file: str,
                                 target_population: int = 1_000_000_000,
                                 scenario_name: str = "Historical",
                                 n_years: int = 100,
                                 n_replications: int = 1000,
                                 output_folder: str = "Output_Scaled"):
    """
    Run PVA analysis with scaled population
    
    Parameters:
    csv_file: Path to all-state combined CSV file
    target_population: Target total population (default: 1 billion)
    scenario_name: Scenario name (Historical, Worst Case, Optimistic)
    n_years: Projection years
    n_replications: Number of Monte Carlo replications
    output_folder: Output folder for plots
    """
    
    # Step 1: Load and scale population
    scaling_results = load_and_scale_population(csv_file, target_population)
    scaled_pop = scaling_results['scaled_pop']
    
    # Step 2: Initialize PVA model
    stages = ['<5cm', '5-10cm', '10.1-17.5cm', '17.6-27.5cm', '27.6-42.5cm', '>42.5cm']
    pva = HemlockPVA(stages, scaled_pop)
    
    # Step 3: Set parameters based on scenario
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name.upper()}")
    print(f"{'='*70}")
    
    if scenario_name.lower() == "historical":
        P = np.array([0.903, 0.961, 0.965, 0.976, 0.963, 0.990])
        G = np.array([0.004, 0.012, 0.017, 0.012, 0.018])
        F = np.array([0, 0.299, 0.774, 1.957, 6.025])
    elif scenario_name.lower() == "worst case":
        P = np.array([0.406, 0.432, 0.434, 0.303, 0.299, 0.307])
        G = np.array([0.002, 0.006, 0.008, 0.004, 0.006])
        F = np.array([0, 0.135, 0.240, 0.607, 1.868])
    elif scenario_name.lower() == "optimistic":
        P = np.array([0.95, 0.97, 0.98, 0.98, 0.97, 0.99])
        G = np.array([0.005, 0.015, 0.020, 0.015, 0.020])
        F = np.array([0, 0.5, 1.0, 2.0, 8.0])
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    # Step 4: Create Lefkovitch matrix
    base_matrix = pva.create_lefkovitch_matrix(P, G, F)
    
    print("\nParameters:")
    print("-"*70)
    print(f"P (Survival):    {P}")
    print(f"G (Transition):  {G}")
    print(f"F (Fecundity):   {F}")
    
    print("\nLefkovitch Matrix:")
    print("-"*70)
    print(base_matrix)
    
    # Calculate lambda
    eigenvalues = np.linalg.eigvals(base_matrix)
    lambda_det = np.max(eigenvalues.real)
    print(f"\nDeterministic growth rate (λ): {lambda_det:.4f}")
    if lambda_det < 1:
        print("⚠ Population is declining (λ < 1)")
    elif lambda_det > 1:
        print("✓ Population is growing (λ > 1)")
    else:
        print("→ Population is stable (λ = 1)")
    
    # Step 5: Run simulation
    print(f"\nRunning simulation ({n_years} years, {n_replications} replications)...")
    print("-"*70)
    
    results = pva.run_simulation(
        base_matrix=base_matrix,
        n_years=n_years,
        n_replications=n_replications,
        demographic_stoch=True,
        environmental_stoch=True,
        cv=0.15
    )
    
    # Step 6: Calculate statistics
    extinction_risk = pva.calculate_extinction_risk(results)
    final_pops = [p for p in results['final_populations'] if p > 0]
    mean_final_pop = np.mean(final_pops) if final_pops else 0
    median_final_pop = np.median(final_pops) if final_pops else 0
    
    print(f"\nResults ({n_years}-year projection):")
    print("-"*70)
    print(f"Extinction risk:        {extinction_risk:>10.2%}")
    print(f"Mean final population:  {mean_final_pop:>15,.0f}")
    print(f"Median final population:{median_final_pop:>15,.0f}")
    print(f"Initial population:     {int(np.sum(scaled_pop)):>15,}")
    
    # Step 7: Plot results
    title = f"All States (Scaled to {target_population:,.0f}) - {scenario_name}"
    
    pva.plot_results(
        results,
        scenario_name=scenario_name,
        state_name=title,
        output_folder=output_folder
    )
    
    # Step 8: Save detailed results
    save_results_summary(scaling_results, results, lambda_det, extinction_risk,
                        mean_final_pop, median_final_pop, scenario_name,
                        n_years, output_folder)
    
    return pva, results, base_matrix, scaling_results


def save_results_summary(scaling_results, simulation_results, lambda_val,
                        extinction_risk, mean_final, median_final,
                        scenario_name, n_years, output_folder):
    """Save detailed results to text file"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    filename = f"summary_{scenario_name.replace(' ', '_')}.txt"
    output_file = os.path.join(output_folder, filename)
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"EASTERN HEMLOCK PVA - {scenario_name.upper()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("POPULATION SCALING\n")
        f.write("-"*70 + "\n")
        f.write(f"Original total:  {int(scaling_results['total_original']):>15,}\n")
        f.write(f"Scaled total:    {int(scaling_results['total_scaled']):>15,}\n")
        f.write(f"Scaling factor:  {scaling_results['scaling_factor']:>15,.2f}x\n\n")
        
        f.write("STAGE DISTRIBUTION (Scaled)\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Stage':<10} {'DBH Range':<15} {'Count':>15} {'Proportion':>12}\n")
        f.write("-"*70 + "\n")
        
        total = scaling_results['total_scaled']
        for i, count in enumerate(scaling_results['scaled_pop']):
            stage_num = i + 1
            dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][i]
            proportion = count / total if total > 0 else 0
            f.write(f"Stage {stage_num:<4} {dbh_range:<15} {int(count):>15,} {proportion:>11.2%}\n")
        
        f.write("-"*70 + "\n")
        f.write(f"{'Total':<10} {'':<15} {int(total):>15,} {1.0:>11.2%}\n\n")
        
        f.write(f"SIMULATION RESULTS ({n_years}-year projection)\n")
        f.write("-"*70 + "\n")
        f.write(f"Deterministic growth rate (λ): {lambda_val:.4f}\n")
        f.write(f"Extinction risk:               {extinction_risk:.2%}\n")
        f.write(f"Mean final population:         {mean_final:,.0f}\n")
        f.write(f"Median final population:       {median_final:,.0f}\n")
        
        # Add interpretation
        f.write("\n" + "="*70 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*70 + "\n")
        if lambda_val < 1:
            f.write("⚠ Population is DECLINING (λ < 1)\n")
        elif lambda_val > 1:
            f.write("✓ Population is GROWING (λ > 1)\n")
        else:
            f.write("→ Population is STABLE (λ = 1)\n")
        
        if extinction_risk > 0.5:
            f.write(f"⚠ HIGH extinction risk ({extinction_risk:.1%})\n")
        elif extinction_risk > 0.1:
            f.write(f"⚠ MODERATE extinction risk ({extinction_risk:.1%})\n")
        else:
            f.write(f"✓ LOW extinction risk ({extinction_risk:.1%})\n")
    
    print(f"\n✔ Summary saved to: {output_file}")


# Main execution
if __name__ == "__main__":
    
    # Path to your all-state combined file
    csv_file = "./Filter_Data/All_State.csv"
    
    # Target population (1 billion)
    target_pop = 1_000_000_000
    
    # Run multiple scenarios
    scenarios = ["Historical", "Worst Case", "Optimistic"]
    
    all_results = {}
    
    for scenario in scenarios:
        print("\n\n" + "="*70)
        print(f"RUNNING SCENARIO: {scenario.upper()}")
        print("="*70 + "\n")
        
        output_folder = f"Output_Scaled_{scenario.replace(' ', '_')}"
        
        pva, results, matrix, scaling = run_scaled_hemlock_analysis(
            csv_file=csv_file,
            target_population=target_pop,
            scenario_name=scenario,
            n_years=100,
            n_replications=1000,
            output_folder=output_folder
        )
        
        all_results[scenario] = {
            'pva': pva,
            'results': results,
            'matrix': matrix,
            'scaling': scaling
        }
        
        print(f"\n✔ {scenario} scenario completed!")
        print(f"✔ Results saved to: {output_folder}/")
    
    # Print comparison summary
    print("\n\n" + "="*70)
    print("COMPARISON ACROSS SCENARIOS")
    print("="*70)
    print(f"{'Scenario':<20} {'Lambda (λ)':<12} {'Extinction Risk':<18} {'Mean Final Pop':<20}")
    print("-"*70)
    
    for scenario, data in all_results.items():
        results = data['results']
        matrix = data['matrix']
        
        lambda_val = np.max(np.linalg.eigvals(matrix).real)
        extinction_risk = data['pva'].calculate_extinction_risk(results)
        final_pops = [p for p in results['final_populations'] if p > 0]
        mean_final = np.mean(final_pops) if final_pops else 0
        
        print(f"{scenario:<20} {lambda_val:<12.4f} {extinction_risk:<18.2%} {mean_final:<20,.0f}")
    
    print("\n✔ All scenarios completed!")
