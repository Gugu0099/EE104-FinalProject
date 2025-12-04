import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
import os

class HemlockPVA:
    """Population Viability Analysis for Eastern Hemlock - 6 Stage Model"""
    
    def __init__(self, stages: List[str], initial_population: np.ndarray):
        """
        Initialize PVA model
        
        Parameters:
        stages: List of life stages
        initial_population: Initial population vector for each stage
        """
        self.stages = stages
        self.n_stages = len(stages)
        self.initial_population = initial_population
        
    def create_lefkovitch_matrix(self, 
                                  P: np.ndarray,
                                  G: np.ndarray,
                                  F: np.ndarray) -> np.ndarray:
        """
        Create a Lefkovitch matrix for 6-stage model
        
        Parameters:
        P: Survival probabilities (staying in same stage) [P1, P2, P3, P4, P5, P6]
        G: Transition probabilities (moving to next stage) [G1, G2, G3, G4, G5]
        F: Fecundities (reproduction) [F2, F3, F4, F5, F6] - note: no F1
        
        Returns:
        Lefkovitch matrix
        """
        L = np.zeros((self.n_stages, self.n_stages))
        
        # Diagonal: Survival in same stage (P)
        np.fill_diagonal(L, P)
        
        # Sub-diagonal: Transitions to next stage (G)
        for i in range(len(G)):
            L[i+1, i] = G[i]
            
        # First row (columns 1-5): Fecundities (F)
        # F starts from stage 2 (index 1), so F[0] goes to L[0,1]
        L[0, 1:] = F
        
        return L
    
    def add_demographic_stochasticity(self, 
                                       population: np.ndarray,
                                       matrix: np.ndarray) -> np.ndarray:
        """
        Add demographic stochasticity using binomial distributions
        """
        new_pop = np.zeros_like(population)
        
        for i in range(self.n_stages):
            # Expected individuals from each stage
            expected = matrix[i, :] @ population
            
            if expected > 0:
                # Use Poisson for reproduction, Binomial for survival
                if i == 0:  # Stage 1 (from reproduction)
                    new_pop[i] = np.random.poisson(expected)
                else:  # Survival/transition
                    # For transitions and survival
                    total_expected = 0
                    # Survival in current stage
                    if i < len(population):
                        n_survive = int(population[i])
                        if n_survive > 0 and matrix[i, i] > 0:
                            new_pop[i] += np.random.binomial(n_survive, matrix[i, i])
                    
                    # Transition from previous stage
                    if i > 0:
                        n_transition = int(population[i-1])
                        if n_transition > 0 and matrix[i, i-1] > 0:
                            new_pop[i] += np.random.binomial(n_transition, matrix[i, i-1])
        
        return new_pop
    
    def add_environmental_stochasticity(self, 
                                         base_matrix: np.ndarray,
                                         cv: float = 0.1) -> np.ndarray:
        """
        Add environmental stochasticity to matrix elements
        
        Parameters:
        base_matrix: Mean transition matrix
        cv: Coefficient of variation for environmental stochasticity
        
        Returns:
        Stochastic matrix for this time step
        """
        stochastic_matrix = np.copy(base_matrix)
        
        # Add variation to non-zero elements
        for i in range(self.n_stages):
            for j in range(self.n_stages):
                if base_matrix[i, j] > 0:
                    mean_val = base_matrix[i, j]
                    
                    if 0 < mean_val < 1:  # For probabilities (P and G)
                        # Use beta distribution for probabilities
                        var = (cv * mean_val) ** 2
                        var = min(var, mean_val * (1 - mean_val) * 0.99)  # Ensure valid variance
                        
                        alpha = mean_val * ((mean_val * (1 - mean_val) / var) - 1)
                        beta_param = (1 - mean_val) * ((mean_val * (1 - mean_val) / var) - 1)
                        
                        if alpha > 0 and beta_param > 0:
                            stochastic_matrix[i, j] = np.random.beta(alpha, beta_param)
                    else:  # For fecundities (F)
                        # Use lognormal for fecundities
                        std = cv * mean_val
                        if std > 0:
                            stochastic_matrix[i, j] = np.random.lognormal(
                                np.log(mean_val) - 0.5 * np.log(1 + (std/mean_val)**2),
                                np.sqrt(np.log(1 + (std/mean_val)**2))
                            )
        
        return stochastic_matrix
    
    def run_simulation(self,
                       base_matrix: np.ndarray,
                       n_years: int = 100,
                       n_replications: int = 1000,
                       demographic_stoch: bool = True,
                       environmental_stoch: bool = True,
                       cv: float = 0.1) -> Dict:
        """
        Run population projection with stochasticity
        
        Returns:
        Dictionary with simulation results
        """
        results = {
            'trajectories': np.zeros((n_replications, n_years, self.n_stages)),
            'total_pop': np.zeros((n_replications, n_years)),
            'extinction_times': [],
            'final_populations': []
        }
        
        for rep in range(n_replications):
            population = np.copy(self.initial_population).astype(float)
            
            for year in range(n_years):
                # Add environmental stochasticity
                if environmental_stoch:
                    matrix = self.add_environmental_stochasticity(base_matrix, cv)
                else:
                    matrix = base_matrix
                
                # Project population
                if demographic_stoch:
                    population = self.add_demographic_stochasticity(population, matrix)
                else:
                    population = matrix @ population
                
                # Store results
                results['trajectories'][rep, year, :] = population
                results['total_pop'][rep, year] = np.sum(population)
                
                # Check for extinction
                if np.sum(population) < 1:
                    results['extinction_times'].append(year)
                    break
            
            results['final_populations'].append(np.sum(population))
        
        return results
    
    def calculate_extinction_risk(self, results: Dict, threshold: float = 1.0) -> float:
        """
        Calculate probability of extinction
        """
        n_extinct = sum(1 for pop in results['final_populations'] if pop < threshold)
        return n_extinct / len(results['final_populations'])
    def plot_results(self, results: Dict, scenario_name: str = "", state_name: str = "", output_folder: str = "plots"):
        """
        Visualize simulation results and save each plot as a separate image.
    
        Parameters:
        results: Simulation results dictionary
        scenario_name: Name of the scenario
        state_name: State name to include in titles
        output_folder: Folder to save plots
        """
    
        # ---- Create output folder if needed ----
        os.makedirs(output_folder, exist_ok=True)
    
        years = np.arange(results['total_pop'].shape[1])
    
        # ---------------- Plot 1: Population Trajectories ----------------
        fig1, ax1 = plt.subplots(figsize=(10, 6))
    
        mean_trajectory = np.mean(results['total_pop'], axis=0)
        percentiles = np.percentile(results['total_pop'], [5, 25, 75, 95], axis=0)
    
        ax1.plot(years, mean_trajectory, 'b-', label='Mean', linewidth=2)
        ax1.fill_between(years, percentiles[0], percentiles[3], alpha=0.2, color='blue', label='5–95%')
        ax1.fill_between(years, percentiles[1], percentiles[2], alpha=0.4, color='blue', label='25–75%')
        ax1.set_xlabel('Years', fontsize=12)
        ax1.set_ylabel('Total Population', fontsize=12)
    
        # Add state name to title
        title1 = f'{state_name} - Population Trajectory - B2 x F2' if state_name else 'Population Trajectory'
        ax1.set_title(title1, fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
        plt.tight_layout()
        filename1 = f"{state_name}_population_trajectory.png" if state_name else "population_trajectory.png"
        save_path1 = os.path.join(output_folder, filename1)
        fig1.savefig(save_path1, dpi=300, bbox_inches="tight")
        print(f"✔ Saved: {save_path1}")
        plt.close(fig1)
    
        # ---------------- Plot 2: Stage Distribution ----------------
        fig2, ax2 = plt.subplots(figsize=(10, 6))
    
        mean_stages = np.mean(results['trajectories'], axis=0)
    
        for i, stage in enumerate(self.stages):
            ax2.plot(years, mean_stages[:, i], label=f'Stage {i+1}: {stage}', linewidth=1.5)
    
        ax2.set_xlabel('Years', fontsize=12)
        ax2.set_ylabel('Mean Population', fontsize=12)
    
        title2 = f'{state_name} - Stage Distribution Over Time  - B2 x F2' if state_name else 'Stage Distribution Over Time'
        ax2.set_title(title2, fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        filename2 = f"{state_name}_stage_distribution.png" if state_name else "stage_distribution.png"
        save_path2 = os.path.join(output_folder, filename2)
        fig2.savefig(save_path2, dpi=300, bbox_inches="tight")
        print(f"✔ Saved: {save_path2}")
        plt.close(fig2)
    
        # ---------------- Plot 3: Extinction Risk ----------------
        fig3, ax3 = plt.subplots(figsize=(10, 6))
    
        extinction_curve = []
        for year in range(results['total_pop'].shape[1]):
            extinct_at_year = np.sum(results['total_pop'][:, year] < 1) / results['total_pop'].shape[0]
            extinction_curve.append(extinct_at_year)
    
        ax3.plot(extinction_curve, 'r-', linewidth=2)
        ax3.set_xlabel('Years', fontsize=12)
        ax3.set_ylabel('Cumulative Extinction Probability', fontsize=12)
    
        title3 = f'{state_name} - Extinction Risk Over Time - B2 x F2' if state_name else 'Extinction Risk Over Time'
        ax3.set_title(title3, fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
        plt.tight_layout()
        filename3 = f"{state_name}_extinction_risk.png" if state_name else "extinction_risk.png"
        save_path3 = os.path.join(output_folder, filename3)
        fig3.savefig(save_path3, dpi=300, bbox_inches="tight")
        print(f"✔ Saved: {save_path3}")
        plt.close(fig3)
    
        # ---------------- Plot 4: Distribution of Final Populations ----------------
        fig4, ax4 = plt.subplots(figsize=(10, 6))
    
        final_pops = [p for p in results['final_populations'] if p > 0]
        if final_pops:
            ax4.hist(final_pops, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
        ax4.set_xlabel('Final Population Size', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
    
        title4 = f'{state_name} - Final Population Distribution (Non-extinct) - B2 x F2' if state_name else 'Final Population Distribution (Non-extinct)'
        ax4.set_title(title4, fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
        plt.tight_layout()
        filename4 = f"{state_name}_final_population_dist.png" if state_name else "final_population_dist.png"
        save_path4 = os.path.join(output_folder, filename4)
        fig4.savefig(save_path4, dpi=300, bbox_inches="tight")
        print(f"✔ Saved: {save_path4}")
        plt.close(fig4)
    
        print(f"\n✔ All 4 plots saved to: {output_folder}/\n")

def load_initial_population_from_csv(csv_file: str) -> np.ndarray:
    """
    Load initial population from CSV file and count individuals in each stage
    Properly handles TREECOUNT for seedlings
    
    Parameters:
    csv_file: Path to CSV file with STAGE and TREECOUNT columns
    
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
    
    print("\nInitial Population from CSV:")
    print("-" * 40)
    for i, count in enumerate(initial_pop):
        dbh_range = ['<5', '5-10', '10.1-17.5', '17.6-27.5', '27.6-42.5', '>42.5'][i]
        print(f"Stage {i+1} (DBH {dbh_range} cm): {int(count)} individuals")
    print(f"Total: {int(np.sum(initial_pop))} individuals")
    
    return initial_pop


def run_hemlock_analysis(csv_file: str):
    """
    Run hemlock PVA analysis with parameters from your table
    """
    
    # Define 6 stages based on DBH
    stages = ['<5cm', '5-10cm', '10.1-17.5cm', '17.6-27.5cm', '27.6-42.5cm', '>42.5cm']
    
    # Load initial population from CSV
    initial_pop = load_initial_population_from_csv(csv_file)
    
    # Initialize model
    pva = HemlockPVA(stages, initial_pop)
    
    # Parameters from your table
    # B1: 75%
    # P = np.array([0.531, 0.564, 0.567, 0.471, 0.465, 0.478])  # Survival in same stage (6 values)
    # G = np.array([0.003, 0.010, 0.014, 0.009, 0.014])
    # F = np.array([0, 0.082, 0.208, 0.543, 1.550])
    # B2: 50%
    # P = np.array([0.655, 0.697, 0.700, 0.639, 0.631, 0.649])  # Survival in same stage (6 values)
    # G = np.array([0.003, 0.011, 0.015, 0.010, 0.015])
    # F = np.array([0, 0.155, 0.396, 1.015, 3.042])
    # B2 x F2
    P = np.array([0.655, 0.697, 0.700, 0.639, 0.631, 0.649])
    G = np.array([0.003, 0.011, 0.015, 0.010, 0.015])
    F = np.array([0, 0.464, 1.189, 3.044, 9.125])
    # F2: Tripled
    # P = np.array([0.406, 0.432, 0.434, 0.303, 0.299, 0.307])  # Survival in same stage (6 values)
    # G = np.array([0.002, 0.006, 0.008, 0.004, 0.006])
    # F = np.array([0, 0.404, 0.720, 1.820, 5.603])
    # F3: Tripled
    # P = np.array([0.406, 0.432, 0.434, 0.303, 0.299, 0.307])  # Survival in same stage (6 values)
    # G = np.array([0.002, 0.006, 0.008, 0.004, 0.006])
    # F = np.array([0, 0.538, 0.960, 2.427, 7.471])
    # B1 x F2
    # P = np.array([0.531, 0.564, 0.567, 0.471, 0.465, 0.478])
    # G = np.array([0.003, 0.010, 0.014, 0.009, 0.014])
    # F = np.array([0, 0.247, 0.623, 1.629, 4.650])
    # B2 x F1
    # P = np.array([0.655, 0.697, 0.700, 0.639, 0.631, 0.649])
    # G = np.array([0.003, 0.011, 0.015, 0.010, 0.015])
    # F = np.array([0, 0.309, 0.793, 2.029, 6.084])
    # Biological Control + Outplanting
    # P = np.array([6.55**-1, 6.97**-1, 7.00**-1, 6.39**-1, 6.31**-1, 6.49**-1])  # Survival in same stage (6 values)
    # G = np.array([2.76**-3, 8.88**-3, 1.26**-1, 7.89**-1, 1.20**-1])
    # F = np.array([0, 0.538, 0.960, 2.427, 7.471])
    # Outplating
    # P = np.array([0.406, 0.432, 0.434, 0.303, 0.299, 0.307])  # Survival in same stage (6 values)
    # G = np.array([0.002, 0.006, 0.008, 0.004, 0.006])  # Transition to next stage (5 values)
    # F = np.array([0, 0.538, 0.960, 2.427, 7.471])
    # Biological Control
    # P = np.array([6.55**-1, 6.97**-1, 7.00**-1, 6.39**-1, 6.31**-1, 6.49**-1])  # Survival in same stage (6 values)
    # G = np.array([2.76**-3, 8.88**-3, 1.26**-1, 7.89**-1, 1.20**-1])  # Transition to next stage (5 values)
    # F = np.array([0, 0.135, 0.240, 0.607, 1.868])
    # Historical case
    # P = np.array([0.903, 0.961, 0.965, 0.976, 0.963, 0.990])  # Survival in same stage (6 values)
    # G = np.array([0.004, 0.012, 0.017, 0.012, 0.018])  # Transition to next stage (5 values)
    # F = np.array([0, 0.299, 0.774, 1.957, 6.025])  # Fecundity for stages 2-6 (5 values) - FIXED!
    # wrost case
    # P = np.array([0.406, 0.432, 0.434, 0.303, 0.299, 0.307])  # Survival in same stage (6 values)
    # G = np.array([0.002, 0.006, 0.008, 0.004, 0.006])  # Transition to next stage (5 values)
    # F = np.array([0, 0.135, 0.240, 0.607, 1.868])
    # Create Lefkovitch matrix
    base_matrix = pva.create_lefkovitch_matrix(P, G, F)
    
    print("\nLefkovitch Matrix:")
    print("-" * 80)
    print(base_matrix)
    
    # Calculate deterministic growth rate (lambda)
    eigenvalues = np.linalg.eigvals(base_matrix)
    lambda_det = np.max(eigenvalues.real)
    print(f"\nDeterministic growth rate (λ): {lambda_det:.4f}")
    if lambda_det < 1:
        print("⚠ Population is declining (λ < 1)")
    elif lambda_det > 1:
        print("✓ Population is growing (λ > 1)")
    else:
        print("→ Population is stable (λ = 1)")
    
    # Run simulation
    print("\nRunning simulation...")
    print("-" * 40)
    
    results = pva.run_simulation(
        base_matrix=base_matrix,
        n_years = 100,
        n_replications=1000,
        demographic_stoch=True,
        environmental_stoch=True,
        cv=0.15  # 15% coefficient of variation
    )
    
    # Calculate statistics
    extinction_risk = pva.calculate_extinction_risk(results)
    final_pops = [p for p in results['final_populations'] if p > 0]
    mean_final_pop = np.mean(final_pops) if final_pops else 0
    median_final_pop = np.median(final_pops) if final_pops else 0
    
    print(f"\nResults (100-year projection):")
    print("-" * 40)
    print(f"Extinction risk: {extinction_risk:.2%}")
    print(f"Mean final population: {mean_final_pop:.0f}")
    print(f"Median final population: {median_final_pop:.0f}")
    print(f"Initial population: {int(np.sum(initial_pop))}")
    
    # Plot results
    # pva.plot_results(results, "Eastern Hemlock - Current Parameters")
    pva.plot_results(
    results,
    scenario_name="Eastern Hemlock - Current Parameters",
    state_name="ALL STATE",          # <-- Big title across 4 plots
    output_folder="B2 x F2"  # <-- Folder created automatically
    )
    
    return pva, results, base_matrix


# Run the analysis
if __name__ == "__main__":
    # Update this path to your CSV file
    csv_file = "./Filter_Data/All_State.csv"  # Fixed typo: Filter_DATA -> Filter_Data
    
    pva, results, matrix = run_hemlock_analysis(csv_file)