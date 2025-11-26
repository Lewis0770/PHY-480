import numpy as np
import itertools
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import time

class SpinGlassGNN:
        """ Made explicitly with the help of Claude, did the bulk of the work """

    def __init__(self, n_vertices: int = 16, n_layers: int = 10, alpha: float = 0.5):
        self.N = n_vertices
        self.m = n_layers
        self.alpha = alpha
        
        self.weights = []
        for l in range(self.m):
            w = np.random.uniform(-0.5, 0.5, (self.N, self.N))
            self.weights.append(w)
    
    def compute_hamiltonian(self, spins: np.ndarray, J: np.ndarray, h: np.ndarray = None) -> float:
        if h is None:
            h = np.zeros(self.N)
        
        energy = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                energy += J[i, j] * spins[i] * spins[j]
            energy += h[i] * spins[i]
        
        return energy
    
    def find_exact_ground_state(self, J: np.ndarray, h: np.ndarray = None) -> Tuple[np.ndarray, float]:
        if h is None:
            h = np.zeros(self.N)
        
        min_energy = float('inf')
        ground_state = None
        
        for config in itertools.product([-1, 1], repeat=self.N):
            spins = np.array(config)
            energy = self.compute_hamiltonian(spins, J, h)
            
            if energy < min_energy:
                min_energy = energy
                ground_state = spins.copy()
        
        return ground_state, min_energy
    
    def forward_pass(self, x_input: np.ndarray, J: np.ndarray, store_activations: bool = False) -> np.ndarray:
        x = x_input.copy().astype(float)
        
        if store_activations:
            self.activations = [x.copy()]
        
        for l in range(self.m):
            x_new = np.zeros(self.N)
            
            for i in range(self.N):
                graph_term = 0.0
                for j in range(self.N):
                    if j != i:
                        graph_term += J[i, j] * x[j]
                graph_term /= np.sqrt(self.N)
                
                dl_term = 0.0
                for j in range(self.N):
                    dl_term += self.weights[l][i, j] * x[j]
                dl_term /= np.sqrt(self.N)
                
                x_new[i] = self.alpha * np.tanh(self.alpha * graph_term + 
                                                (1 - self.alpha) * dl_term)
            
            x = x_new
            
            if store_activations:
                self.activations.append(x.copy())
        
        x_out = np.sign(x)
        x_out[x_out == 0] = 1
        
        return x_out
    
    def compute_loss_single(self, J: np.ndarray, x_exact: np.ndarray) -> float:
        x_input = np.random.choice([-1, 1], size=self.N)
        x_out = self.forward_pass(x_input, J)
        loss = np.sum((x_out - x_exact) ** 2) / self.N
        
        return loss
    
    def compute_loss(self, training_data: List[Tuple[np.ndarray, np.ndarray, float]]) -> float:
        total_loss = 0.0
        
        for J, x_exact, _ in training_data:
            total_loss += self.compute_loss_single(J, x_exact)
        
        loss = total_loss / len(training_data)
        return loss
    
    def compute_gradient_layer_sample(self, J: np.ndarray, x_exact: np.ndarray, 
                                     layer_idx: int, epsilon: float = 1e-4) -> np.ndarray:
        gradient = np.zeros((self.N, self.N))
        x_input = np.random.choice([-1, 1], size=self.N)
        
        x_out_base = self.forward_pass(x_input, J)
        loss_base = np.sum((x_out_base - x_exact) ** 2) / self.N
        
        for i in range(self.N):
            for j in range(self.N):
                original_weight = self.weights[layer_idx][i, j]
                
                self.weights[layer_idx][i, j] = original_weight + epsilon
                x_out_plus = self.forward_pass(x_input, J)
                loss_plus = np.sum((x_out_plus - x_exact) ** 2) / self.N
                
                self.weights[layer_idx][i, j] = original_weight
                
                gradient[i, j] = (loss_plus - loss_base) / epsilon
        
        return gradient
    
    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray, float]], 
              learning_rate: float = 0.01, n_epochs: int = 50, 
              batch_size: int = 10, verbose: bool = True) -> List[float]:
        loss_history = []
        n_samples = len(training_data)
        
        print(f"Training GNN with {n_samples} samples...")
        print(f"Network: {self.N} vertices, {self.m} layers, α={self.alpha}")
        print(f"Learning rate: {learning_rate}, Epochs: {n_epochs}, Batch size: {batch_size}")
        print("-" * 70)
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            loss = self.compute_loss(training_data)
            loss_history.append(loss)
            
            batch_indices = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
            batch_data = [training_data[i] for i in batch_indices]
            
            for layer_idx in range(self.m):
                avg_gradient = np.zeros((self.N, self.N))
                
                for J, x_exact, _ in batch_data:
                    gradient = self.compute_gradient_layer_sample(J, x_exact, layer_idx)
                    avg_gradient += gradient
                
                avg_gradient /= len(batch_data)
                self.weights[layer_idx] -= learning_rate * avg_gradient
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"Epoch {epoch+1:3d}/{n_epochs}: Loss = {loss:.6f}  (time: {epoch_time:.1f}s)")
        
        print("-" * 70)
        print(f"Training complete! Final loss: {loss_history[-1]:.6f}")
        
        return loss_history
    
    def test(self, test_data: List[Tuple[np.ndarray, np.ndarray, float]], 
             n_attempts: int = 10) -> Dict:
        results = {
            'overlap': [],
            'energy_diff': [],
            'exact_energies': [],
            'predicted_energies': [],
            'predicted_configs': [],
            'success': []
        }
        
        print(f"\nTesting on {len(test_data)} cases...")
        print("-" * 70)
        
        for idx, (J, x_exact, E_exact) in enumerate(test_data):
            best_config = None
            best_energy = float('inf')
            
            for attempt in range(n_attempts):
                x_input = np.random.choice([-1, 1], size=self.N)
                x_pred = self.forward_pass(x_input, J)
                E_pred = self.compute_hamiltonian(x_pred, J)
                
                if E_pred < best_energy:
                    best_energy = E_pred
                    best_config = x_pred
            
            overlap = np.abs(np.dot(best_config, x_exact)) / self.N
            energy_diff = best_energy - E_exact
            success = np.abs(energy_diff) < 1e-6
            
            results['overlap'].append(overlap)
            results['energy_diff'].append(energy_diff)
            results['exact_energies'].append(E_exact)
            results['predicted_energies'].append(best_energy)
            results['predicted_configs'].append(best_config)
            results['success'].append(success)
            
            status = "✓ EXACT" if success else "✗ approx"
            print(f"Test {idx+1:2d}: E_exact={E_exact:8.4f}, E_pred={best_energy:8.4f}, "
                  f"ΔE={energy_diff:8.4f}, overlap={overlap:.3f} {status}")
        
        print("-" * 70)
        
        return results


def generate_training_data(n_samples: int = 100, n_vertices: int = 16) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    print(f"\nGenerating {n_samples} training samples (N={n_vertices})...")
    print("This may take a few minutes for N=16 (2^16 = 65536 configurations per sample)")
    print("-" * 70)
    
    training_data = []
    gnn = SpinGlassGNN(n_vertices=n_vertices)
    
    start_time = time.time()
    
    for i in range(n_samples):
        J = np.random.uniform(-0.5, 0.5, (n_vertices, n_vertices))
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        
        ground_state, ground_energy = gnn.find_exact_ground_state(J)
        training_data.append((J, ground_state, ground_energy))
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"  Progress: {i+1:3d}/{n_samples} samples "
                  f"({100*(i+1)/n_samples:.1f}%) - ETA: {remaining:.1f}s")
    
    total_time = time.time() - start_time
    print(f"  Complete! Generated {n_samples} samples in {total_time:.1f}s")
    print("-" * 70)
    
    return training_data


def plot_results(loss_history: List[float], test_results: Dict, n_test: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(loss_history, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    x = np.arange(1, n_test + 1)
    axes[1].scatter(x, test_results['exact_energies'], label='Exact Ground State', 
                   s=120, alpha=0.7, marker='o')
    axes[1].scatter(x, test_results['predicted_energies'], label='GNN Prediction', 
                   s=120, alpha=0.7, marker='x', linewidths=2)
    axes[1].set_xlabel('Test Case', fontsize=12)
    axes[1].set_ylabel('Energy', fontsize=12)
    axes[1].set_title('Exact vs Predicted Ground State Energies', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(x)
    
    plt.tight_layout()
    plt.savefig('gnn_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved to: gnn_results.png")


def print_summary_statistics(test_results: Dict):
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    avg_overlap = np.mean(test_results['overlap'])
    avg_energy_diff = np.mean(test_results['energy_diff'])
    success_rate = 100 * np.sum(test_results['success']) / len(test_results['success'])
    
    print(f"Average overlap with exact ground state: {avg_overlap:.4f}")
    print(f"Average energy difference (ΔE):         {avg_energy_diff:.6f}")
    print(f"Exact solutions found:                  {success_rate:.1f}%")
    print(f"Min energy difference:                  {np.min(test_results['energy_diff']):.6f}")
    print(f"Max energy difference:                  {np.max(test_results['energy_diff']):.6f}")
    print("="*70)


def main():
    print("="*70)
    print("GNN DEEP LEARNING FOR SHERRINGTON-KIRKPATRICK SPIN GLASS MODEL")
    print("PHY480 - Classes 25 & 26")
    print("="*70)
    
    N_VERTICES = 16
    N_LAYERS = 10
    ALPHA = 0.5
    N_TRAINING = 100
    N_TEST = 10
    LEARNING_RATE = 0.01
    N_EPOCHS = 10
    BATCH_SIZE = 10
    N_ATTEMPTS = 10
    
    print(f"\nParameters:")
    print(f"  N (vertices/spins):     {N_VERTICES}")
    print(f"  m (layers):             {N_LAYERS}")
    print(f"  α (balance param):      {ALPHA}")
    print(f"  Training samples:       {N_TRAINING}")
    print(f"  Test samples:           {N_TEST}")
    print(f"  Learning rate:          {LEARNING_RATE}")
    print(f"  Epochs:                 {N_EPOCHS}")
    print(f"  Batch size:             {BATCH_SIZE}")
    print(f"  Attempts per test:      {N_ATTEMPTS}")
    
    training_data = generate_training_data(N_TRAINING, N_VERTICES)
    test_data = generate_training_data(N_TEST, N_VERTICES)
    
    print("\n" + "="*70)
    print("TRAINING PHASE")
    print("="*70)
    
    gnn = SpinGlassGNN(n_vertices=N_VERTICES, n_layers=N_LAYERS, alpha=ALPHA)
    loss_history = gnn.train(training_data, learning_rate=LEARNING_RATE, 
                            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)
    
    print("\n" + "="*70)
    print("TESTING PHASE")
    print("="*70)
    
    test_results = gnn.test(test_data, n_attempts=N_ATTEMPTS)
    
    print_summary_statistics(test_results)
    
    plot_results(loss_history, test_results, N_TEST)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
