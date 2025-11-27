import numpy as np
import matplotlib.pyplot as plt
import random 
from matplotlib.animation import FuncAnimation

class ExpectationMaximizationAlgorithm():
    def __init__(self, num_components: int, max_iteration: int = 1000, tolerance: float = 1e-3):
        self.num_components = num_components
        self.max_iteration = max_iteration
        self.tolerance = tolerance

        # Parameters
        self.means = None
        self.standards = None
        self.weights = None
        self.respons = None
        
        self.means_history = []
        self.standards_history = []
        self.weights_history = []
        self.log_likelihood_history = []

    def initialize_parameters(self, X):
        """Random initialization from data"""
        N = X.shape[0]
        self.num_dimension = 1  # 1D data
        self.means = X[np.random.choice(N, self.num_components, replace=False)]
        self.standards = np.std(X) * np.ones(self.num_components)
        self.weights = np.ones(self.num_components) / self.num_components
        self.means_history.append(self.means.copy())
        self.standards_history.append(self.standards.copy())
        self.weights_history.append(self.weights.copy())

    def gaussian_pdf(self, x, mean, standard):
        """Compute 1D Gaussian PDF"""
        coef = 1 / (np.sqrt(2 * np.pi) * standard)
        exponent = np.exp(-0.5 * ((x - mean) / standard) ** 2)
        return coef * exponent

    def e_step(self, X):
        """E-step: compute responsibilities"""
        K = self.num_components
        N = X.shape[0]
        self.respons = np.zeros((N, K))
        
        for i in range(N):
            for k in range(K):
                self.respons[i, k] = self.weights[k] * self.gaussian_pdf(X[i], self.means[k], self.standards[k])
            # Normalize
            self.respons[i, :] /= np.sum(self.respons[i, :])

    def m_step(self, X):
        """M-step: update parameters"""
        N = X.shape[0]
        K = self.num_components
        
        for k in range(K):
            N_k = np.sum(self.respons[:, k])
            # Update mean
            self.means[k] = np.sum(self.respons[:, k] * X) / N_k
            # Update std
            variance = np.sum(self.respons[:, k] * (X - self.means[k])**2) / N_k
            self.standards[k] = np.sqrt(variance)
            # Update weight
            self.weights[k] = N_k / N
            
        self.means_history.append(self.means.copy())
        self.standards_history.append(self.standards.copy())
        self.weights_history.append(self.weights.copy())

    def compute_log_likelihood(self, X):
        """Compute log-likelihood for convergence check"""
        N = X.shape[0]
        log_likelihood = 0.0
        for i in range(N):
            prob = 0.0
            for k in range(self.num_components):
                prob += self.weights[k] * self.gaussian_pdf(X[i], self.means[k], self.standards[k])
            log_likelihood += np.log(prob + 1e-12)  # avoid log(0)
        return log_likelihood

    def fit(self, X):
        """Fit EM algorithm to data X"""
        X = np.asarray(X)
        self.initialize_parameters(X)
        
        prev_log_likelihood = None
        
        for iteration in range(self.max_iteration):
            self.e_step(X)
            self.m_step(X)
            
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                    break
            prev_log_likelihood = log_likelihood

    def predict_proba(self, X):
        """Return responsibilities (soft assignment)"""
        X = np.asarray(X)
        N = X.shape[0]
        K = self.num_components
        resp = np.zeros((N, K))
        
        for i in range(N):
            for k in range(K):
                resp[i, k] = self.weights[k] * self.gaussian_pdf(X[i], self.means[k], self.standards[k])
            resp[i, :] /= np.sum(resp[i, :])
        return resp

    def predict(self, X):
        """Return hard cluster assignment"""
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)
    
    def demo(self, X):
        fig, (ax_gmm, ax_ll) = plt.subplots(1, 2, figsize=(15, 6))
        x_range = np.linspace(np.min(X) - 1, np.max(X) + 1, 1000)
        colors = ['red', 'blue', 'green']
        
        # GMM Subplot
        ax_gmm.hist(X, bins=40, density=True, alpha=0.3, color='gray', label='Data')
        ax_gmm.set_title(f'EM Process (Iteration 0)', fontsize=14, fontweight='bold')
        ax_gmm.set_xlabel('Value')
        ax_gmm.set_ylabel('Density')
        ax_gmm.grid(True, alpha=0.3)
        ax_gmm.set_ylim(0, 0.4) 

        # Intial plot Gaussian
        gmm_lines = []
        for k in range(self.num_components):
            line, = ax_gmm.plot(x_range, np.zeros_like(x_range), 
                                '--', color=colors[k % len(colors)], 
                                label=f'Component {k+1}')
            gmm_lines.append(line)
        ax_gmm.legend(loc='upper right')
        
        # Log-Likelihood Subplot
        ax_ll.set_title('Log-Likelihood History', fontsize=14, fontweight='bold')
        ax_ll.set_xlabel('Iteration')
        ax_ll.set_ylabel('Log-Likelihood')
        ax_ll.grid(True, alpha=0.3)
        ll_line, = ax_ll.plot([], [], 'o-', color='purple', linewidth=2, markersize=4)
        
        if self.log_likelihood_history:
             min_ll = min(self.log_likelihood_history) - 10 
             max_ll = max(self.log_likelihood_history) + 10
             ax_ll.set_ylim(min_ll, max_ll)
        
        # Update Function
        def update(frame):
            means = self.means_history[frame]
            standards = self.standards_history[frame]
            weights = self.weights_history[frame]
            
            for k in range(self.num_components):
                comp_pdf = self.gaussian_pdf(x_range, means[k], standards[k])
                weighted_comp_pdf = weights[k] * comp_pdf
                
                gmm_lines[k].set_ydata(weighted_comp_pdf)
                gmm_lines[k].set_label(f'Comp {k+1}: μ={means[k]:.2f}, σ={standards[k]:.2f}, W={weights[k]:.2f}')

            ax_gmm.set_title(
                f'EM Process (Iteration {frame}) | Log-Likelihood: {self.log_likelihood_history[frame-1]:.2f}' 
                if frame > 0 else 'EM Process (Iteration 0)'
            )
            ax_gmm.legend(loc='upper right')
            ll_history_frame = self.log_likelihood_history[:frame]
            ll_line.set_data(range(len(ll_history_frame)), ll_history_frame)
            ax_ll.set_xlim(0, max(1, frame + 1))
            return gmm_lines + [ax_gmm.title, ll_line] 

        # FuncAnimation
        num_frames = len(self.means_history)
        ani = FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            interval=300, 
            blit=False, 
            repeat=False   
        )
        
        plt.tight_layout()
        plt.show() 
        
        return ani

def plot_parameter_history(self):
    iterations = range(len(self.means_history))
    means_hist = np.array(self.means_history)
    stds_hist = np.array(self.standards_history)
    weights_hist = np.array(self.weights_history)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = ['r', 'g', 'b'] 

    # 1. plot Mean
    for k in range(self.num_components):
        axes[0].plot(iterations, means_hist[:, k], f'{colors[k]}o-', label=f'Cluster {k+1}')
    axes[0].set_ylabel('Mean')
    axes[0].set_title('Convergence of Mean')
    axes[0].grid(True)
    axes[0].legend()

    # 2. plot Std
    for k in range(self.num_components):
        axes[1].plot(iterations, stds_hist[:, k], f'{colors[k]}o-')
    axes[1].set_ylabel('Std (Standard Deviation)')
    axes[1].set_title('Convergence of Standard Deviation')
    axes[1].grid(True)
    
    # 3. plot Weight
    for k in range(self.num_components):
        axes[2].plot(iterations, weights_hist[:, k], f'{colors[k]}o-')
    axes[2].set_ylabel('Weight')
    axes[2].set_xlabel('Iteration')
    axes[2].set_title('Convergence of Weight')
    axes[2].grid(True)
    

    plt.savefig("parameter_history.png")
    plt.tight_layout()
    plt.show()
    
def visualize_em_results(em, X):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # create x values for plotting Gaussian components
    x_min, x_max = np.min(X) - 1, np.max(X) + 1
    x_grid = np.linspace(x_min, x_max, 1000)
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # --- CHART 1: INITIALIZATION STATE (Iteration 0) ---
    ax1 = axes[0, 0]
    ax1.hist(X, bins=50, density=True, alpha=0.3, color='gray', label='Original Data')
    ax1.set_title('1. Initialization (Iteration 0)', fontsize=12, fontweight='bold')
    
    # Get parameters from the first iteration history
    init_means = em.means_history[0]
    init_stds = em.standards_history[0]
    init_weights = em.weights_history[0]
    
    for k in range(em.num_components):
        # Gaussian PDF formula: (1 / (std * sqrt(2pi))) * exp(...)
        y = (1 / (np.sqrt(2 * np.pi) * init_stds[k])) * \
            np.exp(-0.5 * ((x_grid - init_means[k]) / init_stds[k]) ** 2) * init_weights[k]
        ax1.plot(x_grid, y, color=colors[k % len(colors)], linestyle='--', linewidth=2, label=f'Cụm {k+1} (Init)')
    ax1.legend(loc='upper right')

    # --- CHART 2: FINAL RESULTS (After all iterations) ---
    ax2 = axes[0, 1]
    ax2.hist(X, bins=50, density=True, alpha=0.3, color='gray', label='Original Data')
    ax2.set_title(f'2. Final Results (After {len(em.log_likelihood_history)} Iterations)', fontsize=12, fontweight='bold')
    
    for k in range(em.num_components):
        # Use final parameters (self.means, self.standards...)
        y = (1 / (np.sqrt(2 * np.pi) * em.standards[k])) * \
            np.exp(-0.5 * ((x_grid - em.means[k]) / em.standards[k]) ** 2) * em.weights[k]
        ax2.plot(x_grid, y, color=colors[k % len(colors)], linewidth=2, label=f'Cluster {k+1} (Final)')
    ax2.legend(loc='upper right')

    # --- CHART 3: LOG-LIKELIHOOD ---
    ax3 = axes[1, 0]
    ax3.plot(range(len(em.log_likelihood_history)), em.log_likelihood_history, 'o-', color='purple', markersize=4)
    ax3.set_title('3. Log-Likelihood (Model Likelihood)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Log-Likelihood')
    ax3.grid(True, alpha=0.3)

    # --- CHART 4: THE CONVERGENCE OF MEANS (Tracking) ---
    ax4 = axes[1, 1]
    means_hist = np.array(em.means_history)
    iterations = range(len(em.means_history))
    
    for k in range(em.num_components):
        ax4.plot(iterations, means_hist[:, k], marker='o', markersize=3, 
                 color=colors[k % len(colors)], label=f'Mean Cluster {k+1}')
    
    ax4.set_title('4. The Convergence of Means', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Mean Value')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.savefig("em_results_visualization.png")
    plt.tight_layout()
    plt.show()

def generate_data():
    """Generate synthetic data from 3 Gaussian distributions"""
    np.random.seed(42)
    
    # Generate data randomly
    n_samples = 5000
    true_means = [-1.3, 3, 1.0]
    true_stds = [0.8, 1.2, 0.6]
    true_weights = [0.3, 0.4, 0.3]
    
    data = []
    for k in range(3):
        n = int(n_samples * true_weights[k])
        samples = np.random.normal(true_means[k], true_stds[k], n)
        data.extend(samples)
    
    data = np.array(data)
    np.random.shuffle(data)
    
    return data, true_means, true_stds, true_weights

def main():
    X, true_means, true_stds, true_weights = generate_data()

    # Fit EM algorithm
    em = ExpectationMaximizationAlgorithm(num_components=3, max_iteration=200)
    em.fit(X)

    print("True parameters:")
    print(f"Means: {true_means}")
    print(f"Standard deviations: {true_stds}")
    print(f"Weights: {true_weights}")

    print("Final parameters:")
    print(f"Means: {em.means}")
    print(f"Standard deviations: {em.standards}")
    print(f"Weights: {em.weights}")

    plot_parameter_history(em)
    visualize_em_results(em, X)
    # Animation demo
    em.demo(X)


if __name__ == "__main__":
    main()