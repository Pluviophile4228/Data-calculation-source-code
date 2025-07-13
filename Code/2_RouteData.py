import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.ticker as ticker
from scipy.integrate import cumulative_trapezoid
plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]

class FailureRateAnalyzer:
    def __init__(self, median=1e-5, EF=2, t_obs=1.5e4, k_obs=1, n_intervals=20):
        """Initialize analysis parameters"""
        self.median = median  # Median failure rate
        self.EF = EF          # Error factor
        self.t_obs = t_obs    # Observation time
        self.k_obs = k_obs    # Observed number of failures
        self.n = n_intervals  # Number of focal elements in evidence body

        # Calculate prior interval
        self.b = median / EF  # Lower bound of interval
        self.c = median * EF  # Upper bound of interval
        self.mu = median      # Expected value

        # Initialize variables
        self.lmbda_values = None  # Failure rate array
        self.prior_vals = None    # Prior possibility distribution
        self.posterior_vals = None  # Posterior possibility distribution (normalized)
        self.focal_elements = []  # Focal element intervals
        self.mass_values = []     # Basic belief assignments
        self.bel_values = []      # Belief values (Bel)
        self.pl_values = []       # Plausibility values (Pl)

        # Execute core analysis process
        self._run_analysis()

    def _prior_possibility(self, x):
        """Prior possibility distribution (Equation 21)"""
        if x < self.b or x > self.c:
            return 0.0
        elif x <= self.mu:
            return (self.c - self.mu) / (self.c - x)
        else:
            return 1 - (x - self.mu) / (x - self.b)

    def _likelihood(self, lmbda):
        """Likelihood function (Poisson distribution)"""
        return (lmbda * self.t_obs) ** self.k_obs * np.exp(-lmbda * self.t_obs)

    def _likelihood_possibility(self, lmbda):
        """Normalized likelihood possibility distribution"""
        # Maximize likelihood function to find normalization constant
        res = minimize_scalar(
            lambda x: -self._likelihood(x),
            bounds=(self.b, self.c),
            method='bounded'
        )
        max_likelihood = self._likelihood(res.x)
        return self._likelihood(lmbda) / max_likelihood

    def _posterior_possibility(self, lmbda):
        """Posterior possibility distribution (Equation 23)"""
        prior = self._prior_possibility(lmbda)
        if prior == 0:
            return 0.0
        return self._likelihood_possibility(lmbda) / prior

    def _calculate_cdf(self):
        """Calculate probability envelope and actual CDF"""
        # Normalize posterior distribution (maximum value is 1)
        max_posterior = np.max(self.posterior_vals)
        posterior_normalized = self.posterior_vals / max_posterior

        # Calculate actual CDF (cumulative integral)
        area = np.trapezoid(posterior_normalized, self.lmbda_values)
        pdf_vals = posterior_normalized / area
        cdf_vals = cumulative_trapezoid(pdf_vals, self.lmbda_values, initial=0)
        cdf_vals /= cdf_vals[-1]  # Ensure maximum value is 1

        # Calculate mode position (for upper and lower envelopes)
        a_index = np.argmax(posterior_normalized)
        self.a = self.lmbda_values[a_index]  # Mode

        # Upper and lower envelope functions
        def cdf_lower(x):
            idx = np.searchsorted(self.lmbda_values, x)
            return posterior_normalized[idx] if idx <= a_index else 1.0

        def cdf_upper(x):
            idx = np.searchsorted(self.lmbda_values, x)
            return 0.0 if idx <= a_index else 1 - posterior_normalized[idx]

        return cdf_lower, cdf_upper, cdf_vals

    def _generate_evidence_body(self):
        """Generate evidence body (focal elements and basic beliefs)"""
        # Generate focal elements based on α-cuts
        alpha_levels = np.linspace(1, 0, self.n + 1)[:-1]  # α from 1 to 0 with n values
        self.focal_elements = []
        for alpha in alpha_levels:
            # Find interval for α-cut
            above_alpha = self.lmbda_values[self.posterior_vals >= alpha]
            if len(above_alpha) > 0:
                l_i = np.min(above_alpha)
                u_i = np.max(above_alpha)
            else:
                l_i = u_i = self.a  # Default to mode
            self.focal_elements.append((l_i, u_i))

        # Uniformly distribute basic belief
        self.mass_values = np.ones(self.n) / self.n

    def _calculate_bel_pl(self):
        """Calculate Bel and Pl for each focal element"""
        self.bel_values = []
        self.pl_values = []
        for (target_l, target_u) in self.focal_elements:
            bel = 0.0  # Belief: sum of beliefs for focal elements fully contained in target interval
            pl = 0.0   # Plausibility: sum of beliefs for focal elements intersecting with target interval
            for (l, u), m in zip(self.focal_elements, self.mass_values):
                # Calculate Bel
                if l >= target_l and u <= target_u:
                    bel += m
                # Calculate Pl (skip if no intersection)
                if not (u < target_l or l > target_u):
                    pl += m
            self.bel_values.append(bel)
            self.pl_values.append(pl)

    def _find_min_bel_pl_interval(self):
        """Find interval with minimum Bel-Pl difference"""
        min_diff = float('inf')
        best_interval = None
        for i, (l, u) in enumerate(self.focal_elements):
            diff = self.pl_values[i] - self.bel_values[i]
            if diff < min_diff:
                min_diff = diff
                best_interval = (l, u)
        return min_diff, best_interval

    def _visualize(self):
        """Visualize results (4 core charts)"""
        # Scientific notation formatter
        def sci_formatter(x, pos):
            if x == 0:
                return "0"
            exponent = int(np.floor(np.log10(abs(x))))
            coeff = x / (10 **exponent)
            return f"{coeff:.1f}E{exponent}"

        # 1. Prior possibility distribution chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.lmbda_values, self.prior_vals, 'b-', linewidth=2, label='Prior Possibility')
        plt.fill_between(self.lmbda_values, self.prior_vals, 0, alpha=0.1, color='blue')
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        plt.title('Prior Possibility Distribution')
        plt.xlabel('Failure Rate (λ, h⁻¹)')
        plt.ylabel('Possibility')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig('prior_distribution.png', dpi=300)
        plt.show()

        # 2. Prior vs posterior comparison chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.lmbda_values, self.prior_vals, 'b-', linewidth=2, label='Prior')
        plt.plot(self.lmbda_values, self.posterior_vals, 'r-', linewidth=2, label='Posterior')
        plt.fill_between(self.lmbda_values, self.posterior_vals, 0, alpha=0.1, color='red')
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        plt.title('Prior vs Posterior Possibility Distributions')
        plt.xlabel('Failure Rate (λ, h⁻¹)')
        plt.ylabel('Possibility')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig('prior_vs_posterior.png', dpi=300)
        plt.show()

        # 3. Probability envelope chart
        cdf_lower, cdf_upper, cdf_vals = self._calculate_cdf()
        cdf_low_vals = np.array([cdf_lower(x) for x in self.lmbda_values])
        cdf_up_vals = np.array([cdf_upper(x) for x in self.lmbda_values])
        plt.figure(figsize=(10, 6))
        plt.plot(self.lmbda_values, cdf_low_vals, 'g-', linewidth=2, label='Lower CDF Envelope')
        plt.plot(self.lmbda_values, cdf_up_vals, 'm-', linewidth=2, label='Upper CDF Envelope')
        plt.plot(self.lmbda_values, cdf_vals, 'b--', linewidth=1.5, label='Actual CDF')
        plt.fill_between(self.lmbda_values, cdf_low_vals, cdf_up_vals, alpha=0.1, color='cyan')
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        plt.title('Probability Envelope with Actual CDF')
        plt.xlabel('Failure Rate (λ, h⁻¹)')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig('probability_envelope.png', dpi=300)
        plt.show()

        # # 4. Evidence body focal elements chart
        # plt.figure(figsize=(12, 8))
        # for i, (l_i, u_i) in enumerate(self.focal_elements):
        #     plt.plot([l_i, u_i], [i, i], 'ko-', markersize=4)
        #     plt.text(self.c * 1.3, i, f"[{l_i:.1e}, {u_i:.1e}]", va='center', fontsize=9)
        # plt.xscale('log')
        # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter))
        # plt.xlabel('Failure Rate (λ, h⁻¹)')
        # plt.yticks([])
        # plt.title(f'Evidence Body Focal Elements (n={self.n})')
        # plt.ylim(-1, self.n)
        # plt.grid(True, which="both", ls="--", axis='x')
        # plt.tight_layout()
        # plt.savefig('focal_elements.png', dpi=300)
        # plt.show()

    def _print_results(self):
        """Print evidence body table (including Bel and Pl)"""
        print("\nEvidence Body (Focal Elements with Belief and Plausibility):")
        print("=" * 100)
        print(f"{'Index':^6} | {'Lower Bound':^20} | {'Upper Bound':^20} | {'Mass':^8} | {'Bel':^8} | {'Pl':^8}")
        print("-" * 100)
        for i in range(self.n):
            l, u = self.focal_elements[i]
            print(f"{i+1:^6} | {l:^20.3e} | {u:^20.3e} | {self.mass_values[i]:^8.4f} | {self.bel_values[i]:^8.4f} | {self.pl_values[i]:^8.4f}")
        print("=" * 100)

        # Output interval with minimum Bel-Pl difference
        min_diff, best_interval = self._find_min_bel_pl_interval()
        print(f"\nInterval with minimum Bel-Pl difference: [{best_interval[0]:.3e}, {best_interval[1]:.3e}], Difference: {min_diff:.4f}")

    def _run_analysis(self):
        """Execute complete analysis process"""
        # Generate failure rate array (logarithmic space)
        self.lmbda_values = np.logspace(np.log10(self.b), np.log10(self.c), 1000)

        # Calculate prior and posterior distributions
        self.prior_vals = np.array([self._prior_possibility(lmbda) for lmbda in self.lmbda_values])
        self.posterior_vals = np.array([self._posterior_possibility(lmbda) for lmbda in self.lmbda_values])

        # Generate evidence body
        self._generate_evidence_body()

        # Calculate Bel and Pl
        self._calculate_bel_pl()

        # Visualization and output
        self._visualize()
        self._print_results()


def main():
    """Main function: configure parameters and run analysis"""
    # Parameter configuration
    median = 1e-6       # Median failure rate
    EF = 5              # Error factor
    t_obs = 1.0e7       # Observation time (hours)
    k_obs = 1           # Observed number of failures
    n_intervals = 20    # Number of focal elements in evidence body

    # Initialize analyzer and execute
    analyzer = FailureRateAnalyzer(
        median=median,
        EF=EF,
        t_obs=t_obs,
        k_obs=k_obs,
        n_intervals=n_intervals
    )


if __name__ == "__main__":
    main()