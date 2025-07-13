import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
from scipy.stats import lognorm


class FailureRateEnvelope:
    """Failure Rate Probability Envelope and Evidence Body Analysis Class"""

    def __init__(self, lambda_bar=None, mu=None, sigma=None, EF=2, n=20):
        """Initialize parameters (remain unchanged)"""
        if lambda_bar is not None:
            self.lambda_bar = lambda_bar
        elif mu is not None and sigma is not None:
            self.lambda_bar = np.exp(mu + sigma ** 2 / 2)
        else:
            raise ValueError("Must provide either lambda_bar or mu+sigma")

        self.EF = EF
        self.n = n
        self.mu = mu
        self.sigma = sigma
        self.B = self.lambda_bar / EF
        self.C = self.lambda_bar * EF
        self.intervals = []
        self.m_values = []
        self.generate_evidence_body()

    # The following methods remain unchanged (upper_cdf, lower_cdf, lognormal_cdf, generate_evidence_body, calculate_bel_pl)
    def upper_cdf(self, lambda_val):
        if lambda_val <= self.B:
            return 0.0
        if lambda_val >= self.lambda_bar:
            return 1.0
        return (self.C - self.lambda_bar) / (self.C - lambda_val)

    def lower_cdf(self, lambda_val):
        if lambda_val <= self.lambda_bar:
            return 0.0
        return (lambda_val - self.lambda_bar) / (lambda_val - self.B)

    def lognormal_cdf(self, lambda_val):
        if self.mu is not None and self.sigma is not None:
            return lognorm.cdf(lambda_val, s=self.sigma, scale=np.exp(self.mu))
        return None

    def generate_evidence_body(self):
        Mj = 1 / self.n
        T = (self.C - self.lambda_bar) / (self.C - self.B)
        for j in range(1, self.n + 1):
            pj = (j - 0.5) * Mj
            if pj >= 1:
                lambda1 = self.C
            else:
                lambda1 = self.B if pj <= T else self.C - (self.C - self.lambda_bar) / pj
            lambda2 = (pj * self.B - self.lambda_bar) / (pj - 1) if pj <= T else self.C
            self.intervals.append((lambda1, lambda2))
            self.m_values.append(Mj)

    def calculate_bel_pl(self, target_interval):
        target_lower, target_upper = target_interval
        bel, pl = 0.0, 0.0
        for (interval_lower, interval_upper), m in zip(self.intervals, self.m_values):
            if interval_lower >= target_lower and interval_upper <= target_upper:
                bel += m
            if not (interval_upper < target_lower or interval_lower > target_upper):
                pl += m
        return bel, pl

    def print_results(self):
        """Print table to console (for single failure rate)"""
        print("{:^10} {:^20} {:^20} {:^20} {:^20} {:^20}".format(
            "Index", "Lower Bound (λ)", "Upper Bound (λ)", "Basic Belief (m)", "Bel", "Pl"))
        print("-" * 90)
        for i, ((lower, upper), m) in enumerate(zip(self.intervals, self.m_values)):
            bel, pl = self.calculate_bel_pl((lower, upper))
            print(f"{i + 1:^10} {lower:^20.5e} {upper:^20.5e} {m:^20.5f} {bel:^20.5f} {pl:^20.5f}")
        global_bel, global_pl = self.calculate_bel_pl((self.B, self.C))
        print(f"\nGlobal target interval [{self.B:.5e}, {self.C:.5e}] → Bel: {global_bel:.5f}, Pl: {global_pl:.5f}")

    def save_table_to_txt(self, filename):
        """Write table to txt file (for multiple failure rates)"""
        with open(filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            # Differentiate lambda_bar source (direct input vs. calculated from mu/sigma)
            if self.mu is not None and self.sigma is not None:
                f.write(f"Failure rate λ_bar = {self.lambda_bar:.5e} (calculated from mu={self.mu}, sigma={self.sigma})\n")
            else:
                f.write(f"Failure rate λ_bar = {self.lambda_bar:.5e}\n")
            f.write(f"{'='*50}\n")
            f.write("{:^10} {:^20} {:^20} {:^20} {:^20} {:^20}\n".format(
                "Index", "Lower Bound (λ)", "Upper Bound (λ)", "Basic Belief (m)", "Bel", "Pl"))
            f.write("-" * 90 + "\n")
            for i, ((lower, upper), m) in enumerate(zip(self.intervals, self.m_values)):
                bel, pl = self.calculate_bel_pl((lower, upper))
                f.write(f"{i + 1:^10} {lower:^20.5e} {upper:^20.5e} {m:^20.5f} {bel:^20.5f} {pl:^20.5f}\n")
            global_bel, global_pl = self.calculate_bel_pl((self.B, self.C))
            f.write(f"\nGlobal target interval [{self.B:.5e}, {self.C:.5e}] → Bel: {global_bel:.5f}, Pl: {global_pl:.5f}\n")

    def find_min_pl_bel_interval(self):
        min_diff = float('inf')
        best_interval = None
        for interval in self.intervals:
            bel, pl = self.calculate_bel_pl(interval)
            if (pl - bel) < min_diff:
                min_diff, best_interval = pl - bel, interval
        return min_diff, best_interval

    def plot_envelope(self):
        """Plot probability envelope (keep commented)"""
        lambda_vals = np.logspace(np.log10(self.B), np.log10(self.C), 1000)
        upper_cdf_vals = np.array([self.upper_cdf(lam) for lam in lambda_vals])
        lower_cdf_vals = np.array([self.lower_cdf(lam) for lam in lambda_vals])
        plt.figure(figsize=(10, 6))
        plt.semilogx(lambda_vals, upper_cdf_vals, 'r-', linewidth=2, label='CDF Upper Bound')
        plt.semilogx(lambda_vals, lower_cdf_vals, 'b-', linewidth=2, label='CDF Lower Bound')
        if self.lognormal_cdf is not None:
            lognormal_vals = np.array([self.lognormal_cdf(lam) for lam in lambda_vals])
            plt.semilogx(lambda_vals, lognormal_vals, 'g--', linewidth=2, label='Lognormal CDF')
        plt.axvline(x=self.lambda_bar, color='k', linestyle=':', linewidth=1, label='$\\bar{\\lambda}$')
        plt.axvline(x=self.B, color='gray', linestyle='-.', linewidth=1, label='B')
        plt.axvline(x=self.C, color='gray', linestyle='-.', linewidth=1, label='C')
        plt.title('Failure Rate Probability Envelope')
        plt.xlabel('Failure Rate (λ)')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig('failure_rate_envelope.png', dpi=300)
        plt.show()


def main():
    """Main function: Supports three input methods, writes tables to txt, loops through high-consensus intervals"""
    # ==================== User Input Area ====================
    # Method 1: Single failure rate
    lambda_bar = 3.0e-6

    # Method 2: Calculate lambda_bar from mu and sigma (single value)
    # mu, sigma = -10.41, 1.40

    # Method 3: Multiple failure rates (array form)
    # lambda_bar = [3.2e-6,
    #               1.3e-6,
    #               7.8e-6,
    #               1.9e-6,
    #               1.6e-5,
    #               2.7e-5,
    #               1.7e-5,
    #               3.1e-5
    #               ]

    # Method 4: Multiple mu/sigma pairs (array form)
    # mu_sigma_pairs = [(-10.41, 1.40), (-10.2, 1.3), (-10.6, 1.5)]

    # Common parameters
    EF = 2  # Error Factor
    n = 20  # Number of evidence body discretizations
    output_txt = "failure_rate_tables.txt"  # Output file
    # ====================================================

    # Clear file (to avoid repeated content when running multiple times)
    with open(output_txt, 'w') as f:
        f.write("Failure Rate Evidence Body Analysis Results\n")
        f.write(f"Error Factor (EF)={EF}, Discretization Number (n)={n}\n")
        f.write("=" * 50 + "\n\n")

    # Determine input type and process
    if 'lambda_bar' in locals() and isinstance(lambda_bar, (list, np.ndarray)):
        # Process multiple failure rates (Method 3)
        print(f"Starting processing of multiple failure rates. Results will be written to {output_txt}\n")
        for lb in lambda_bar:
            envelope = FailureRateEnvelope(lambda_bar=lb, EF=EF, n=n)
            envelope.save_table_to_txt(output_txt)
            min_diff, best_interval = envelope.find_min_pl_bel_interval()
            print(f"λ_bar={lb:.5e} → High-consensus interval: [{best_interval[0]:.5e}, {best_interval[1]:.5e}], Difference={min_diff:.5f}")

    elif 'mu_sigma_pairs' in locals() and isinstance(mu_sigma_pairs, list):
        # Process multiple mu/sigma pairs (Method 4)
        print(f"Starting processing of multiple mu/sigma pairs. Results will be written to {output_txt}\n")
        for mu, sigma in mu_sigma_pairs:
            envelope = FailureRateEnvelope(mu=mu, sigma=sigma, EF=EF, n=n)
            envelope.save_table_to_txt(output_txt)
            min_diff, best_interval = envelope.find_min_pl_bel_interval()
            lambda_bar_calc = np.exp(mu + sigma ** 2 / 2)
            print(
                f"mu={mu}, sigma={sigma} → λ_bar={lambda_bar_calc:.5e} → High-consensus interval: [{best_interval[0]:.5e}, {best_interval[1]:.5e}], Difference={min_diff:.5f}")

    elif 'lambda_bar' in locals():
        # Process single failure rate (Method 1)
        envelope = FailureRateEnvelope(lambda_bar=lambda_bar, EF=EF, n=n)
        envelope.print_results()  # Print table to console
        min_diff, best_interval = envelope.find_min_pl_bel_interval()
        envelope.plot_envelope()  # Keep commented
        print(
            f"\nλ_bar={lambda_bar:.5e} → High-consensus interval: [{best_interval[0]:.5e}, {best_interval[1]:.5e}], Difference={min_diff:.5f}")
        envelope.save_table_to_txt(output_txt)  # Write to file

    elif 'mu' in locals() and 'sigma' in locals():
        # Process single mu/sigma pair (Method 2)
        envelope = FailureRateEnvelope(mu=mu, sigma=sigma, EF=EF, n=n)
        envelope.print_results()  # Print table to console
        min_diff, best_interval = envelope.find_min_pl_bel_interval()
        lambda_bar_calc = np.exp(mu + sigma ** 2 / 2)
        print(
            f"\nmu={mu}, sigma={sigma} → λ_bar={lambda_bar_calc:.5e} → High-consensus interval: [{best_interval[0]:.5e}, {best_interval[1]:.5e}], Difference={min_diff:.5f}")
        envelope.save_table_to_txt(output_txt)  # Write to file

    else:
        raise ValueError("Please select an input method: single lambda_bar, array of lambda_bars, single mu/sigma pair, or multiple mu/sigma pairs")

    print(f"\nResults saved to {output_txt}")


if __name__ == "__main__":
    main()    