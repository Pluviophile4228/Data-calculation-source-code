import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import random

class FaultTreeAnalyzer:
    """Fault Tree Analyzer: Calculates top event failure probability and confidence intervals"""

    def __init__(self, logic_relations, failure_intervals):
        """
        Initialize the analyzer

        Parameters:
            logic_relations: List of fault tree logic relations, in the format [(parent node, child node, logic gate), ...]
            failure_intervals: Dictionary of basic event failure probability intervals, in the format {event: (lower bound, upper bound), ...}
        """
        # Store original data
        self.logic_relations = logic_relations
        self.failure_intervals = self._convert_intervals_to_float(failure_intervals)

        # Build fault tree structure
        self.dependencies, self.gate_types = self._build_dependencies()
        self.all_nodes = self._get_all_nodes()
        self.basic_events = set(self.failure_intervals.keys())
        self.non_basic_events = self.all_nodes - self.basic_events

        # Validate fault tree structure
        self._validate_structure()

    def _convert_intervals_to_float(self, intervals):
        """Convert string-formatted probability intervals to floating-point numbers"""
        return {
            event: (float(low), float(high))
            for event, (low, high) in intervals.items()
        }

    def _build_dependencies(self):
        """Build node dependency relationships and logic gate type dictionary"""
        dependencies = {}  # {parent node: [list of child nodes]}
        gate_types = {}  # {parent node: logic gate type}
        for parent, child, gate in self.logic_relations:
            if parent not in dependencies:
                dependencies[parent] = []
                gate_types[parent] = gate
            dependencies[parent].append(child)
        return dependencies, gate_types

    def _get_all_nodes(self):
        """Extract all nodes (parent nodes + child nodes)"""
        all_nodes = set()
        for parent, children in self.dependencies.items():
            all_nodes.add(parent)
            for child in children:
                all_nodes.add(child)
        return all_nodes

    def _validate_structure(self):
        """Validate the integrity of the fault tree structure (non-basic events must have logic gate definitions)"""
        for node in self.non_basic_events:
            if node not in self.gate_types:
                raise ValueError(f"Non-basic event '{node}' has no defined logic gate type. Please check the logic relations.")

    def compute_node_probability(self, node, prob_dict):
        """
        Recursively calculate the failure probability of a node

        Parameters:
            node: Name of the node to calculate
            prob_dict: Dictionary of basic event probabilities {event: probability value}

        Returns:
            Failure probability of the node
        """
        # For basic events, directly return the probability
        if node in prob_dict:
            return prob_dict[node]

        # Recursively calculate child node probabilities
        child_probs = [
            self.compute_node_probability(child, prob_dict)
            for child in self.dependencies.get(node, [])
        ]

        # Nodes with no dependencies return 0 (theoretically should not occur)
        if not child_probs:
            return 0.0

        # Calculate probability based on logic gate
        gate = self.gate_types[node]
        if gate == 'AND':
            # AND gate: current node fails only when all child nodes fail
            return np.prod(child_probs)
        elif gate == 'OR':
            # OR gate: current node fails when any child node fails
            return 1.0 - np.prod([1.0 - p for p in child_probs])
        else:
            raise ValueError(f"Unsupported logic gate type: {gate} (node: {node})")

    def run_monte_carlo(self, n_simulations=100000):
        """
        Perform Monte Carlo simulation to calculate the distribution of top event failure probabilities

        Parameters:
            n_simulations: Number of simulations

        Returns:
            List of top event failure probabilities
        """
        top_event_probs = []
        for _ in range(n_simulations):
            # 1. Sample basic event probabilities (uniform distribution within intervals)
            sampled_probs = {
                event: random.uniform(low, high)
                for event, (low, high) in self.failure_intervals.items()
            }

            # 2. Calculate top event probability
            top_prob = self.compute_node_probability(
                node='Uncommanded movement of flaps',   #!!!!!
                prob_dict=sampled_probs
            )
            top_event_probs.append(top_prob)

        return top_event_probs

    def analyze(self, n_simulations=100000, plot_cdf=True):
        """
        Complete analysis process: simulation + calculation of 95% confidence level point estimate + visualization

        Parameters:
            n_simulations: Number of Monte Carlo simulations
            plot_cdf: Whether to plot the cumulative distribution function

        Returns:
            Top event failure probability at 95% confidence level (point estimate, taking the upper bound of the confidence interval)
        """
        # Perform Monte Carlo simulation
        print(f"Start Monte Carlo simulation（{n_simulations} times）...")
        top_event_probs = self.run_monte_carlo(n_simulations)

        # Calculate 95% confidence interval and point estimate
        ci_lower, ci_upper = np.percentile(top_event_probs, [2.5, 97.5])
        point_estimate = ci_upper  # Take the upper bound of the confidence interval as a conservative estimate

        # Output results
        print(f"\n95% confidence level: {point_estimate:.6e}")
        print(f"（95% confidence interval: [{ci_lower:.6e}, {ci_upper:.6e}]）")

        # Plot CDF curve
        if plot_cdf:
            self._plot_cdf(top_event_probs, point_estimate, ci_lower, ci_upper)

        return point_estimate

    def _plot_cdf(self, top_probs, point_estimate, ci_lower, ci_upper):
        """Plot the cumulative distribution function (CDF) of top event failure probabilities"""
        sorted_probs = np.sort(top_probs)
        cdf_vals = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_probs, cdf_vals, label='CDF')
        plt.axvline(
            x=point_estimate,
            color='r',
            linestyle='--',
            label=f'95% confidence interval point estimate: {point_estimate:.6e}'
        )
        plt.fill_betweenx(
            cdf_vals,
            sorted_probs,
            where=((sorted_probs >= ci_lower) & (sorted_probs <= ci_upper)),
            color='blue',
            alpha=0.1,
            label='95% confidence interval'
        )

        plt.title('CDF of thermal runaway failure probability')
        plt.xlabel('Failure probability')
        plt.ylabel('Cumulative probability')
        plt.xscale('log')  # Logarithmic scale to accommodate small probability ranges
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    """Main function: define parameters + perform analysis"""
    # 1. Define fault tree parameters
    failure_intervals = {
        "COM1 failure": ["2.000e-06", "4.771e-06"],
        "COM2 failure": ["2.000e-06", "4.771e-06"],
        "MON1 failure": ["2.000e-06", "4.771e-06"],
        "MON2 failure": ["2.000e-06", "4.771e-06"],
        "Sensor 1A channel failure": ["1.000e-07", "4.341e-06"],
        "Sensor 1B channel failure": ["1.000e-07", "4.341e-06"],
        "Sensor 2A channel failure": ["1.000e-07", "4.341e-06"],
        "Sensor 2B channel failure": ["1.000e-07", "4.341e-06"],
        "left WTB failure": ["2.876e-06", "3.750e-05"],
        "right WTB failure": ["2.876e-06", "3.750e-05"],
        "left POB failure": ["2.301e-06", "3.175e-05"],
        "right POB failure": ["2.301e-06", "3.175e-05"]
    }

    logic_relations = [
        ('Uncommanded movement of flaps', 'Incorrect FQL output', 'OR'),
        ('Uncommanded movement of flaps', 'FECU false command issuance', 'OR'),
        ('Uncommanded movement of flaps', 'system loss of braking capability', 'OR'),
        ('Incorrect FQL output', 'Sensor 1 incorrect output', 'OR'),
        ('Incorrect FQL output', 'Sensor 2 incorrect output', 'OR'),
        ('Sensor 1 incorrect output', 'Sensor 1A channel failure', 'AND'),
        ('Sensor 1 incorrect output', 'Sensor 1B channel failure', 'AND'),
        ('Sensor 2 incorrect output', 'Sensor 2A channel failure', 'AND'),
        ('Sensor 2 incorrect output', 'Sensor 2B channel failure', 'AND'),
        ('FECU false command issuance', 'FECU1 false command issuance', 'OR'),
        ('FECU false command issuance', 'FECU2 false command issuance', 'OR'),
        ('FECU1 false command issuance', 'COM1 failure', 'AND'),
        ('FECU1 false command issuance', 'MON1 failure', 'AND'),
        ('FECU2 false command issuance', 'COM2 failure', 'AND'),
        ('FECU2 false command issuance', 'MON2 failure', 'AND'),
        ('system loss of braking capability', 'WTB failure', 'AND'),
        ('system loss of braking capability', 'PDU braking failure', 'AND'),
        ('WTB failure', 'left WTB failure', 'OR'),
        ('WTB failure', 'right WTB failure', 'OR'),
        ('PDU braking failure', 'left POB failure', 'OR'),
        ('PDU braking failure', 'right POB failure', 'OR'),
    ]

    # failure_intervals = {
    #     "COM1 failure": ["1.500e-06", "5.50e-06"],
    #     "COM2 failure": ["1.500e-06", "5.50e-06"],
    #     "MON1 failure": ["1.250e-06", "4.583e-06"],
    #     "MON2 failure": ["1.250e-06", "4.583e-06"],
    #     "Sensor 1A channel failure": ["5.000e-07", "1.511e-06"],
    #     "Sensor 1B channel failure": ["5.000e-07", "1.511e-06"],
    #     "Sensor 2A channel failure": ["5.000e-07", "1.511e-06"],
    #     "Sensor 2B channel failure": ["5.000e-07", "1.511e-06"]
    #     # "left WTB failure": ["2.876e-06", "3.750e-05"],
    #     # "right WTB failure": ["2.876e-06", "3.750e-05"],
    #     # "left POB failure": ["2.301e-06", "3.175e-05"],
    #     # "right POB failure": ["2.301e-06", "3.175e-05"]
    # }

    # logic_relations = [
    #     ('Flaps not moving as commanded', 'Incorrect FQL output', 'OR'),
    #     ('Flaps not moving as commanded', 'FECU false command issuance', 'OR'),
    #     # ('Uncommanded movement of flaps', 'system loss of braking capability', 'OR'),
    #     ('Incorrect FQL output', 'Sensor 1 incorrect output', 'OR'),
    #     ('Incorrect FQL output', 'Sensor 2 incorrect output', 'OR'),
    #     ('Sensor 1 incorrect output', 'Sensor 1A channel failure', 'AND'),
    #     ('Sensor 1 incorrect output', 'Sensor 1B channel failure', 'AND'),
    #     ('Sensor 2 incorrect output', 'Sensor 2A channel failure', 'AND'),
    #     ('Sensor 2 incorrect output', 'Sensor 2B channel failure', 'AND'),
    #     ('FECU false command issuance', 'FECU1 false command issuance', 'OR'),
    #     ('FECU false command issuance', 'FECU2 false command issuance', 'OR'),
    #     ('FECU1 false command issuance', 'COM1 failure', 'AND'),
    #     ('FECU1 false command issuance', 'MON1 failure', 'AND'),
    #     ('FECU2 false command issuance', 'COM2 failure', 'AND'),
    #     ('FECU2 false command issuance', 'MON2 failure', 'AND')
    #     # ('system loss of braking capability', 'WTB failure', 'AND'),
    #     # ('system loss of braking capability', 'PDU braking failure', 'AND'),
    #     # ('WTB failure', 'left WTB failure', 'OR'),
    #     # ('WTB failure', 'right WTB failure', 'OR'),
    #     # ('PDU braking failure', 'left POB failure', 'OR'),
    #     # ('PDU braking failure', 'right POB failure', 'OR'),
    # ]

    # 2. Perform analysis
    analyzer = FaultTreeAnalyzer(
        logic_relations=logic_relations,
        failure_intervals=failure_intervals
    )
    analyzer.analyze(
        n_simulations=100000,  # Number of simulations
        plot_cdf=True  # Plot CDF curve
    )


if __name__ == "__main__":
    main()