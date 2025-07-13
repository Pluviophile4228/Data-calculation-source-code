import itertools
import decimal
import numpy as np

# Set high-precision calculation environment
decimal.getcontext().prec = 50  # Set precision to 50 digits


class EvidenceNetworkNode:
    def __init__(self, name):
        self.name = name
        self.children = []  # List of child nodes
        self.parent = None  # Parent node
        self.bpa = {'F': decimal.Decimal(0), 'C': decimal.Decimal(0), 'FC': decimal.Decimal(0)}  # High-precision BPA
        self.interval = [decimal.Decimal(0), decimal.Decimal(0)]  # High-precision failure rate interval
        self.gate_type = None  # Logic gate type ('AND', 'OR')
        self.cbmt = {}  # Conditional belief mass table

    def set_failure_interval(self, lower, upper):
        """Set the failure rate interval of the node and calculate high-precision BPA"""
        self.interval = [decimal.Decimal(lower), decimal.Decimal(upper)]
        self.bpa = {
            'F': self.interval[0],
            'C': decimal.Decimal(1) - self.interval[1],
            'FC': self.interval[1] - self.interval[0]
        }

    def __repr__(self):
        # Use original high-precision values without formatting
        return (f"<Node {self.name}: Interval=[{self.interval[0]}, {self.interval[1]}], "
                f"Gate={self.gate_type}>")


class EvidenceNetwork:
    def __init__(self):
        self.nodes = {}  # Mapping from node name to node

    def add_node(self, name):
        """Add a new node to the network"""
        if name not in self.nodes:
            self.nodes[name] = EvidenceNetworkNode(name)
        return self.nodes[name]

    def add_relation(self, parent_name, child_name, gate_type):
        """Add parent-child relationship"""
        parent = self.add_node(parent_name)
        child = self.add_node(child_name)
        parent.children.append(child)
        child.parent = parent
        parent.gate_type = gate_type  # Set logic gate type for parent node

    def build_cbmt(self, node):
        """Build Conditional Belief Mass Table (CBMT) for logic gate nodes"""
        if not node.children or node.gate_type not in ['AND', 'OR']:
            return

        # Get state combinations of child nodes
        child_states = [['F', 'C', 'FC'] for _ in node.children]
        combinations = itertools.product(*child_states)

        # Initialize CBMT
        cbmt = {}

        # Calculate parent node state for each state combination
        for state_combo in combinations:
            state_key = tuple(state_combo)

            # AND gate: Parent fails only if all children fail
            if node.gate_type == 'AND':
                if all(state == 'F' for state in state_combo):
                    parent_state = 'F'
                elif any(state == 'C' for state in state_combo):
                    parent_state = 'C'
                else:
                    parent_state = 'FC'

            # OR gate: Parent fails if any child fails
            else:  # node.gate_type == 'OR'
                if any(state == 'F' for state in state_combo):
                    parent_state = 'F'
                elif all(state == 'C' for state in state_combo):
                    parent_state = 'C'
                else:
                    parent_state = 'FC'

            cbmt[state_key] = parent_state

        node.cbmt = cbmt

    def propagate_uncertainty(self, node):
        """Uncertainty propagation calculation - returns BPA distribution of the node"""
        # If node is a leaf node, return its BPA directly
        if not node.children:
            return {
                'F': node.bpa['F'],
                'C': node.bpa['C'],
                'FC': node.bpa['FC']
            }

        # Recursively calculate BPA for all child nodes
        child_bpas = [self.propagate_uncertainty(child) for child in node.children]

        # Build conditional belief mass table
        self.build_cbmt(node)

        # Initialize parent node's BPA
        parent_bpa = {'F': decimal.Decimal(0), 'C': decimal.Decimal(0), 'FC': decimal.Decimal(0)}
        child_state_keys = itertools.product(*[bpa.keys() for bpa in child_bpas])

        # Iterate through all possible state combinations of child nodes
        for state_combination in child_state_keys:
            # Calculate joint probability of current state combination
            prob = decimal.Decimal(1)
            for child_idx, state in enumerate(state_combination):
                prob *= child_bpas[child_idx][state]

            # Get parent node's state under this combination
            parent_state = node.cbmt[state_combination]

            # Accumulate to parent node's BPA
            parent_bpa[parent_state] += prob

        return parent_bpa

    def calculate_failure_interval(self, bpa):
        """Calculate failure rate interval based on BPA"""
        lower = bpa['F']  # Belief
        upper = bpa['F'] + bpa['FC']  # Plausibility
        return [lower, upper], bpa

    def print_network(self):
        """Print network structure"""
        print("\nEvidence Network Structure:")
        for name, node in self.nodes.items():
            children = [child.name for child in node.children]
            print(f"{name} (Gate: {node.gate_type or 'Basic'}) -> "
                  f"Interval=[{node.interval[0]}, {node.interval[1]}] -> "
                  f"Children: {children}")


def format_bpa_value(value):
    """Format BPA value as scientific notation"""
    # Convert Decimal type to scientific notation string, keeping 12 significant digits
    return f"{value:.12e}"


def main():
    # Initialize evidence network
    network = EvidenceNetwork()

    # ==================== Accurate data ====================
    # failure_intervals = {
    #     "COM1 failure": ["1.500e-06", "5.50e-06"],
    #     "COM2 failure": ["1.500e-06", "5.50e-06"],
    #     "MON1 failure": ["1.250e-06", "4.583e-06"],
    #     "MON2 failure": ["1.250e-06", "4.583e-06"],
    #     "Sensor 1A channel failure": ["5.000e-07", "1.511e-06"],
    #     "Sensor 1B channel failure": ["5.000e-07", "1.511e-06"],
    #     "Sensor 2A channel failure": ["5.000e-07", "1.511e-06"],
    #     "Sensor 2B channel failure": ["5.000e-07", "1.511e-06"],
    #     "left WTB failure": ["2.876e-06", "3.750e-05"],
    #     "right WTB failure": ["2.876e-06", "3.750e-05"],
    #     "left POB failure": ["2.301e-06", "3.175e-05"],
    #     "right POB failure": ["2.301e-06", "3.175e-05"]
    # }
    #
    # logic_relations = [
    #     ('Uncommanded movement of flaps', 'Incorrect FQL output', 'OR'),
    #     ('Uncommanded movement of flaps', 'FECU false command issuance', 'OR'),
    #     ('Uncommanded movement of flaps', 'system loss of braking capability', 'OR'),
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
    #     ('FECU2 false command issuance', 'MON2 failure', 'AND'),
    #     ('system loss of braking capability', 'WTB failure', 'AND'),
    #     ('system loss of braking capability', 'PDU braking failure', 'AND'),
    #     ('WTB failure', 'left WTB failure', 'OR'),
    #     ('WTB failure', 'right WTB failure', 'OR'),
    #     ('PDU braking failure', 'left POB failure', 'OR'),
    #     ('PDU braking failure', 'right POB failure', 'OR'),
    # ]

    failure_intervals = {
        "High Temperature Protection Failure": ["1.600e-06", "5.867e-06"],  # 1. High temperature protection failure
        "Cell Defect": ["6.500e-07", "2.383e-06"],  # 2. Cell defect
        "Failure of Charge Prohibition after Over-discharge": ["3.900e-06", "1.430e-05"],  # 3. Failure of charge prohibition after over-discharge
        "Short Circuit": ["9.500e-07", "3.483e-06"],  # 4. Short circuit
        "Communication Fault": ["8.000e-07", "2.933e-06"],  # 5. Communication fault
        "Acquisition Unit Fault": ["1.350e-06", "4.950e-06"],  # 6. Acquisition unit fault
        "Main Control Unit Fault": ["8.500e-07", "3.117e-06"],  # 7. Main control unit fault
        "Power Supply Fault": ["1.600e-06", "5.867e-06"],  # 8. Power supply fault
        "Crushing and Collision": ["5.007e-06", "2.000e-05"],  # 9. Crushing and collision
        "High Temperature Environment": ["2.626e-06", "4.377e-05"],  # 10. High temperature environment
        "Overcharge": ["2.751e-06", "1.782e-05"],  # 11. Overcharge
        "External Puncture Short Circuit": ["1.738e-05", "3.747e-05"],  # 12. External puncture short circuit
    }

    logic_relations = [
        # Top-level relationship - must be AND gate!!!
        ('Battery Thermal Runaway Fire', 'Battery Thermal Runaway', 'AND'),
        ('Battery Thermal Runaway Fire', 'Failure of Thermal Runaway Monitoring or Communication', 'AND'),
        # Battery Thermal Runaway section
        ('Battery Thermal Runaway', 'Thermal Abuse', 'OR'),
        ('Battery Thermal Runaway', 'Cell Defect', 'OR'),
        ('Battery Thermal Runaway', 'Electrical Abuse', 'OR'),
        ('Battery Thermal Runaway', 'Mechanical Abuse', 'OR'),
        # Thermal Abuse section
        ('Thermal Abuse', 'High Temperature Environment', 'OR'),
        ('Thermal Abuse', 'High Temperature Protection Failure', 'OR'),
        # Electrical Abuse section
        ('Electrical Abuse', 'Overcharge', 'OR'),
        ('Electrical Abuse', 'Failure of Charge Prohibition after Over-discharge', 'OR'),
        ('Electrical Abuse', 'Short Circuit', 'OR'),
        # Mechanical Abuse section
        ('Mechanical Abuse', 'External Puncture Short Circuit', 'OR'),
        ('Mechanical Abuse', 'Crushing and Collision', 'OR'),
        # Failure of Thermal Runaway Monitoring or Communication section
        ('Failure of Thermal Runaway Monitoring or Communication', 'Communication Fault', 'OR'),
        ('Failure of Thermal Runaway Monitoring or Communication', 'Acquisition Unit Fault', 'OR'),
        ('Failure of Thermal Runaway Monitoring or Communication', 'Main Control Unit Fault', 'OR'),
        ('Failure of Thermal Runaway Monitoring or Communication', 'Power Supply Fault', 'OR'),
    ]

    # 1. Add nodes and set failure rate intervals
    print("\nSetting basic event failure rate intervals:")
    for node_name, interval_str in failure_intervals.items():
        lower = decimal.Decimal(interval_str[0])
        upper = decimal.Decimal(interval_str[1])
        node = network.add_node(node_name)
        node.set_failure_interval(lower, upper)
        print(f"{node_name:<45} Interval: [{lower:.2e}, {upper:.2e}]")

    # 2. Add logic relations
    print("\nAdding logic relations:")
    for parent, child, gate_type in logic_relations:
        network.add_relation(parent, child, gate_type)
        print(f"{parent:>45} ({gate_type}) -> {child}")

    # 3. Print network structure
    network.print_network()

    # 4. Start uncertainty propagation from top node
    top_node_name = 'Battery Thermal Runaway Fire'
    top_node = network.nodes.get(top_node_name)
    if not top_node:
        print(f"Error: Top node '{top_node_name}' not found")
        return

    # 5. Calculate BPA and failure rate interval for top event
    top_bpa = network.propagate_uncertainty(top_node)
    top_interval, _ = network.calculate_failure_interval(top_bpa)

    print("\n" + "=" * 80)
    print(f"Calculation results for top node '{top_node.name}':")
    print(f"  m(F): {format_bpa_value(top_bpa['F'])}")
    print(f"  m(C): {format_bpa_value(top_bpa['C'])}")
    print(f"  m(FC): {format_bpa_value(top_bpa['FC'])}")
    print(f"Failure rate interval: [{format_bpa_value(top_interval[0])}, {format_bpa_value(top_interval[1])}]")
    print("=" * 80)

if __name__ == "__main__":
    main()