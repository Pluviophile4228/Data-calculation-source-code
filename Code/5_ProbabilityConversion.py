import math
from typing import Dict, Tuple, List, Callable, Any, Union
from collections import defaultdict
from decimal import Decimal, getcontext, InvalidOperation

# Set high-precision calculation context
getcontext().prec = 40  # 40 decimal places precision

# Type aliases
FrozensetBPA = Dict[frozenset, float]  # BPA type
ProbabilityDist = Dict[str, float]  # Probability distribution type
HighPrecProbDist = Dict[str, Decimal]  # High-precision probability distribution type

# Global precision control - but keep all values without setting to zero
SMALLEST_VALUE = Decimal('1e-40')  # Minimum value to prevent division by zero errors


class HighPrecisionEvidenceProcessor:
    def __init__(self, bpa: FrozensetBPA):
        """Initialize high-precision evidence processor

        Core principles:
        1. Use Decimal for all core calculations
        2. Strictly maintain all numerical precision without truncation
        3. Absolutely no approximation allowed
        """
        # Convert BPA to high-precision Decimal
        self.original_bpa = bpa
        self.high_prec_bpa = {}

        max_mass = Decimal(0)
        for fs, mass in bpa.items():
            # Convert small values to sufficiently precise string representation
            s = str(mass)
            if 'e' in s:
                base, exp = s.split('e')
                value = Decimal(base) * 10 ** Decimal(exp)
            else:
                value = Decimal(s)
            self.high_prec_bpa[fs] = value
            if abs(value) > max_mass:
                max_mass = abs(value)

        # Validate and normalize BPA
        total_mass = self._high_prec_sum(self.high_prec_bpa.values())
        if total_mass == 0:
            # Special case when all masses are zero
            n = len(self.high_prec_bpa)
            self.bpa = {fs: Decimal(1) / n if n > 0 else Decimal(0) for fs in self.high_prec_bpa}
        else:
            # Ensure high-precision normalization
            self.bpa = {fs: mass / total_mass for fs, mass in self.high_prec_bpa.items()}

        # Get frame of discernment Θ and singleton focal elements
        self.singletons = self._get_singletons()
        self.singleton_elements = [next(iter(s)) for s in self.singletons]

        # Precompute Bel and Pl functions (high-precision)
        self.bel, self.pl = self._precompute_bel_pl()

    def _high_prec_sum(self, values) -> Decimal:
        """Safe summation of Decimal values"""
        total = Decimal(0)
        for v in values:
            total += v
        return total

    def _get_singletons(self) -> List[frozenset]:
        """Extract all singleton focal elements from BPA"""
        all_elements = set()
        for fs in self.bpa.keys():
            all_elements |= fs

        # Create singleton frozenset collection
        return [frozenset({elem}) for elem in all_elements]

    def _precompute_bel_pl(self) -> Tuple[Dict[frozenset, Decimal], Dict[frozenset, Decimal]]:
        """Precompute Bel and Pl functions (exact implementation)"""
        bel = defaultdict(lambda: Decimal(0))
        pl = defaultdict(lambda: Decimal(0))

        # Iterate through all focal elements
        for subset, mass in self.bpa.items():
            if not subset:  # Skip empty set
                continue

            # Calculate Bel and Pl for each singleton
            for singleton in self.singletons:
                # Bel calculation: subset ⊆ singleton
                if subset.issubset(singleton):
                    bel[singleton] += mass

                # Pl calculation: subset ∩ singleton ≠ ∅
                if subset.intersection(singleton):
                    pl[singleton] += mass

        return dict(bel), dict(pl)

    def to_probability_distribution(self, probs: Dict[Any, Decimal]) -> ProbabilityDist:
        """Convert high-precision probabilities to standard float probability distribution (strictly preserve all values)"""
        return {str(key): float(prob) for key, prob in probs.items()}

    def to_high_precision_dist(self, probs: Dict[Any, Union[float, Decimal]]) -> HighPrecProbDist:
        """Convert and validate probability distribution (preserve all decimal places)"""
        prob_dist = {}
        for key, value in probs.items():
            if isinstance(key, frozenset) and len(key) == 1:
                element = next(iter(key))
            else:
                element = str(key)

            if isinstance(value, float):
                # Convert float to exact Decimal representation
                prob_dist[element] = Decimal(str(value))
            else:
                prob_dist[element] = value
        return prob_dist

    def normalize_dist(self, dist: Dict[Any, Decimal]) -> Dict[Any, Decimal]:
        """High-precision normalization without precision loss"""
        total = self._high_prec_sum(dist.values())
        if total == 0:
            n = len(dist)
            equal_share = Decimal(1) / n if n > 0 else Decimal(0)
            return {k: equal_share for k in dist.keys()}

        return {k: v / total for k, v in dist.items()}

    # ======================= Probability transformation methods (fully preserve precision) =======================

    def pignistic_transform(self) -> HighPrecProbDist:
        """Pignistic transform (exact version)"""
        result = {s: Decimal(0) for s in self.singletons}

        for subset, mass in self.bpa.items():
            n = len(subset)
            if n == 0:  # Skip empty set
                continue

            # Calculate current subset's cardinality
            subset_cardinality = Decimal(n)

            # Distribute to each singleton
            for element in subset:
                fs = frozenset({element})
                intersection_cardinality = Decimal(1)  # Singleton intersection cardinality with subset is 1

                # Calculate PSD Pignistic transform weight
                numerator = 2 ** intersection_cardinality - 1
                denominator = 2 ** subset_cardinality - 1
                weight = numerator / denominator

                # Update result
                result[fs] = result.get(fs, Decimal(0)) + mass * weight

        # Return exact values after normalization
        normalized = self.normalize_dist(
            {str(next(iter(k))): v for k, v in result.items()}
        )
        return normalized

    def pbetp_transform(self) -> Dict[str, Decimal]:
        """
        PSD Pignistic transform (exact version)
        Calculate PBetP value for each singleton
        """
        result = {str(s): Decimal(0) for s in self.singletons}

        for subset, mass in self.bpa.items():
            n = len(subset)
            if n == 0:  # Skip empty set
                continue

            # Calculate current subset's cardinality |D|
            card_d = Decimal(n)
            # Calculate 2^|D| - 1
            denominator = (Decimal(2) ** card_d) - Decimal(1)
            if denominator == 0:
                continue  # Avoid division by zero

            for element in self.singletons:
                # Calculate |θ_i ∩ D|
                intersection = subset.intersection({element})
                card_intersection = Decimal(len(intersection))
                # Calculate 2^|θ_i ∩ D| - 1
                numerator = (Decimal(2) ** card_intersection) - Decimal(1)

                # Calculate current subset's contribution to current element
                contribution = mass * (numerator / denominator)
                result[str(element)] += contribution

        # Return exact values after normalization
        normalized = self.normalize_dist(result)
        return normalized

    def robust_interval_transform(self) -> HighPrecProbDist:
        """Interval probability transformation method (exact version)"""
        prob_dist = {}
        total_uncertainty = Decimal(0)
        total_bel = Decimal(0)

        # Exactly calculate bel and pl for each singleton
        for s in self.singletons:
            element = next(iter(s))
            bel_val = self.bel.get(s, Decimal(0))
            pl_val = self.pl.get(s, Decimal(0))
            uncertainty = pl_val - bel_val

            prob_dist[element] = bel_val + uncertainty
            total_uncertainty += uncertainty
            total_bel += bel_val

        # Normalization
        return self.normalize_dist(prob_dist)

    # ======================= Sudano transformation series (fully preserve precision) =======================

    def sudano_general_transform(self, weight_func: Callable[[frozenset], Decimal]) -> HighPrecProbDist:
        """Sudano transformation general framework (fully preserve precision)"""
        prob_dist = {s: Decimal(0) for s in self.singletons}

        for subset, mass in self.bpa.items():
            if not subset:  # Skip empty set
                continue

            if len(subset) == 1:
                # Directly assign mass to singleton focal elements
                prob_dist[subset] += mass
                continue

            # Process multi-element focal elements
            elements_in_subset = [s for s in self.singletons if s.issubset(subset)]

            # Exactly calculate total weight
            total_weight = Decimal(0)
            weight_per_element = {}
            for elem in elements_in_subset:
                w = weight_func(elem)
                weight_per_element[elem] = w
                total_weight += w

            # Distribute mass (fully preserve all decimals)
            if total_weight > SMALLEST_VALUE:  # Valid weight
                for elem in elements_in_subset:
                    w = weight_per_element[elem]
                    ratio = w / total_weight
                    prob_dist[elem] += mass * ratio
            else:
                # Equal distribution when total weight is zero (preserve full precision)
                share = mass / len(elements_in_subset)
                for elem in elements_in_subset:
                    prob_dist[elem] += share

        # Convert to dictionary with singleton string keys
        str_result = {}
        for key, value in prob_dist.items():
            str_key = next(iter(key)) if isinstance(key, frozenset) and len(key) == 1 else str(key)
            str_result[str_key] = value

        # Return normalized exact result
        return self.normalize_dist(str_result)

    def sudano_pr_pl_transform(self) -> HighPrecProbDist:
        """PrPl transform (likelihood function weighting)"""
        return self.sudano_general_transform(
            weight_func=lambda s: self.pl.get(s, Decimal(0))
        )

    def sudano_pr_bel_transform(self) -> HighPrecProbDist:
        """PrBel transform (belief function weighting)"""
        return self.sudano_general_transform(
            weight_func=lambda s: self.bel.get(s, Decimal(0))
        )

    def sudano_pr_npl_transform(self) -> HighPrecProbDist:
        """PrNpl transform (non-likelihood weighting)"""
        return self.sudano_general_transform(
            weight_func=lambda s: Decimal(1) - self.pl.get(s, Decimal(1))
        )

    def sudano_pr_apl_transform(self) -> HighPrecProbDist:
        """PraPl transform (average likelihood weighting)"""
        return self.sudano_general_transform(
            weight_func=lambda s: (self.bel.get(s, Decimal(0)) + self.pl.get(s, Decimal(0))) / Decimal(2)
        )

    # ======================= Evaluation metrics (fully preserve precision) =======================

    def log_with_zero(self, p: Decimal) -> Decimal:
        """Safely calculate log2(p), fully preserve calculation precision"""
        if p <= 0:
            return Decimal('-inf')
        try:
            # Use ln to calculate log2
            return p.ln() / Decimal(2).ln()
        except (InvalidOperation, OverflowError):
            return Decimal('-inf')

    def shannon_entropy(self, prob_dist: Dict[str, Decimal]) -> Decimal:
        """Shannon entropy calculation with full precision preservation"""
        entropy = Decimal(0)

        for element, prob in prob_dist.items():
            if prob <= 0:
                continue  # Ignore zero probability terms

            term = prob * self.log_with_zero(prob)
            if term != Decimal('-inf'):
                entropy -= term

        return entropy

    def max_entropy(self, n: int) -> Decimal:
        """Exact maximum entropy calculation"""
        if n <= 1:
            return Decimal(0)
        return Decimal(n).ln() / Decimal(2).ln()

    def pic(self, prob_dist: Dict[str, Decimal]) -> Decimal:
        """PIC metric calculation with full precision preservation"""
        n = len(prob_dist)
        if n == 0:
            return Decimal(0)

        h_max = self.max_entropy(n)
        if h_max <= 0:
            return Decimal(1) if n == 1 else Decimal(0)

        entropy = self.shannon_entropy(prob_dist)
        if h_max > 0:
            pic_value = 1 - entropy / h_max
        else:
            pic_value = Decimal(1) if entropy == 0 else Decimal(0)

        return pic_value

    # ======================= Result output =======================
    def sort_by_pic(self, results: Dict[str, Tuple[HighPrecProbDist, Decimal, Decimal]]) -> List[Tuple[str, Decimal]]:
        """
        Sort PIC results of transformation methods in descending order
        Return format: [(method_name, PIC_value), ...]
        """
        # Extract method names and corresponding PIC values
        pic_list = [(method_name, result[2]) for method_name, result in results.items()]
        # Sort by PIC value in descending order (Decimal type supports direct comparison)
        pic_list.sort(key=lambda x: x[1], reverse=True)
        return pic_list

    def print_sorted_pic(self, sorted_pic: List[Tuple[str, Decimal]]) -> None:
        """Format and print sorted PIC results (preserve 12-digit scientific notation)"""
        print("\n===== PIC values sorted (descending) =====")
        for i, (method, pic) in enumerate(sorted_pic, 1):
            # Use existing formatting method (12-digit scientific notation)
            formatted_pic = self.format_exact_value(pic)
            print(f"{i}. {method}: {formatted_pic}")

    def format_exact_value(self, value: Decimal, precision=12) -> str:
        """Exact value formatting, preserve 12-digit scientific notation"""
        if value == 0:
            return "0e+0"

        # Convert to scientific notation with 12 significant digits
        fmt = f".{precision - 1}e"  # 12 significant digits corresponds to `.11e` format
        s = format(value, fmt)

        # Adjust exponent representation (optional, ensure uniform format)
        if 'e+' in s:
            base, exp = s.split('e+')
            return f"{base}e+{int(exp)}"
        elif 'e-' in s:
            base, exp = s.split('e-')
            return f"{base}e-{int(exp)}"
        return s

    def create_results_table(self, results: Dict[str, Tuple[HighPrecProbDist, Decimal, Decimal]]) -> str:
        """Generate exact value table without losing any precision"""
        # Prepare column names
        elements = sorted(self.singleton_elements)
        columns = ["Transformation Method"] + elements + ["Shannon Entropy", "PIC"]

        # Prepare row data
        table_rows = []
        for method, (prob_dist, entropy, pic) in results.items():
            row = [method]

            # Add probability values for each element (fully preserve precision)
            for elem in elements:
                val = prob_dist.get(elem, Decimal(0))
                row.append(self.format_exact_value(val))

            # Add Shannon entropy and PIC (fully preserve precision)
            row.append(self.format_exact_value(entropy))
            row.append(self.format_exact_value(pic))
            table_rows.append(row)

        # Calculate column widths
        col_widths = []
        for i in range(len(columns)):
            max_width = max(len(str(row[i])) for row in [columns] + table_rows)
            col_widths.append(max_width + 2)  # Add 2 as buffer

        # Build table
        separator = '+' + '+'.join(['-' * w for w in col_widths]) + '+'
        header = '|' + '|'.join([col.center(col_widths[i]) for i, col in enumerate(columns)]) + '|'

        # Generate table
        table = [
            separator,
            header,
            separator,
        ]

        for row in table_rows:
            row_str = '|' + '|'.join([
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            ]) + '|'
            table.append(row_str)
            table.append(separator)

        return "\n".join(table)

    # ======================= Comprehensive evaluation interface =======================
    def evaluate_all_methods(self) -> Dict[str, Tuple[HighPrecProbDist, Decimal, Decimal]]:
        """Evaluate all transformation methods and return exact results and metrics"""
        methods = {
            "Pignistic": self.pignistic_transform,
            "Interval": self.robust_interval_transform,
            "Sudano_PrPl": self.sudano_pr_pl_transform,
            "Sudano_PrBel": self.sudano_pr_bel_transform,
            "Sudano_PrNpl": self.sudano_pr_npl_transform,
            "Sudano_PraPl": self.sudano_pr_apl_transform,
        }

        results = {}
        for name, method in methods.items():
            prob_dist = method()
            entropy_val = self.shannon_entropy(prob_dist)
            pic_val = self.pic(prob_dist)
            results[name] = (prob_dist, entropy_val, pic_val)

        return results


if __name__ == "__main__":
    # Define BPA (containing very small values)
    # bpa = {
    #     frozenset({"θ1"}): 3.072063548051e-11,
    #     frozenset({"θ2"}): 9.999999951827e-1,
    #     frozenset({"θ1", "θ2"}): 4.786593706112e-9
    # }

    bpa = {
        frozenset({"θ1"}): 1.604e-10,
        frozenset({"θ2"}):  9.999999975529e-1,
        frozenset({"θ1", "θ2"}): 2.287e-9
    }

    # Create high-precision processor
    processor = HighPrecisionEvidenceProcessor(bpa)
    # Evaluate all methods
    results = processor.evaluate_all_methods()
    # Generate and print exact results table
    print(processor.create_results_table(results))
    # Sort PIC results and print
    sorted_pic = processor.sort_by_pic(results)
    processor.print_sorted_pic(sorted_pic)