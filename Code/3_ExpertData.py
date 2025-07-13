import numpy as np


class FailureRateFusion:
    def __init__(self):
        """Initialize failure rate fusion model"""
        self.expert_scores_list = []  # Store multiple sets of expert scores
        self.expert_intervals_list = []  # Store multiple sets of expert failure rate intervals (with associated score indices)
        self.num_experts = 0
        self.num_levels = 0
        self.correlation_matrix_list = []  # Store multiple sets of correlation coefficient matrices
        self.weights_list = []  # Store multiple sets of expert weights
        self.final_intervals_list = []  # Store multiple sets of final failure rate intervals

    def input_expert_scores(self, scores):
        """Input a set of expert scores and return the score group index"""
        scores_array = np.array(scores)
        self.expert_scores_list.append(scores_array)
        self.num_experts, self.num_levels = scores_array.shape
        # Initialize correlation matrix and weights for the corresponding score group (to avoid subsequent index out of bounds)
        self.correlation_matrix_list.append(None)
        self.weights_list.append(None)
        return len(self.expert_scores_list) - 1  # Return current score group index

    def input_expert_intervals(self, intervals, scores_index):
        """Input a set of failure rate intervals, associate with score group index, and return the interval group index"""
        intervals_array = np.array(intervals)
        scores = self.expert_scores_list[scores_index]
        if intervals_array.shape[0] != scores.shape[0]:
            raise ValueError(f"Number of experts does not match: interval group({intervals_array.shape[0]}) vs score group({scores.shape[0]})")

        self.expert_intervals_list.append({
            'intervals': intervals_array,
            'scores_index': scores_index
        })
        self.final_intervals_list.append(None)  # Initialize final interval
        return len(self.expert_intervals_list) - 1  # Return current interval group index

    def calculate_correlation_matrix(self, scores_index):
        """Calculate the correlation coefficient matrix for the specified score group"""
        scores = self.expert_scores_list[scores_index]
        correlation_matrix = np.zeros((self.num_experts, self.num_experts))
        expectations = np.mean(scores, axis=1)
        variances = np.var(scores, axis=1)

        for i in range(self.num_experts):
            for j in range(self.num_experts):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    covariance = np.mean((scores[i] - expectations[i]) * (scores[j] - expectations[j]))
                    if variances[i] * variances[j] == 0:
                        correlation_matrix[i, j] = 0.0
                    else:
                        correlation_matrix[i, j] = covariance / np.sqrt(variances[i] * variances[j])

        self.correlation_matrix_list[scores_index] = correlation_matrix
        return correlation_matrix

    def calculate_weights(self, scores_index):
        """Calculate expert weights for the specified score group"""
        if self.correlation_matrix_list[scores_index] is None:
            self.calculate_correlation_matrix(scores_index)
        correlation_matrix = self.correlation_matrix_list[scores_index]

        sum_correlations = np.sum(correlation_matrix, axis=1) - 1  # Subtract self-correlation
        total_sum = np.sum(sum_correlations)

        if total_sum == 0:
            weights = np.ones(self.num_experts) / self.num_experts  # Equal distribution
        else:
            weights = sum_correlations / total_sum  # Normalization

        self.weights_list[scores_index] = weights
        return weights

    def calculate_weighted_interval(self, intervals_index):
        """Calculate the final fused interval for the specified interval group"""
        interval_info = self.expert_intervals_list[intervals_index]
        intervals = interval_info['intervals']
        scores_index = interval_info['scores_index']

        if self.weights_list[scores_index] is None:
            self.calculate_weights(scores_index)
        weights = self.weights_list[scores_index]

        # Weighted fusion of lower and upper bounds
        lower = np.sum(weights * intervals[:, 0])
        upper = np.sum(weights * intervals[:, 1])
        self.final_intervals_list[intervals_index] = np.array([lower, upper])
        return self.final_intervals_list[intervals_index]

    def get_result(self, intervals_index=None):
        """Get results for a single or all interval groups"""
        if intervals_index is not None:
            if not (0 <= intervals_index < len(self.expert_intervals_list)):
                raise IndexError(f"Interval group index {intervals_index} is out of range")
            self.calculate_weighted_interval(intervals_index)
            info = self.expert_intervals_list[intervals_index]
            return {
                'expert_scores': self.expert_scores_list[info['scores_index']],
                'expert_intervals': info['intervals'],
                'correlation_matrix': self.correlation_matrix_list[info['scores_index']],
                'weights': self.weights_list[info['scores_index']],
                'final_interval': self.final_intervals_list[intervals_index]
            }
        else:
            return {
                'all_results': [self.get_result(i) for i in range(len(self.expert_intervals_list))]
            }

    def print_summary(self, intervals_index=None):
        """Print results for a single or all interval groups"""
        if intervals_index is not None:
            result = self.get_result(intervals_index)
            # print(f"\n===== Fused results for basic event {intervals_index + 1} =====")
            # print("Expert score data:")
            # for i, scores in enumerate(result['expert_scores']):
            #     print(f"Expert {i + 1}: {[round(x, 4) for x in scores]}")
            #
            # print("\nExpert failure rate intervals:")
            # for i, (lower, upper) in enumerate(result['expert_intervals']):
            #     print(f"Expert {i + 1}: [{lower:.5e}, {upper:.5e}]")
            #
            # print("\nCorrelation coefficient matrix:")
            # print(np.round(result['correlation_matrix'], 4))
            #
            # print("\nExpert weights:")
            # for i, w in enumerate(result['weights']):
            #     print(f"Expert {i + 1}: {w:.4f}")

            print("\nFinal fused failure rate interval:", f"[{result['final_interval'][0]:.5e}, {result['final_interval'][1]:.5e}]")
        else:
            for i in range(len(self.expert_intervals_list)):
                self.print_summary(i)


if __name__ == "__main__":
    model = FailureRateFusion()

    # 1. Input three sets of expert scores (corresponding to three basic events)
    WTBFailure_scores = [
        [0.08, 0.22, 0.58, 0.12],
        [0.12, 0.18, 0.57, 0.13],
        [0.07, 0.28, 0.53, 0.12],
        [0.11, 0.24, 0.54, 0.11]
    ]
    heat_idx = model.input_expert_scores(WTBFailure_scores)

    model.input_expert_intervals([  # WTB failure intervals
        [2.0e-6, 3.0e-5],
        [3.0e-6, 4.0e-5],
        [2.5e-6, 3.5e-5],
        [4.0e-6, 4.5e-5]
    ], heat_idx)

    POBFailure_scores = [
        [0.07, 0.23, 0.59, 0.11],
        [0.11, 0.19, 0.58, 0.12],
        [0.06, 0.27, 0.54, 0.13],
        [0.10, 0.23, 0.55, 0.12]
    ]
    pob_idx = model.input_expert_scores(POBFailure_scores)

    model.input_expert_intervals([  # POB failure intervals
        [1.5e-6, 2.5e-5],
        [2.0e-6, 3.0e-5],
        [2.2e-6, 3.2e-5],
        [3.5e-6, 4.0e-5]
    ], pob_idx)

    model.print_summary(0)
    model.print_summary(1)


    # # 1. Input three sets of expert scores (corresponding to three basic events)
    # heat_scores = [  # High temperature environment: 4 experts × 4 levels
    #     [0.10, 0.60, 0.25, 0.05],
    #     [0.15, 0.55, 0.25, 0.05],
    #     [0.05, 0.65, 0.25, 0.05],
    #     [0.10, 0.60, 0.25, 0.05]
    # ]
    # heat_idx = model.input_expert_scores(heat_scores)
    #
    # overcharge_scores = [  # Overcharge: 4 experts × 4 levels
    #     [0.05, 0.25, 0.60, 0.10],
    #     [0.10, 0.20, 0.60, 0.10],
    #     [0.05, 0.30, 0.55, 0.10],
    #     [0.10, 0.25, 0.55, 0.10]
    # ]
    # overcharge_idx = model.input_expert_scores(overcharge_scores)
    #
    # puncture_scores = [  # External puncture short circuit: 4 experts × 4 levels
    #     [0.05, 0.15, 0.60, 0.20],
    #     [0.05, 0.20, 0.55, 0.20],
    #     [0.10, 0.15, 0.55, 0.20],
    #     [0.05, 0.20, 0.55, 0.20]
    # ]
    # puncture_idx = model.input_expert_scores(puncture_scores)
    #
    # # 2. Input three sets of failure rate intervals (associated with corresponding score groups)
    # model.input_expert_intervals([  # High temperature environment intervals
    #     [1.0e-6, 2.0e-5],
    #     [2.5e-6, 3.5e-5],
    #     [2.0e-6, 4.0e-5],
    #     [5.0e-6, 8.0e-5]
    # ], heat_idx)
    #
    # model.input_expert_intervals([  # Overcharge intervals
    #     [1.0e-6, 5.0e-5],
    #     [1.0e-6, 8.0e-6],
    #     [4.0e-6, 7.0e-6],
    #     [5.0e-6, 6.0e-6]
    # ], overcharge_idx)
    #
    # model.input_expert_intervals([  # External puncture short circuit intervals
    #     [9.5e-6, 2.0e-5],
    #     [3.0e-5, 4.0e-5],
    #     [1.0e-5, 5.0e-5],
    #     [2.0e-5, 4.0e-5]
    # ], puncture_idx)
    #
    # model.print_summary(0)  # High temperature environment
    # model.print_summary(1)  # Overcharge
    # model.print_summary(2)  # External puncture short circuit