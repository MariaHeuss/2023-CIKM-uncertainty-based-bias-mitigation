import numpy as np


def median_intersection_number(
    candidate_data, list_of_queries, size_of_errorbounds=1, num_ranks=100, mode="median"
):
    intersections_per_rank = {rank: [] for rank in range(num_ranks)}
    for query in list_of_queries:
        query_data = candidate_data[candidate_data.qid == query]
        means = np.array(query_data.mean_lm_score)
        stds = np.array(query_data.std_lm_score)
        risk_aware_upper = means + size_of_errorbounds * stds
        risk_aware_lower = means - size_of_errorbounds * stds
        for rank, mean in enumerate(means[:num_ranks]):
            lower_intersection_index, upper_intersection_index = rank, rank
            while (
                risk_aware_lower[rank] <= risk_aware_upper[lower_intersection_index]
                and lower_intersection_index < len(risk_aware_lower) - 1
            ):
                lower_intersection_index += 1

            while (
                risk_aware_upper[rank] >= risk_aware_lower[upper_intersection_index]
                and upper_intersection_index > 0
            ):
                upper_intersection_index -= 1
            intersections_per_rank[rank].append(
                lower_intersection_index - upper_intersection_index
            )
    median_intersection_number = [
        np.median(intersections_per_rank[rank]) for rank in intersections_per_rank
    ]
    return median_intersection_number
