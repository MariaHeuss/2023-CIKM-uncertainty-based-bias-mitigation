import math
from queue import PriorityQueue
from pufr.evaluation import *


def cum_distribution_binomial(x, n, p):
    return sum(
        [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(x + 1)]
    )


def fair_criterium(
    num_protected,
    top_k,
    minimal_proportion,
    significance=0.1,
    precomputed_min_num_prot=None,
):
    if precomputed_min_num_prot is not None:
        return num_protected >= precomputed_min_num_prot[top_k]
    return (
        cum_distribution_binomial(num_protected, top_k, minimal_proportion)
        > significance
    )


def precompute_min_num_protected(minimal_proportion, num_ranks, significance=0.1):
    # iterative implementation that saves computing time on the factorials
    min_num_protected = []
    previous_iteration = [1]

    rank = 1
    while rank <= num_ranks + 1:
        previous_iteration = [
            prev * (1 - minimal_proportion) * (rank + 1) / (rank + 1 - i)
            for i, prev in enumerate(previous_iteration)
        ]
        if not sum(previous_iteration) > significance:
            tau = len(previous_iteration)
            previous_iteration.append(
                math.comb(rank, tau)
                * (minimal_proportion ** tau)
                * ((1 - minimal_proportion) ** (rank - tau))
            )
        min_num_protected.append(len(previous_iteration) - 1)
        rank += 1
    return min_num_protected


def fair_rerank_data_from_one_query(
    data,
    ranking_score_name,
    min_share,
    protected_attribute_column_name="protected_attribute",
    protected_attribute=1,
    precomputed_min_num_prot=None,
):
    data = rerank_based_on_score_per_query(data, ranking_score_name, "temp_rank")

    # Create priority queues for the two groups
    protected_priority_queue = PriorityQueue()
    non_protected_priority_queue = PriorityQueue()
    protected_attributes = 0
    for index, row in data.iterrows():
        if row[protected_attribute_column_name] == protected_attribute:
            protected_priority_queue.put((-1 * row[ranking_score_name], index))
            protected_attributes += 1
        else:
            non_protected_priority_queue.put((-1 * row[ranking_score_name], index))

    protected_priority_queue.put((float("inf"), None))
    non_protected_priority_queue.put((float("inf"), None))

    # Create ranking based on FA*IR min-share rule
    num_protected_attributes = 0
    prot = protected_priority_queue.get()
    non_prot = non_protected_priority_queue.get()
    for rank in range(1, len(data) + 1):
        if (
            fair_criterium(
                num_protected_attributes,
                rank,
                min_share,
                precomputed_min_num_prot=precomputed_min_num_prot,
            )
            and non_prot[0] < prot[0]
            or num_protected_attributes == protected_attributes
        ):
            data.at[non_prot[1], "temp_rank"] = rank
            non_prot = non_protected_priority_queue.get()
        else:
            data.at[prot[1], "temp_rank"] = rank
            num_protected_attributes += 1
            prot = protected_priority_queue.get()

    return data["temp_rank"]  # Returns a series with new rank for each index


def rarank_FAIR_approach(
    dataframe,
    ranking_score_name,
    min_share,
    new_ranking_name,
    protected_attribute_column_name="protected_attribute",
    protected_attribute=1,
    num_candidates_per_query=1000,
):
    dfs = []
    precomputed_min_prot = precompute_min_num_protected(
        minimal_proportion=min_share, num_ranks=num_candidates_per_query
    )
    for query in set(dataframe.qid):
        query_df = dataframe.loc[dataframe["qid"] == query]

        new_ranking = fair_rerank_data_from_one_query(
            data=query_df,
            ranking_score_name=ranking_score_name,
            min_share=min_share,
            protected_attribute_column_name=protected_attribute_column_name,
            protected_attribute=protected_attribute,
            precomputed_min_num_prot=precomputed_min_prot,
        )
        query_df[new_ranking_name] = new_ranking

        dfs.append(query_df)
    return pd.concat(dfs, axis=0)
