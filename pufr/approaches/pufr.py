from pufr.re_ranking import rerank_based_on_score_per_query
import numpy as np
import pandas as pd


def rerank_pufr(
    candidate_data,
    list_protected_attributes,
    std_name="std_lm_score",
    mean_score_name="mean_lm_score",
    mean_score_ranking="lm_rank",
    alpha=1,
    ranking_name="pufr",
):

    risk_data = candidate_data.sort_values(
        by=["qid", mean_score_name], ascending=True
    ).reset_index(drop=True)

    risk_data["raw_pessimistic_score"] = (
        risk_data[mean_score_name] - alpha * risk_data[std_name]
    )
    risk_data["raw_optimistic_score"] = (
        risk_data[mean_score_name] + alpha * risk_data[std_name]
    )

    # To prevent inner group swapping each pessimistic score should be bigger than
    # the pessimistic score of each document ranked lower in the original ranking
    # (See line 7 in Algorithm 1)
    pess_score = []
    for name, group in risk_data.groupby(["qid"]):
        group["pessimistic_score"] = group["raw_pessimistic_score"].cummax() - (
            group[mean_score_ranking] * 1e-6
        )
        pess_score.append(group)
    risk_data = pd.concat(pess_score)

    # Similarly for the optimistic score. (Line 3 in Algorithm 1)
    risk_data = risk_data.sort_values(
        by=["qid", mean_score_name], ascending=False
    ).reset_index(drop=True)
    opt_score = []
    for name, group in risk_data.groupby(["qid"]):
        group["optimistic_score"] = group["raw_optimistic_score"].cummin() - (
            group[mean_score_ranking] * 1e-6
        )
        opt_score.append(group)
    risk_data = pd.concat(opt_score)

    risk_data["adjusted_score"] = np.where(
        risk_data["protected_attribute"].isin(list_protected_attributes),
        risk_data["optimistic_score"],
        risk_data["pessimistic_score"],
    )

    risk_data = rerank_based_on_score_per_query(
        risk_data, "adjusted_score", ranking_name
    )
    return risk_data
