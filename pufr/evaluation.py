import numpy as np
from pufr.re_ranking import *
import pytrec_eval
import collections


def get_fairr_at_all_k(dataframe, name_ranking_column):
    fairr_per_query = {}
    for query in set(dataframe.qid.values):
        query_df = dataframe.loc[dataframe["qid"] == query]
        sum_neutrality_scores = np.array(
            query_df.groupby(query_df[name_ranking_column]).sum()["protected_attribute"]
        )
        discount = np.array(
            [1 / rank for rank in range(1, len(sum_neutrality_scores) + 1)]
        )
        cum_fairr_k = (sum_neutrality_scores * discount).cumsum()
        fairr_per_query[query] = cum_fairr_k
    return fairr_per_query


def get_fairr(dataframe, name_ranking_column, top_k, normalize=False):
    if top_k == None:
        top_k = 100000
    fairr_per_query = {}

    # create an "ideal" ranking that is sorted based on neutrality score
    nfairr_perfect_ranking = rerank_based_on_score_per_query(
        dataframe, "protected_attribute", "ideal_nfairr_rank"
    )

    cum_ideal_fairr_k = get_fairr_at_all_k(nfairr_perfect_ranking, "ideal_nfairr_rank")

    cum_fairr_k = get_fairr_at_all_k(dataframe, name_ranking_column)

    for query in set(dataframe.qid.values):
        t_k = min(len(cum_fairr_k[query]), top_k)
        if normalize:
            fairr_per_query[query] = (
                cum_fairr_k[query][t_k - 1] / cum_ideal_fairr_k[query][t_k - 1]
            )
        else:
            fairr_per_query[query] = cum_fairr_k[query][t_k - 1]
    return fairr_per_query


def eval_with_trec(candidate_data, ranking_name, path_to_qrels):
    with open(path_to_qrels, "r") as f_qrel:
        qrel_dict = pytrec_eval.parse_qrel(f_qrel)
    candidate_data["inverted_rank"] = (1 / candidate_data[ranking_name]).fillna(0)

    run = collections.defaultdict(dict)
    for _, row in candidate_data.iterrows():
        query_id, object_id, score = (
            str(int(row["qid"])),
            str(int(row["docid"])),
            row["inverted_rank"],
        )

        assert object_id not in run[query_id]
        run[query_id][object_id] = float(score)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel_dict,
        {
            "map",
            "ndcg",
            "ndcg_cut_5",
            "ndcg_cut_10",
            "ndcg_cut_100",
            "ndcg_cut_1000",
            "recip_rank",
        },
    )
    results = evaluator.evaluate(run)
    results = pd.DataFrame.from_dict(results, orient="index")
    return results


def eval_msmarco(candidate_data, ranking_name):
    path_to_qrels = "data/msmarco_fair_data/qrels.dev.tsv"
    results_per_query = eval_with_trec(
        candidate_data, path_to_qrels=path_to_qrels, ranking_name=ranking_name
    )
    results_per_query["nfairr"] = get_fairr(
        candidate_data, ranking_name, None, normalize=True
    ).values()
    results_per_query["nfairr5"] = get_fairr(
        candidate_data, ranking_name, 5, normalize=True
    ).values()
    results_per_query["nfairr10"] = get_fairr(
        candidate_data, ranking_name, 10, normalize=True
    ).values()
    results_per_query["nfairr50"] = get_fairr(
        candidate_data, ranking_name, 50, normalize=True
    ).values()
    results_per_query["nfairr100"] = get_fairr(
        candidate_data, ranking_name, 100, normalize=True
    ).values()
    results_per_query["fairr"] = get_fairr(candidate_data, ranking_name, None).values()
    results_per_query["fairr5"] = get_fairr(candidate_data, ranking_name, 5).values()
    results_per_query["fairr10"] = get_fairr(candidate_data, ranking_name, 10).values()
    results_per_query["fairr50"] = get_fairr(candidate_data, ranking_name, 50).values()
    results_per_query["fairr100"] = get_fairr(
        candidate_data, ranking_name, 100
    ).values()
    return results_per_query
