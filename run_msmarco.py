import warnings

warnings.filterwarnings("ignore")
from pufr.approaches.pufr import *
from pufr.approaches.baselines.fair import *
from pufr.approaches.baselines.convex_optimization import (
    rerank_convex_optimization_approach,
)
from pufr.get_candidate_data import get_ms_marco_fair_candidate_data
from pufr.evaluation import eval_msmarco
import time
import numpy as np

list_protected_attributes = [1]

experiment = "table"
assert experiment in ["table", "ablation", "trade-off-curve"]

if experiment == "table":
    experiment_range_pufr = [0, 2.5, 7]
    experiment_range_ablation = []
    experiment_range_fair = [0.7, 0.85]
    experiment_range_cvxopt = [0.8, 0.91]

elif experiment == "ablation":
    experiment_range_pufr = []
    experiment_range_ablation = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2]
    experiment_range_fair = []
    experiment_range_cvxopt = []

elif experiment == "trade-off-curve":
    experiment_range_pufr = (
        list(np.arange(0, 2, 0.2))
        + list(np.arange(2, 10, 0.5))
        + list(np.arange(10, 20, 2))
    )
    experiment_range_ablation = []
    experiment_range_fair = [0.2, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    experiment_range_cvxopt = [0.6, 0.7, 0.75, 0.775, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95, 0.97]

path_to_results_per_query = "results/msmarco_results_per_query_" + experiment + ".csv"
path_to_results = "results/msmarco_mean_results_" + experiment + ".csv"
path_to_reranked_candidates = "results/msmarco_ranked_candidates_" + experiment + ".csv"


candidate_data = get_ms_marco_fair_candidate_data()

# Rank documents wrt mean model scores
candidate_data = rerank_based_on_score_per_query(
    candidate_data, "mean_score", "original_rank"
)
num_queries = len(list(set(candidate_data.qid)))

results_df = pd.DataFrame()

############################# PUFR experiment #######################
runtimes = []
results_pufr = [pd.DataFrame()]
for i in experiment_range_pufr:
    start_time = time.time()

    candidate_data = rerank_pufr(
        candidate_data,
        list_protected_attributes=list_protected_attributes,
        std_name="std_score",
        mean_score_name="mean_score",
        alpha=i,
        ranking_name="pufr_" + str(i),
        mean_score_ranking="original_rank",
    )

    runtimes.append((time.time() - start_time) / num_queries)

    results_per_query = eval_msmarco(candidate_data, "pufr_" + str(i))
    results_per_query = results_per_query.reset_index().rename({"index": "qid"}, axis=1)
    results_per_query["alpha"] = i
    results_pufr.append(results_per_query)

results_pufr = pd.concat(results_pufr)
results_pufr["avg_runtime"] = np.mean(runtimes)
results_pufr["approach"] = "pufr"

results_df = pd.concat([results_df, results_pufr])


############################# Ablation experiment #######################
# Ablation study, correcting for one constant value instead of std
mean_std_lm = candidate_data["std_score"].mean()
candidate_data["score_mean"] = mean_std_lm

runtimes = []
results_ablation = [pd.DataFrame()]
for i in experiment_range_ablation:
    start_time = time.time()

    candidate_data = rerank_pufr(
        candidate_data,
        list_protected_attributes=list_protected_attributes,
        std_name="std_score",
        mean_score_name="score_mean",
        alpha=i,
        ranking_name="ablation_" + str(i),
        mean_score_ranking="original_rank",
    )

    runtimes.append((time.time() - start_time) / num_queries)

    results_per_query = eval_msmarco(candidate_data, "ablation_" + str(i))
    results_per_query["alpha"] = i
    results_per_query = results_per_query.reset_index()
    results_per_query = results_per_query.rename({"index": "qid"}, axis=1)
    results_ablation.append(results_per_query)
results_ablation = pd.concat(results_ablation)
results_ablation["avg_runtime"] = np.mean(runtimes)
results_ablation["approach"] = "ablation"

results_df = pd.concat([results_df, results_ablation])

###################### FA*IR baseline ##############

results_fa_ir = [pd.DataFrame()]
runtimes = []
for i in experiment_range_fair:
    start_time = time.time()

    candidate_data = rarank_FAIR_approach(
        candidate_data,
        ranking_score_name="mean_score",
        min_share=i,
        new_ranking_name="FA*IR_rank_" + str(i),
        protected_attribute=list_protected_attributes,
        num_candidates_per_query=1000,
    )

    runtimes.append((time.time() - start_time) / num_queries)
    results_per_query = eval_msmarco(candidate_data, "FA*IR_rank_" + str(i))
    results_per_query["alpha"] = i
    results_per_query = results_per_query.reset_index().rename({"index": "qid"}, axis=1)
    results_fa_ir.append(results_per_query)
    candidate_data.to_csv(path_to_reranked_candidates, index=False)

results_fa_ir = pd.concat(results_fa_ir)
results_fa_ir["avg_runtime"] = np.mean(runtimes)
results_fa_ir["approach"] = "fa*ir"

results_df = pd.concat([results_df, results_fa_ir])

###################### Convex optimization baseline ##############
rerank_top = 50

results_cvxopt = [pd.DataFrame()]
runtimes = []
for i in experiment_range_cvxopt:
    start_time = time.time()

    candidate_data = rerank_convex_optimization_approach(
        candidate_data,
        i,
        ranking_name="cvxopt_rank_" + str(i),
        top_k=rerank_top,
        ranking_score_name="mean_score",
    )

    runtimes.append((time.time() - start_time) / num_queries)
    results_per_query = eval_msmarco(candidate_data, "cvxopt_rank_" + str(i))
    results_per_query["alpha"] = i
    results_per_query = results_per_query.reset_index().rename({"index": "qid"}, axis=1)
    results_cvxopt.append(results_per_query)
    candidate_data.to_csv(path_to_reranked_candidates, index=False)

results_cvxopt = pd.concat(results_cvxopt)
results_cvxopt["avg_runtime"] = np.mean(runtimes)
results_cvxopt["approach"] = "cvxopt"

results_df = pd.concat([results_df, results_cvxopt])
results_df.to_csv(path_to_results_per_query, index=False)
candidate_data.to_csv(path_to_reranked_candidates, index=False)


print(
    results_df.groupby(["approach", "alpha"], as_index=True).mean()[
        ["ndcg_cut_10", "ndcg_cut_100", "nfairr10", "nfairr50", "avg_runtime"]
    ]
)
