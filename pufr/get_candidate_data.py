import numpy as np
import pandas as pd


def get_ms_marco_fair_candidate_data(determine_std=False):
    folder_name = "data/msmarco_fair_data/"
    path_to_candidate_list = folder_name + "collection_neutralityscores.tsv"
    path_to_qrels = folder_name + "qrels.dev.tsv"

    if determine_std:
        path_to_uncertainty_scores = (
            folder_name + "GENDERtest-run-full-rerank.uncertainty.txt"
        )
        uncertainty_data = pd.read_csv(
            path_to_uncertainty_scores, delimiter=" ", header=None
        )
        uncertainty_data = uncertainty_data.rename(columns={0: "qid", 2: "docid"})
        uncertainty_data = uncertainty_data.rename(
            columns={i: "prediction_" + str(i - 3) for i in range(4, 154)}
        )
        uncertainty_data = uncertainty_data.set_index(["qid", "docid"])
        uncertainty_data = uncertainty_data.drop(columns=[1, 3, 154])

        uncertainty_data["mean_score"] = uncertainty_data.mean(axis=1)
        uncertainty_data["std_score"] = uncertainty_data.std(axis=1)
        uncertainty_data = uncertainty_data[["mean_score", "std_score"]]
        uncertainty_data.reset_index().to_csv(
            folder_name + "msmarco_pointwise_uncertainty.csv", index=False
        )
    else:
        uncertainty_data = pd.read_csv(
            folder_name + "msmarco_pointwise_uncertainty.csv",
            index_col=["qid", "docid"],
        )

    candidate_data = pd.read_csv(
        path_to_candidate_list,
        delimiter="\t",
        header=None,
        names=["docid", "protected_attribute"],
    )
    candidate_data = candidate_data.set_index("docid")
    candidate_data["protected_attribute"] = np.where(
        candidate_data["protected_attribute"] > 0.5, 1, 0
    )

    qrels_data = pd.read_csv(
        path_to_qrels, delimiter="\t", header=None, names=["qid", "?", "docid", "rel"]
    )
    qrels_data = qrels_data.drop("?", axis=1)

    uncertainty_data = uncertainty_data.reset_index()
    uncertainty_data["protected_attribute"] = uncertainty_data["docid"].map(
        dict(candidate_data["protected_attribute"])
    )

    uncertainty_data = uncertainty_data.merge(
        qrels_data, how="left", on=["qid", "docid"]
    )
    uncertainty_data = uncertainty_data.fillna(0)
    return uncertainty_data
