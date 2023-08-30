import pandas as pd


def rerank_based_on_score_per_query(data_frame, name_of_score, name_of_ranking):
    dfs = []
    for query in set(data_frame.qid.values):
        data_frame_query = data_frame[data_frame["qid"] == query]
        data_frame_query[name_of_ranking] = (
            data_frame_query[[name_of_score]]
            .apply(tuple, axis=1)
            .rank(method="first", ascending=False)
            .astype(int)
        )
        dfs.append(data_frame_query)
    return pd.concat(dfs, axis=0)
