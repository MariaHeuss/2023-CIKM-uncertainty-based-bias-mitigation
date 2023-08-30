from cvxopt import spmatrix, matrix, sparse
from cvxopt.glpk import ilp
import math
from pufr.evaluation import *


def solve_ilp(
    learned_scores,
    neutrality_scores,
    lower_interval_nfairr_score,
    top_k=None,
):
    """
    Solve the linear program with the disparate treatment constraint.

    Returns ranking matrix as numpy array
    """

    n = len(learned_scores)
    if top_k is None:
        top_k = n

    # initialize position-based exposure v
    position_bias = np.arange(1, (top_k + 1), 1)
    position_bias = 1 / np.log2(1 + position_bias)
    position_bias = np.reshape(position_bias, (1, top_k))

    # initialize position-based discount for nfairr
    nfairr_discount = np.arange(1, (top_k + 1), 1)
    nfairr_discount = 1 / nfairr_discount
    nfairr_discount = np.reshape(nfairr_discount, (1, top_k))

    ideal_fairr = sorted(neutrality_scores, reverse=True)
    learned_scores = np.asarray(learned_scores)
    learned_scores = learned_scores - learned_scores.min() + 1e-5
    learned_scores = learned_scores / learned_scores.max()
    neutrality_scores = np.asarray(neutrality_scores)

    # determine the ideal fairr value:
    ideal_fairr = np.sum(ideal_fairr * nfairr_discount)

    I = []
    J = []
    I2 = []
    J2 = []

    # set up indices for column and row constraints
    for j in range(n * top_k):
        J.append(j)

    for i in range(n):
        for j in range(top_k):
            J2.append(j * n + i)
            I.append(i)
            I2.append(j)

    learned_scores = np.reshape(learned_scores, (n, 1))
    neutrality_scores = np.reshape(neutrality_scores, (n, 1))

    # uv contains the product of position bias at each position with the
    # each item (flattened and negated).
    uv = learned_scores.dot(position_bias)

    # Set up the fairness constraint
    # nw contains the product of nfairr position discount at each position
    # with the neutrality score at each item (flattened and negated).
    nw = neutrality_scores.dot(nfairr_discount)
    nw = nw.flatten()

    # we define f and f_value to be used for the fairness constraint
    nw = np.reshape(nw, (1, n * top_k))
    f = matrix(-nw)
    f_value = -lower_interval_nfairr_score * ideal_fairr

    # Set up objectives

    uv = uv.flatten()
    # negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)

    # set up constraints x <= 1
    A = spmatrix(1.0, range(n * top_k), range(n * top_k))
    # set up constraints x >= 0
    A1 = spmatrix(-1.0, range(n * top_k), range(n * top_k))
    # set up constraints that sum(rows) <= 1
    M1 = spmatrix(1.0, I, J)
    # set up constraints sum(columns) <= 1
    M2 = spmatrix(1.0, I2, J)

    alpha = 0.99999  # we tolerate an error of 1e-5
    # set up constraints sum(columns)>alpha
    M3 = spmatrix(-1.0, I2, J)

    # values for x<=1
    a = matrix(1.0, (n * top_k, 1))
    # values for x >= 0
    a1 = matrix(0.0, (n * top_k, 1))
    # values for sums columns <= 1
    h1 = matrix(1.0, (n, 1))
    # values for sums rows <= 1
    h2 = matrix(1.0, (top_k, 1))
    # values for sums columns > alpha
    h3 = matrix(-alpha, (top_k, 1))

    # construct objective function
    c = matrix(uv)

    G = sparse([M1, M2, M3, A, A1, f])
    h = matrix([h2, h1, h3, a, a1, f_value])

    # Since we want a deterministic ranking policy, we use a inverse linear program
    (status, x) = ilp(
        c,
        G,
        h,
        spmatrix(0.0, I, J),
        matrix(0.0, (top_k, 1)),
        set(range(len(a))),
        set([]),
    )
    return x


def output_cvxopt_to_ranking(output):
    dim = int(math.sqrt(len(output)))
    output = np.array(output)
    output = output.reshape((dim, dim))
    rank_per_doc = [[i + 1 for i, x in enumerate(list(doc)) if x][0] for doc in output]
    return rank_per_doc


def rerank_convex_optimization_approach(
    dataframe,
    nfairr_threshold,
    ranking_score_name="mean_lm_score",
    ranking_name="cvxopt_rank",
    top_k=None,
):

    risk_data = rerank_based_on_score_per_query(
        dataframe, ranking_score_name, "original_rank"
    )
    dfs = []
    for query in set(dataframe.qid):
        query_df = dataframe.loc[dataframe["qid"] == query]
        if top_k is not None:
            not_to_rerank = query_df[query_df["original_rank"] > top_k]
            not_to_rerank[ranking_name] = not_to_rerank["original_rank"]
            query_df = query_df[query_df["original_rank"] <= top_k]

        learned_scores = list(query_df[ranking_score_name].values)
        neutrality_scores = list(query_df.protected_attribute.values)

        out = solve_ilp(learned_scores, neutrality_scores, nfairr_threshold)
        ranking = output_cvxopt_to_ranking(out)

        query_df[ranking_name] = ranking
        dfs.append(query_df)
        if top_k is not None:
            dfs.append(not_to_rerank)
    return pd.concat(dfs, axis=0)
