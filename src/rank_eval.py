import numpy as np
from scipy.stats import spearmanr,kendalltau

PGM_ranks =[[0, 7, 5, 6, 8, 4, 10, 3, 1, 2, 11, 9],
            [1, 7, 0, 5, 6, 8, 4, 10, 3, 2, 11, 9], 
            [2, 7, 5, 0, 6, 8, 4, 10, 3, 1, 11, 9], 
            [3, 7, 0, 5, 6, 8, 4, 10, 1, 2, 11, 9], 
            [4, 7, 5, 0, 6, 8, 10, 3, 1, 2, 11, 9], 
            [5, 7, 0, 6, 8, 4, 10, 3, 1, 2, 11, 9], 
            [6, 7, 5, 0, 8, 4, 10, 3, 1, 2, 11, 9], 
            [7, 0, 5, 6, 8, 4, 10, 3, 1, 2, 11, 9], 
            [8, 7, 0, 5, 6, 4, 10, 3, 1, 2, 11, 9], 
            [9, 7, 0, 5, 8, 6, 4, 10, 3, 1, 2, 11], 
            [10, 7, 0, 5, 6, 8, 4, 3, 1, 2, 11, 9], 
            [11, 7, 0, 5, 6, 8, 4, 10, 3, 1, 2, 9]]

finetune_ranks = [[j for j in range(12)] for i in range(12)]

print(finetune_ranks)

spearmanr_results = []
kendalltau_results = []

for i in range(12):

    expected_ranking = PGM_ranks[i]
    actual_ranking = finetune_ranks[i]

    # 值在 -1 和 1 之间，+1 表示完全一致的排序，-1 表示完全相反的排序，0表示没有关联。
    # p值表示相关性是否显著，通常如果p值小于0.05，可以认为相关性是显著的。
    rho, p1 = spearmanr(expected_ranking, actual_ranking)

    # print(f"Spearman等级相关系数: {rho}, p值: {p1}")

    tau, p2 = kendalltau(expected_ranking, actual_ranking)

    # print(f"Kendall Tau系数: {tau}, p值: {p2}")

    spearmanr_results.append(rho)
    kendalltau_results.append(tau)

print(spearmanr_results)
print(kendalltau_results)

spearmanr_avg, kendalltau_avg= 0.0,0.0
for i in range(12):
    spearmanr_avg+=spearmanr_results[i]
    kendalltau_avg+=kendalltau_results[i]
spearmanr_avg, kendalltau_avg = spearmanr_avg/12, kendalltau_avg/12
print(spearmanr_avg, kendalltau_avg)