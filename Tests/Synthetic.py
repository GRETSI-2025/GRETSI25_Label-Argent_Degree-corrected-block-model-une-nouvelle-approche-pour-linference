
import random
import matplotlib.pyplot as plt
import OtrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import pysbm
import time
import numpy as np
import pandas as pd
from Utils import DC_BM
import csv
import matplotlib.cm as cm
def Generate_G(p_in,p_out):
    n = 100
    r = 2
    s = n // r
    #z= np.random.zipf(2.5, n)
    #z = z[z < 100]
    z=np.random.uniform(0.05, 1,n)
    z=z/np.mean(z)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    assert n % r == 0, "n doit être divisible par r pour des groupes de taille égale."
    shuffle_indices = np.random.permutation(n)
    groups = [list(shuffle_indices[i * s: (i + 1) * s]) for i in range(r)]
    for group, community in zip(groups, range(r)):
        for node in group:
            G.nodes[node]['community'] = community

    for i in range(n):
        for j in range(n):
            if j < i:
                if G.nodes[i]['community'] == G.nodes[j]['community'] and random.random() < z[i]*z[j]*p_in:
                    G.add_edge(i, j)
                if G.nodes[i]['community'] != G.nodes[j]['community'] and random.random() < z[i]*z[j]*p_out:
                    G.add_edge(i, j)
    return G
def synt_test():
    r=2
    G=Generate_G(0.8, 0.4)
    # colors = cm.rainbow(np.linspace(0, 1, r))
    # community_color_map = {i: colors[i] for i in range(r)}
    # node_colors = [community_color_map[G.nodes[node]['community']] for node in G.nodes()]
    #
    # # Dessiner le graphe avec les couleurs des communautés
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G, seed=42)  # Position des nœuds
    #
    # nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, edge_color="gray")
    #
    # plt.title("Graph avec coloration par communautés")
    # plt.show()
    labels = [G.nodes[v]['community'] for v in sorted(G.nodes)]
    EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,init_partition=labels)

    print(normalized_mutual_info_score(labels, EM_partition))
    X = nx.adjacency_matrix(G, nodelist=sorted(G.nodes))

    w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r,numTrials=10,init_method="SVCA")

    print(normalized_mutual_info_score(labels, v_best))

    # z=40
    # nbr_tests=100
    # results_list = []  # Liste pour stocker les résultats
    # for z_out in range(25,z+1):
    #     results = {
    #         "OtrisymNMF": {"NMI": [], "Time": []},
    #         "KL_G": {"NMI": [], "Time": []},
    #         "KL_EM": {"NMI": [], "Time": []},
    #         "MHA250k": {"NMI": [], "Time": []},
    #         "KL_EM_initsol": {"NMI": [], "Time": []}
    #     }
    #     for test in range(nbr_tests):
    #         G=Generate_GN(z,z_out)
    #         labels = [G.nodes[v]['community'] for v in sorted(G.nodes)]
    #         r=max(labels)+1
    #
    #
    #         # # KL_G
    #         start_time=time.time()
    #         KLG_partition=DC_BM(G, r,pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,pysbm.KarrerInference, numTrials=10)
    #         end_time=time.time()
    #         NMI=normalized_mutual_info_score(labels,KLG_partition)
    #         results["KL_G"]["NMI"].append(NMI)
    #         results["KL_G"]["Time"].append(end_time-start_time)
    #         print(NMI)
    #         #
    #         # KL_EM
    #         start_time = time.time()
    #         EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10)
    #         end_time = time.time()
    #         NMI = normalized_mutual_info_score(labels, EM_partition)
    #         results["KL_EM"]["NMI"].append(NMI)
    #         results["KL_EM"]["Time"].append(end_time-start_time)
    #         print(NMI)
    #         #
    #         # MHA50
    #         start_time = time.time()
    #         MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceTenK, numTrials=10)
    #         end_time = time.time()
    #         NMI = normalized_mutual_info_score(labels, MHA_partition)
    #         results["MHA250k"]["NMI"].append(NMI)
    #         results["MHA250k"]["Time"].append(end_time - start_time)
    #         print(NMI)
    #
    #         #OtrisymNMF
    #
    #         X = nx.adjacency_matrix(G, nodelist=sorted(G.nodes))
    #         start_time = time.time()
    #         w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r,numTrials=10,init_method="SVCA")
    #         end_time = time.time()
    #         init_time=end_time - start_time
    #         NMI = normalized_mutual_info_score(labels, v_best)
    #         results["OtrisymNMF"]["NMI"].append(NMI)
    #         results["OtrisymNMF"]["Time"].append(end_time - start_time)
    #         print(NMI)
    #
    #         #KL_EM_initsol
    #
    #         start_time = time.time()
    #         EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=1,init_partition=labels)
    #         end_time = time.time()
    #         NMI = normalized_mutual_info_score(labels, EM_partition)
    #         results["KL_EM_initsol"]["NMI"].append(NMI)
    #         results["KL_EM_initsol"]["Time"].append(end_time-start_time+ init_time)
    #         print(NMI)
    #
    #     # Calcul des statistiques
    #     for algo, data in results.items():
    #         results_list.append({
    #             "Algorithm": algo,
    #             "z_out": z_out,
    #             "NMI Mean": np.mean(data["NMI"]),
    #             "NMI Std": np.std(data["NMI"], ddof=1),
    #             "Time Mean (s)": np.mean(data["Time"]),
    #             "Time Std (s)": np.std(data["Time"], ddof=1)
    #         })
    #         print(
    #             f"Algorithm: {algo}, z_out: {z_out}, NMI Mean: {np.mean(data['NMI'])}, NMI Std: {np.std(data['NMI'], ddof=1)}, Time Mean (s): {np.mean(data['Time'])}, Time Std (s): {np.std(data['Time'], ddof=1)}")
    #
    #     with open("GN_test_results2.csv", "w", newline='') as csvfile:
    #         fieldnames = ["Algorithm", "z_out", "NMI Mean", "NMI Std", "Time Mean (s)", "Time Std (s)"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         writer.writerows(results_list)
    #     # Sauvegarde dans un fichier CSV




if __name__ == "__main__":
    synt_test()
