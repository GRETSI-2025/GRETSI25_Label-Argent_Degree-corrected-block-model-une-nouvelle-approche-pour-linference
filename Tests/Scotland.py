import pysbm
import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pylab as pl

# Chemin vers votre fichier .net
file_path = "../Data/Scotland.net"

# Charger le graphe en utilisant la fonction de lecture Pajek de NetworkX
graph = nx.read_pajek(file_path)

# Convertir en un graph NetworkX standard (optionnel)
graph = nx.Graph(graph)  # ou nx.DiGraph(G) pour un graphe orienté
labels = graph.nodes  # Insérez ici vos données de labels
clusters = np.ones(len(labels), dtype=int)
clusters[:108]=0
maxiter=10
# Afficher des informations de base sur le graphe
best_standard_partition = pysbm.NxPartition(
    graph=graph,
    number_of_blocks=2)
best_degree_partition= pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2,
        representation=best_standard_partition.get_representation())
obj_max_st=-float('inf')
obj_max_d=-float('inf')
model_degree=False
# partition de départ
for itt in range(maxiter):

    standard_partition = pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2)
    degree_corrected_partition = pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2,
        representation=standard_partition.get_representation())
    standard_objective_function = pysbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)
    degree_corrected_objective_function = pysbm.DegreeCorrectedUnnormalizedLogLikelyhood(is_directed=False)

    standard_inference = pysbm.MetropolisInference(graph, standard_objective_function, standard_partition)
    degree_corrected_inference = pysbm.MetropolisInference(graph, degree_corrected_objective_function, degree_corrected_partition)

    standard_inference.infer_stochastic_block_model(10000)
    degree_corrected_inference.infer_stochastic_block_model(10000)
    obj_value_s = standard_objective_function.calculate(standard_partition)
    obj_value_d = degree_corrected_objective_function.calculate(degree_corrected_partition)
    if obj_value_s > obj_max_st:
        best_standard_partition=standard_partition
        obj_max_st = obj_value_s

    if obj_value_d > obj_max_d:
        best_degree_partition = degree_corrected_partition
        obj_max_d = obj_value_d


essaid=[best_degree_partition.get_block_of_node(node) for node in graph]
nmi_score_d = normalized_mutual_info_score(clusters,essaid)
print("degree correction")
print("Normalized Mutual Information degree (NMI) :", nmi_score_d)
# Calculer la modularité
group=[]
group.append([node  for node in graph if best_degree_partition.get_block_of_node(node) == 0])
group.append([node  for node in graph if best_degree_partition.get_block_of_node(node) == 1])
modularity_value = nx.algorithms.community.modularity(graph, group)

print("modularity",modularity_value)
print(essaid)
print(group)
arretes_groupe1 = [
    (u, v) for u, v in graph.edges() if u in group[0] and v in group[0]
]
arretes_groupe2 = [
    (u, v) for u, v in graph.edges() if u in group[1] and v in group[1]
]
print("le nombre d'arrêtes intergroupe", len(arretes_groupe1)+len(arretes_groupe2))

essai=[best_standard_partition.get_block_of_node(node) for node in graph]
nmi_score = normalized_mutual_info_score(clusters,essai)
print("standard")
print("Normalized Mutual Information standard (NMI) :", nmi_score)
# Calculer la modularité
group=[]
group.append([node  for node in graph if best_standard_partition.get_block_of_node(node) == 0])
group.append([node  for node in graph if best_standard_partition.get_block_of_node(node) == 1])
modularity_value = nx.algorithms.community.modularity(graph, group)
print("modularity",modularity_value)
print(essai)
print(group)
arretes_groupe1 = [
    (u, v) for u, v in graph.edges() if u in group[0] and v in group[0]
]
arretes_groupe2 = [
    (u, v) for u, v in graph.edges() if u in group[1] and v in group[1]
]
print("le nombre d'arrêtes intergroupe", len(arretes_groupe1)+len(arretes_groupe2))
