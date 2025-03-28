import pysbm
import networkx as nx
import matplotlib.pylab as pl
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# Chemin vers votre fichier .net
file_path = "../Data/dolphins.net"

# Charger le graphe en utilisant la fonction de lecture Pajek de NetworkX
graph = nx.read_pajek(file_path)

# Convertir en un graph NetworkX standard (optionnel)
graph = nx.Graph(graph)  # ou nx.DiGraph(G) pour un graphe orienté

# Afficher des informations de base sur le graphe
# Exemple : Remplacez labels par la liste ou tableau contenant vos labels.
labels = graph.nodes  # Insérez ici vos données de labels

# Initialiser clusters à 1, de la même longueur que labels
clusters = np.ones(len(labels), dtype=int)

# Définir group2
group2 = [61, 33, 57, 23, 6, 10, 7, 32, 14, 18, 26, 49, 58, 42, 55, 28, 27, 2, 20, 8]

# Remplir clusters en utilisant les indices de group2 (attention aux indices en Python)
for i in group2:
    clusters[i - 1] = 0 # i-1 pour s'adapter aux indices Python (qui commencent à 0)

maxiter=40
standard_partition = pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2)
best_partition= pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2,
        representation=standard_partition.get_representation())
obj_max=-float('inf')


# partition de départ
for itt in range(maxiter):
    standard_partition = pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2)
    degree_corrected_partition = pysbm.NxPartition(
        graph=graph,
        number_of_blocks=2,
        representation=standard_partition.get_representation())


    degree_corrected_objective_function = pysbm.DegreeCorrectedUnnormalizedLogLikelyhood(is_directed=False)
    # As inference method we use MCMC with few steps.

    degree_corrected_inference = pysbm.MetropolisInference(graph, degree_corrected_objective_function, degree_corrected_partition)

    degree_corrected_inference.infer_stochastic_block_model(2000)
    obj_value=degree_corrected_objective_function.calculate(degree_corrected_partition)
    if obj_value > obj_max:
        best_partition=degree_corrected_partition
        obj_max=obj_value

position = nx.spring_layout(graph)

print("Degree Corrected SBM")
nx.draw(graph, position, node_color=['r' if best_partition.get_block_of_node(node) == 0 else 'b' for node in graph],with_labels=False,node_size=100)
label_pos = {node: (x, y - 0.08) for node, (x, y) in position.items()}  # Déplacement des labels vers le bas

nx.draw_networkx_labels(
    graph,
    label_pos,
    labels={node: str(node) for node in graph.nodes},
    font_size=8,               # Taille de police plus petite
    verticalalignment='bottom'  # Placer les étiquettes en dessous
)
pl.show()

essai=[best_partition.get_block_of_node(node) for node in graph]
nmi_score = normalized_mutual_info_score(clusters,essai)


print("Normalized Mutual Information (NMI):", nmi_score)




# Calculer la modularité
group=[]
group.append([node  for node in graph if best_partition.get_block_of_node(node) == 0])
group.append([node  for node in graph if best_partition.get_block_of_node(node) == 1])
modularity_value = nx.algorithms.community.modularity(graph, group)

print(f"La modularité de la partition est : {modularity_value:.4f}")