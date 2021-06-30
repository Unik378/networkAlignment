import numpy as np 
import networkx as nx 
import pickle
import math,time,os,sys
from parameters import *

# Calculates the similarity between two node using structural and attribute-based identity
def similarity_nodes(du,dv,gamma_s,gamma_a,fu = None, fv = None):

	dist = gamma_s * np.linalg.norm(du-dv)
	if fv is not None:
		dist = dist + gamma_a * np.sum(fu != fv)

	return np.exp(-dist)

# Randomly choose p landmarks from n nodes
def chooseLandmarks(p,nodes):
	landmarks = np.random.permutation(np.arange(nodes))[:p]
	return landmarks

# Calculate neighbours of each node upto K-hop
# Outputs a dictionary of dictionaries
# { node : { k : {set of kth hop neighbours}}}
def khopNeighbors(G,K):

	# dictionary to store k-hop neighbors of all nodes
	k_hop_neigh = {}
	n = G.number_of_nodes()

	# dictionary to store neighbours which we already addend in k_hop_dictionary 
	neig_found = {}

	# 1st hop neighbors are directly connected nodes
	for u in range(n):

		# neighbours of current node
		neighbors = set(G.neighbors(u))

		k_hop_neigh[u] = {1:neighbors}

		neig_found[u] = neighbors.union(set([u]))

	# using previous hop neighbor to calculate next hop neighbors
	current_hop = 2
	while current_hop <= K:

		# for every node find current hop neighbours using previous hop nodes
		for u in range(n):
			currentHopneighbors = set()

			for prev in k_hop_neigh[u][current_hop-1]:
				neighbors = set(G.neighbors(prev))
				currentHopneighbors = currentHopneighbors.union(neighbors)

			# remove nodes which are already discovered in previous hops
			currentHopneighbors = currentHopneighbors - neig_found[u]

			k_hop_neigh[u][current_hop] = currentHopneighbors

			neig_found[u] = neig_found[u].union(currentHopneighbors)

		current_hop += 1

	return k_hop_neigh

# Calculate the embeddings for every node
def embeddings(G,parameters,attribute=None):

	maxDegree = max(np.array(G.degree)[:,1])
	print("maxDegree : ",maxDegree)

	# log binning
	d_len = int(math.log(maxDegree,parameters.base)) + 1
	nodes = G.number_of_nodes()
	p = int(parameters.size_landmarks*math.log(nodes,parameters.base))

	if p > nodes:
		p = nodes

	parameters.p = p

	# get k-hop neighbours
	before = time.time()
	khop = khopNeighbors(G,parameters.max_layer)
	after = time.time()

	print("Got k-hop neighbors in time:",after-before)

	d = np.zeros([nodes,d_len],dtype=float)

	# get degree sequences/distribution
	before = time.time()
	for u in range(nodes):
		for k in range(1,parameters.max_layer+1):

			duk = [0.0] * d_len
			for v in khop[u][k]:
				duk[int(math.log(G.degree(v),parameters.base))] += 1

			d[u] = d[u] + [math.pow(parameters.delta,k-1) * x for x in duk];
	after = time.time()

	print("Got degree sequence in time :",after-before)

	L = chooseLandmarks(p,nodes)
	C = np.zeros([nodes,p],dtype=float)

	# computing similarity of p landmarks nodes with all nodes
	for u in range(nodes):
		for v in range(p):

			if attribute is not None:
				C[u][v] = similarity_nodes(d[u],d[L[v]],parameters.gamma_s,parameters.gamma_a, attribute[u], attribute[v])
			else:
				C[u][v] = similarity_nodes(d[u],d[L[v]],parameters.gamma_s,parameters.gamma_a)


	# computing nystrom-based embeddings 
	before = time.time()
	W_pinv = np.linalg.pinv(C[L])
	U,sigma,V = np.linalg.svd(W_pinv)

	Y = np.dot( C, np.dot( U, np.diag(np.sqrt(sigma))))
	after = time.time()

	print("Got representation in time : ",after-before)

	Y = Y / np.linalg.norm(Y, axis = 1).reshape((Y.shape[0],1))

	return Y


if __name__ == "__main__":

	G = nx.read_edgelist("data/combined_edges.txt" , nodetype=int,comments="%")
	parameters = Parameters()
	Y = embeddings(G,parameters)

	print(Y)

