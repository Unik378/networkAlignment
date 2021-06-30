import numpy as np
import argparse
import networkx as nx
import sklearn.metrics.pairwise
import time,math,sys,os,pickle
from embeddings import *
from parameters import *

# parsing arguments
def argParser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input', nargs='?', default='data/combined_edges.txt', help="Edgelist of combined input graph")

	parser.add_argument('--output', nargs='?', default='embeddings/embeddings.emb', help="File to save Embeedings of combined input graph")

	parser.add_argument('--attributes', nargs='?', default=None,help='File with saved numpy matrix of node attributes')

	parser.add_argument('--g1_size', type=int, default=None,
							help='Nodes in Graph 1 in combined graph.If none we assume both graphs have same size.')

	parser.add_argument('--landmark_size', type=int, default=10,help='Controls of landmarks to sample. Default is 10.')

	parser.add_argument('--max_layer', type=int, default=2,help='Calculate degree sequence upto max_layer hop neighbours.')

	parser.add_argument('--delta', type=float, default = 0.01, help = "Discount factor for further layers")

	parser.add_argument('--gamma_s', type=float, default = 1, help = "Weight on structural similarity")

	parser.add_argument('--gamma_a', type=float, default = 1, help = "Weight on attribute similarity")

	parser.add_argument('--base', default=2, type=float, help="base of log for degree (node feature) binning")

	return parser.parse_args()

# split the combined embeddings of all graphs 
# g_size contains all indexes where we have to split
# this code is for only two graphs but can be modified to use for multiple graphs
def split_embeddings(Y,g_size = None):

	if g_size is not None:
		nodes = g_size
	else:
		nodes = int(Y.shape[0] / 2)

	Y1 = Y[:nodes]
	Y2 = Y[nodes:]

	return Y1,Y2

# calculate similarities between embeddings and return alignment matrix
def embedding_similarities(Y1, Y2, sim_measure="euclidean"):

	if sim_measure == "cosine":
		alignment_matrix = sklearn.metrics.pairwise.cosine_similarity(Y1, Y2)
	else:
		alignment_matrix = sklearn.metrics.pairwise.euclidean_distances(Y1, Y2)
		alignment_matrix = np.exp(-alignment_matrix)

	return alignment_matrix

def alignment_score( alignment_matrix, true_alignments, top_k):

	score = 0

	g_size = alignment_matrix.shape[0]

	for u in range(g_size):

		target_alignment = u

		if true_alignments is not None:
			target_alignment = int(true_alignments[u])

		sortedIndices = np.argsort(alignment_matrix[u])

		if target_alignment in sortedIndices[-top_k:]:
			score += 1

	score = score/g_size

	return score

def main(arg):

	graphFile = arg.input
	true_alignments_path = "data/true_alignments.txt"

	G = nx.read_edgelist(graphFile , nodetype=int, comments="%")
	print("Read in Graph")

	true_alignments = None
	if os.path.exists(true_alignments_path):
		with open(true_alignments_path,"rb") as true_alignments_file:
			true_alignments = pickle.load(true_alignments_file , encoding="latin1")

	parameters = Parameters(size_landmarks = arg.landmark_size,
							max_layer = arg.max_layer,
							delta = arg.delta,
							base = arg.base,
							gamma_s = arg.gamma_s,
							gamma_a = arg.gamma_a,
							g1_size = arg.g1_size)

	if arg.attributes is not None:
		attributes = np.load(arg.attributes)
	else:
		attributes = None

	Y = embeddings(G,parameters,attributes)

	print("\n\nFeature dimenson : ",Y.shape[1])

	print("\nEmbeddings : ")
	print(Y)

	np.save(arg.output , Y)

	Y1,Y2 = split_embeddings(Y,parameters.g1_size)

	alignment_matrix = embedding_similarities(Y1,Y2)

	print("\n\n")
	for topk in [1,2,3,4,5]:
		score = alignment_score(alignment_matrix,true_alignments,topk)

		print("Score top %d: %f" % (topk,score))

if __name__ == '__main__':
	arg = argParser()
	main(arg)
	