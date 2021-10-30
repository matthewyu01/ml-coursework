import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
# point #1
for K in range(1,5):
    for seed in range(5):
        mixture,post = common.init(X, K, seed)

        _,_, cost = kmeans.run(X,mixture, post)
        print("K: {} seed: {} cost: {}".format(K, seed, cost))