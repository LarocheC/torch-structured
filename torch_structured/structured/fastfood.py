from .hadamard import hadamard_transform


# S,G,B: diagonal
# P: permutation
# x: batch_size x n_features
def fastfood_multiply(S, G, B, P, x):
    HBx = hadamard_transform(B * x)
    PHBx = HBx[:, P]
    HGPHBx = hadamard_transform(G * PHBx)
    return S * HGPHBx
