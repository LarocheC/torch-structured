import torch_structured


# S,G,B: diagonal
# P: permutation
# x: batch_size x n_features
def fastfood_multiply(S, G, B, P, x):
    HBx = torch_structured._ops.hadamard_transform(B * x)
    PHBx = HBx[:, P]
    HGPHBx = torch_structured._ops.hadamard_transform(G * PHBx)
    return S * HGPHBx
