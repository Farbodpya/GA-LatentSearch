# projection.py
import numpy as np

D = 5000  # High-dimensional space
d = 2     # Latent (projected) dimension

# Random projection matrix
P = np.random.randn(d, D)
P /= np.linalg.norm(P, axis=1, keepdims=True)

def decode(z):
    return P.T @ z
