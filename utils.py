import numpy as np

# Projection matrix
P = np.random.randn(2, 5000)
P /= np.linalg.norm(P, axis=1, keepdims=True)

def decode(z): return P.T @ z



