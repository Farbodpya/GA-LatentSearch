import numpy as np

# Original and latent dimensions
D = 5000  # Original dimension
d = 2     # Latent dimension

# Random projection matrix (2D → 1000D and vice versa)
P = np.random.randn(d, D)
P /= np.linalg.norm(P, axis=1, keepdims=True)

def project(x):
    """Project from 1000D to 2D"""
    return P @ x  # 1000D → 2D

def decode(z):
    """Decode from 2D to 1000D (approximate inverse)"""
    return P.T @ z  # 2D → 1000D
