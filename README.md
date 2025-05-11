# GA-LatentSearch

**GA-LatentSearch** is a high-dimensional optimization framework that leverages a Genetic Algorithm (GA) to perform search in a low-dimensional latent space. This approach enables efficient optimization of challenging benchmark functions in thousands of dimensions by projecting them into a compact subspace.

Zenodo DOI:(https://doi.org/10.5281/zenodo.15342205)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌱 Why Latent Search?

High-dimensional optimization is notoriously difficult due to the **curse of dimensionality**. Instead of searching in the full space (e.g. 5000D), GA-LatentSearch projects the problem into a 2D latent space using a random linear projection, optimizing within that space and decoding solutions back to the original space.

---

## ⚙️ How It Works

1. **Random Projection**: A `D × d` projection matrix (e.g. 5000×2) maps latent vectors to the high-dimensional space.
2. **Decode Function**: Latent solutions are decoded using matrix multiplication.
3. **GA Optimization**: A Genetic Algorithm operates on latent space vectors using blend crossover and Gaussian mutation.
4. **Fitness Evaluation**: Benchmark functions are applied on the decoded vectors.
5. **Dynamic Population**: The population size shrinks over time for convergence.

---

## 🧪 Benchmarks

Implemented benchmark functions include:

- Sphere
- Rastrigin
- Ackley
- Griewank
- Zakharov
- Rastrigin II

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Farbodpya/GA-LatentSearch.git
cd GA-LatentSearch

# Run the optimizer
python main.py
