# Dynamic Genetic Algorithm in Latent Space (dynGA)

This project implements a dynamic genetic algorithm (dynGA) that operates in latent space, allowing for efficient optimization of high-dimensional problems using latent-space techniques. The algorithm uses a custom projection matrix for encoding and decoding between a high-dimensional and a latent space.

## Features

- **Dynamic Population Size**: Population size decreases dynamically with each iteration.
- **Latent-Space Optimization**: The algorithm operates in a latent 2D space to optimize a problem in 1000D.
- **Crossover & Mutation**: The algorithm uses BLX-alpha crossover and Gaussian mutation strategies.
- **Cost Function**: The cost function used is the Sphere function, evaluated in the 1000D space.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Farbodpya/GA-LatentSearch.git
   cd GA-LatentSearch
