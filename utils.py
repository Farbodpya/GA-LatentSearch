import matplotlib.pyplot as plt

def plot_results(nfe, best_cost, time):
    fig, ax = plt.subplots(2, 1, figsize=(5, 8))
    ax[0].plot(nfe, best_cost, linewidth=2)
    ax[0].set_xlabel('NFE')
    ax[0].set_ylabel('Best Cost')
    ax[0].set_title('Best Cost vs NFE')

    ax[1].plot(time, linewidth=2)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Time (seconds)')
    ax[1].set_title('Time per Iteration')

    plt.tight_layout()
    plt.show()
