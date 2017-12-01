#Utility file for visualization of results

import matplotlib.pyplot as plt

def plot_graph(train_loss, eval_loss=None, test_loss=None):
    num_iterations = len(train_loss)
    plt.plot(range(num_iterations), train_loss, 'b', label="Training Loss")
    plt.plot(range(num_iterations), eval_loss, 'g--', label="Validation Loss")
    if test_loss is not None:
        plt.plot(range(num_iterations), test_loss, 'r', label="Testing Loss")

    plt.xlabel('Epochs-->')
    plt.ylabel('Loss-->')
    plt.title('Top-1 loss plotting') 
    plt.legend()

    plt.show()
