import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clt()
    plt.title("Uczenie...")
    plt.xlabel("Liczba gier")
    plt.ylabel("Punkty")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.pause(.1)