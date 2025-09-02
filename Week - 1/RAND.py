import random
import matplotlib.pyplot as plt

def GeneratePoints(n=100):
    return [random.random() for _ in range(n)]

def PlotPoints(values):
    plt.scatter(range(len(values)), values, s=10, c="green")
    plt.xlabel("Index")
    plt.ylabel("Random Value")
    plt.title("Scatter Plot of random.random() values")
    plt.show()

def main():
    values = GeneratePoints(100)
    PlotPoints(values)

if __name__ == "__main__":
    main()
