import matplotlib.pyplot as plt

def LCG(a, c, m, seed, N):
    values = []
    I = seed
    for _ in range(N):
        I = (a * I + c) % m
        values.append(I)
    return values

def PlotLCG(values, m):
    scaled = [v / (2 ** m) for v in values]  
    plt.scatter(range(len(values)), scaled, s=10, c="blue")
    plt.xlabel("Step n")
    plt.ylabel("I_n / 2^m")
    plt.title("LCG Scatter Plot")
    plt.show()

def ShowRepeats(values):
    seen = {}
    for i, v in enumerate(values):
        if v in seen:
            print(f"Repeat detected: value {v} first at step {seen[v]} and again at step {i}")
            return
        seen[v] = i
    print("No repeat detected within given steps")

def main(a, c, m, seed, N):
    sequence = LCG(a, c, m, seed, N)
    ShowRepeats(sequence)
    PlotLCG(sequence, m)

if __name__ == "__main__":
    main(a=3, c=5, m=2**5, seed=7, N=50)
