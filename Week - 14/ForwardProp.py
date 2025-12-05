import numpy as np

np.random.seed(42)

N = 10  
L = 4   
nt = 20  
alpha = 0.2  

print(f"SK-GNN Forward Propagation")
print(f"========================")
print(f"Number of spins (N): {N}")
print(f"Number of layers (L): {L}")
print(f"Training set size: {nt}")
print(f"Mixing parameter (α): {alpha}\n")

print("Generating training set...")
training_set = []

for i in range(nt):
    J = np.random.randn(N, N)
    J = (J + J.T) / 2  
    np.fill_diagonal(J, 0)  
    

    target = np.random.choice([-1, 1], size=N)
    
    training_set.append({'J': J, 'target': target})

print(f"Generated {nt} training examples\n")

print("Initializing network parameters...")

xavier_factor = np.sqrt(3.0 / N)

weights = {}
biases = {}

for l in range(2, L + 1):  
    weights[l] = np.random.randn(N, N) * xavier_factor
    biases[l] = np.zeros(N) 

print(f"Initialized weights and biases for layers 2 to {L}\n")

def forward_propagation(J, a_init, weights, biases, alpha):
    activities = {1: a_init}
    z_values = {}
    
    for l in range(2, L + 1):
        interlayer = alpha * np.dot(weights[l], activities[l-1])
        a_current = activities[l-1].copy()
        
        for iteration in range(5):
            intralayer = (1 - alpha) * np.dot(J, a_current)
            z = interlayer + intralayer + biases[l]
            a_current = np.tanh(z)
        
        z_values[l] = z
        activities[l] = a_current
    
    return activities, z_values

def calculate_loss(prediction, target):
    return np.sum(np.square(prediction - target))

print("Running forward propagation on training set...")
print("=" * 70)
losses = []

for i, data in enumerate(training_set):
    J = data['J']
    target = data['target']
    a_init = np.random.uniform(-1, 1, size=N)
    activities, z_values = forward_propagation(J, a_init, weights, biases, alpha)
    prediction = activities[L]
    loss = calculate_loss(prediction, target)
    losses.append(loss)
    
    print(f"Training example {i+1:2d}:")
    print(f"  Target:     {target}")
    print(f"  Prediction: {np.round(prediction, 3)}")
    print(f"  Loss:       {loss:.6f}")
    print()

print("=" * 70)
print("Summary Statistics:")
print(f"  Mean loss:     {np.mean(losses):.6f}")
print(f"  Std loss:      {np.std(losses):.6f}")
print(f"  Min loss:      {np.min(losses):.6f}")
print(f"  Max loss:      {np.max(losses):.6f}")
print(f"  Total loss:    {np.sum(losses):.6f}")

results = {
    'losses': losses,
    'mean_loss': np.mean(losses),
    'training_set_size': nt,
    'alpha': alpha,
    'N': N,
    'L': L
}

print(f"\nForward propagation complete!")
print(f"Ready for backward propagation and training.")
