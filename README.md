# Generative AI for Validating Physics Laws

This repository contains the replication code.

## Replication Steps

1. Clone the repository:
   - `git clone https://github.com/yourusername/generativeAI.git`
   - `cd generativeAI`
  
 2. Install packages in the requirements.txt
 3. Run jupyter notebook application_stars.ipynb


# Minimalistic Example 


```
model = TreatmentEffectNet(x_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
batch_size = 32
n_samples = len(X_train)


for epoch in range(n_epochs):
    # Convert to numpy arrays for shuffling
    X_train_array = np.array(X_train)
    D_train_array = np.array(D_train)
    Y_train_array = np.array(Y_train)
    
    # Shuffle training data
    idx = np.random.permutation(n_samples)
    X_shuffle = X_train_array[idx]
    D_shuffle = D_train_array[idx]
    Y_shuffle = Y_train_array[idx]
    
    # Mini-batch training
    for i in range(0, n_samples, batch_size):
        batch_X = torch.FloatTensor(X_shuffle[i:i+batch_size])
        batch_D = torch.FloatTensor(D_shuffle[i:i+batch_size])
        batch_Y = torch.FloatTensor(Y_shuffle[i:i+batch_size])
        
        optimizer.zero_grad()
        loss = model.loss_fn(batch_X, batch_Y, batch_D, [1.0, 0.1, 0.1])
        loss.backward()
        optimizer.step()

# Get GenerativeAI predictions on test set
tau_gen_test = []
for i in range(len(X_test)):
    with torch.no_grad():
        x = torch.FloatTensor(X_test[i:i+1])
        z = torch.ones(1)
        tau = 0.5
        _, _, te, _ = model(x, z, tau)
        tau_gen_test.append(te.numpy()[0, 1])
tau_gen_test = np.array(tau_gen_test)

print("\nGenerativeAI  :")
print(f"Average Treatment Effect: {np.mean(tau_gen_test):.2f}")
print(f"Standard Deviation: {np.std(tau_gen_test):.2f}")
```



