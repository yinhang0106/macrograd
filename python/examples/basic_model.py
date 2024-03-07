from macrograd.nn import MLP

# example of training
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] # desired target

# build model
model = MLP(3, [4, 4, 1])

# training loop
rounds = 100 # number of training rounds
ypred = [] # predicted target
for k in range(rounds):

    # forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yi - yi_pred) ** 2 for yi, yi_pred in zip(ys, ypred))

    # backward pass
    loss.zero_grad()
    loss.backward()

    # update weights
    for p in model.parameters():
        p.data += -0.05 * p.grad
    
    # progress
    print(k, loss.data)

# print final prediction
print(ypred)
# Expected output:
print(ys)
