import numpy as np
import matplotlib.pyplot as plt

#Generate Data
np.random.seed(42) #for reproducibility
x = np.random.rand(100) * 50 # random 100 data points between 0 and 50
true_adam = 2.5 #True Slope
true_stacy = 5.0 #True Intercept
noise = np.random.randn(100) * 10 # Gaussian noise
y = true_adam * x + true_stacy + noise # Linear relation with noise


def minibatch(x, y, learning_rate=0.00001 , iterations=1000 , batch = 20 ):
    m, b = 0,0 
    n = len(y)
    mse_history = []

    for i in range(iterations): 
        indices = np.random.choice(n, batch, replace=False) 
        # Select random indices for the minibatch
        #choosing 4 (batch size) random points from the data(n)

        xb = x[indices]
        yb = y[indices]

        y_pred = m * xb + b
        MSE = (1/batch) * np.mean((yb - y_pred)**2) # Mean Squared Error for the minibatch with /batch

        dm = (-2/batch) * np.sum(xb * (yb - y_pred))
        #dm = (-2/batch) * Σ xi(yi - y_pred)

        db = (-2/batch) * np.sum(yb - y_pred)
        #db = (-2/batch) * Σ (yi - y_pred)

        m -= learning_rate * dm 
        b -= learning_rate * db 

        mse_history.append(MSE)

        print(f'Epoch {i + 1}  | y = {m:.4f} x + {b:.4f} | MSE : {MSE:.4f} | b = {batch}')
    
    return m, b, mse_history
    
minibatch

def plot_mse(x, y, m, b, mse_history, method ):
    fig , ax = plt.subplots(1 ,2 , figsize= (12 , 5))

    # Plotting regression line
    ax[0].scatter(x, y, color='blue')
    ax[0].plot(x, m * x + b, color='orange', label=f'{method} Regression Line')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title(f'{method} Linear Regression')
    ax[0].legend()

    # Plotting MSE vs Iterations
    ax[1].plot(range(len(mse_history)),  mse_history, color='green')
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Mean Squared Error")
    ax[1].set_title(f'{method} MSE vs Iterations')

    plt.tight_layout()
    plt.show()

plot_mse(x, y, *minibatch(x, y, iterations=1000, batch=20), method="Minibatch")


#Epoch 1000  | y = 2.6350 x + 0.1198 | MSE : 6.6699 | b = 20 

