import numpy as np
import matplotlib.pyplot as plt

#Generate Synthetic Data
# x, y = generate_synthetic_data(num_points =100, noise= 50)
# Example usage 
x = np.array([500, 550, 620, 630, 665, 700, 770, 880, 920, 1000])
y = np.array([320, 380, 400, 390, 385, 410, 480, 600, 570, 620])

def gradient_descent(x, y, learning_rate=0.000001, iterations=10):
    m, b = 0.0, 0.0
    n = len(y)
    #store history for plotting
    mse_history = []
    m_history = []
    b_history = []

     #set up the subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(iterations):
        y_pred = m * x + b
        mse = (1/n) * np.sum((y - y_pred) ** 2)

        dm = (-2/n) * np.sum(x * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        m -= learning_rate * dm
        b -= learning_rate * db

        mse_history.append(mse)
        m_history.append(m)
        b_history.append(b)
        
        print(f'Epoch {i+1} | y = {m:.4f}x + {b:.4f} | MSE: {mse:.4f}') #:.4f = decimal places 

        # Final plots
        ax[0].scatter(x, y, color='blue')
        ax[0].plot(x,y_pred, label=f'Iteration {i+1}')
        ax[0].set_xlabel('size')
        ax[0].set_ylabel('price')
        ax[0].set_title('Gradient Descent Regression')
        ax[0].legend()
 
    ax[1].plot(range(len(mse_history)), mse_history, color='red')
    ax[1].set_xticks(range(len(mse_history)))
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Mean Squared Error')
    ax[1].set_title('MSE vs Iterations')

    plt.tight_layout()
    plt.show()


#perform Gradient Descent 
gradient_descent(x, y, iterations=10)

