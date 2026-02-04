import numpy as np
import matplotlib.pyplot as plt


#Generate Data
x = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800, 850])
y = np.array([200, 220, 250, 270, 300, 320, 350, 370, 400, 420])

def stochastic_gradient_descent(x , y , learning_rate=0.000001 , iterations = 20):
    m, b = 0, 0
    n = len(y)
    mse_history = []

    for i in range(iterations):
        index = np.random.randint(0, n)
        x_i = x[index]
        y_i = y[index]  
        y_pred = m * x_i + b 
        MSE = (y_i - y_pred)**2

        dm = -2 * x_i * (y_i - y_pred)
        db = -2 * (y_i - y_pred) 
        m -= learning_rate * dm 
        b -= learning_rate * db

        mse_history.append(MSE)

        print(f'Epoch { i + 1 } | y = {m:.3f} x + {b:.3f} | error: {MSE:.4f}')
    return m, b, mse_history 

stochastic_gradient_descent(x , y , iterations=20)
