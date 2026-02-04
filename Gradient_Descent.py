import numpy as np 
import matplotlib.pyplot as plt

#Generate Synthetic Data

x = np.array([500, 550, 620, 630, 665, 700, 770, 880, 920, 1000])
y = np.array([320, 380, 400, 390, 385, 410, 480, 600, 570, 620])

def gradient_descent(x,y,learning_rate=0.000001, iterations=10) : 
    m = b = 0 
    n = len(y)
    mse_history = []
    m_history = []
    b_history = []

    for i in range(iterations) : 
        y_pred = m * x + b 
        mse = (1/n) * sum( [val**2 for val in (y - y_pred) ] )
        dm = -(2/n) * sum(x * (y - y_pred)) 
        db = -(2/n) * sum(y - y_pred) 
        m = m - learning_rate * dm
        b = b - learning_rate * db
        #print(f'Epoch {i+1} | y : {m:.4g}x + {b:.4g}  | MSE: {mse:.4f}')

        mse_history.append(mse)
        m_history.append(m)
        b_history.append(b)

        print(f'Epoch {i+1} | y = {m:.4f}x + {b:.4f} | MSE: {mse:.4f}')

gradient_descent(x, y, iterations=10)
