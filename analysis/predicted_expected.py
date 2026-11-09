import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = ''

data = pd.read_csv(file_path)

plt.figure(figsize=(16, 12), dpi=100)

# Plot the second and third columns
plt.plot(data.iloc[:, 1], label='Expected Return')
plt.plot(data.iloc[:, 2], label='Predicted Return')
plt.ylabel("Return %")
plt.xlabel("Time on Daily Basis")
name = "DJI Return Prediction (25)"
plt.title(name)
plt.legend()
plt.savefig(name + ".png",bbox_inches='tight')  
