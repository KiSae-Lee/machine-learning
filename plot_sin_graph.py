import numpy as np
import matplotlib.pyplot as plt

# Generate data from 0 to 6 with 0.1 interval
x = np.arange(0, 6, 0.1)
# Generate sin value of x
y1 = np.sin(x)
y2 = np.cos(x)

# Draw graph
plt.plot(x, y1, label="sin")
# Draw cos graph with dashed line
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()
