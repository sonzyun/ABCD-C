# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

y = [5, 3, 7, 10, 9, 5, 3.5, 8]
x = range(len(y))
print(x)
plt.title("Bar Graph", fontsize = 20, fontweight = 'bold', loc = 'left', pad = 10)
colors = ['red' if val >7 else 'blue' for val in y]
plt.bar(x, y, width=0.7, color=colors)
plt.show()