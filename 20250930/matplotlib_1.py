# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#plt.plot([2,3,4,5], linewidth=1 , c='b')
#plt.show()

#Title
plt.title("Graph Title", fontsize = 20, color = 'green', fontweight = 'bold', loc = 'left', pad = 10)
#plt.plot([1,2,3], [4,5,6], c = 'green', linewidth=2, linestyle='-')
#plt.plot([1,2,3], [4,5,6], c = 'green', linewidth=2, linestyle='--')
#plt.plot([1,2,3], [4,5,6], c = 'green', linewidth=2, linestyle='-.')
plt.plot([1,2,3], [4,5,6], c = 'green', linewidth=2, linestyle=':')
plt.plot([1,2,3], [1,4,9], c = 'blue', linewidth=2, linestyle=':')

plt.grid(linestyle='--')
plt.xlabel('Sequence', fontsize=10, color='green', fontweight='bold')
plt.ylabel('Time(secs)', fontsize=10, color='green', fontweight='bold')
plt.legend(['Mouse', 'Cat'], loc = 'upper left', fontsize=10)

plt.xlim([0, 4])  # X축의 범위: [xmin, xmax]
plt.ylim([0, 10]) # Y축의 범위: [ymin, ymax]

plt.show()