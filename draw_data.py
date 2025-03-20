import json
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

with open('data.json', 'r',encoding="utf-8") as f:
    data = json.load(f)

data = np.array(data)

y_values = data[:, 1]

sns.lineplot(x=range(len(y_values)), y=y_values)
plt.xlabel('time')
plt.ylabel('acc')
plt.title('time-acc')
plt.yticks(np.arange(min(y_values), max(y_values), (max(y_values) - min(y_values)) / 20))
plt.show()
