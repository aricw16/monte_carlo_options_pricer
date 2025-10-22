import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_style('whitegrid')

data= {'x': [1,2,3,4,5], 'y': [6,7,8,9,10]}
df=pd.DataFrame(data)

sns.scatterplot(data=df, x='x', y='y')

plt.show()