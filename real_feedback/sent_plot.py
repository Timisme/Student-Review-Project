import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib 

# print(matplotlib.__file__)

plt.style.use('fivethirtyeight')
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

df = pd.read_csv('new_feedback.csv', encoding= 'ANSI')
df = df[df['情緒標記']!=100]

Series_counts = df['情緒標記'].value_counts()

labels = ['正面評價','負面評價','中性評價']
counts = [count for (label, count) in Series_counts.items()]
# explode= [0, 0, 1]


# plot chart 
plt.pie(counts, labels= labels, wedgeprops={'edgecolor':'black'}, 
	autopct='%1.1f%%')

plt.title('學生回饋 - 情緒比例')
plt.tight_layout()
plt.show()
# plt.savefig('SentPie.png')

