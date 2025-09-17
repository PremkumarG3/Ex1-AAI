#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -m pip install pybbn')


# In[3]:


# Step 1: Import necessary libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Import pybbn classes
from pybbn.graph.dag import Bbn, Edge, EdgeType
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Step 2: Set pandas options
pd.options.display.max_columns = 50

# Step 3: Read the dataset
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Step 4: Remove records where RainTomorrow is missing
df = df[pd.notnull(df['RainTomorrow'])]

# Step 5: Fill missing values in numeric columns with mean
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 6: Create categorical bands
df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(lambda x: '0.<=40' if x <= 40 else '1.40-50' if x <= 50 else '2.>50')
df['Humidity9amCat'] = df['Humidity9am'].apply(lambda x: '0.<=60' if x <= 60 else '1.>60')
df['Humidity3pmCat'] = df['Humidity3pm'].apply(lambda x: '0.<=60' if x <= 60 else '1.>60')

# Step 7: Define probability calculation function
def probs(data, child, parent1=None, parent2=None):
    if parent1 is None:
        prob = pd.crosstab(data[child], 'Empty', normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent2 is None:
        prob = pd.crosstab(data[parent1], data[child], normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        prob = pd.crosstab([data[parent1], data[parent2]], data[child], normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    return prob

# Step 8: Create BbnNode objects
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

# Step 9: Create BBN and add nodes and edges
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

# Step 10: Convert to join tree for inference
join_tree = InferenceController.apply(bbn)

# Step 11: Set node positions for visualization
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

# Step 12: Set options for the graph appearance
options = {
    "font_size": 14,
    "node_size": 3000,
    "node_color": "pink",
    "edgecolors": "blue",
    "edge_color": "green",
    "linewidths": 2,
    "width": 2,
}

# Step 13 & 14: Generate the graph
fig, ax = plt.subplots(figsize=(8,6))  # Create figure and axes
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, ax=ax, **options)
ax.margins(0.10)
plt.axis("off")
plt.show()

# Step 15: Optional - Print probability tables
print("P(Humidity9amCat):", probs(df, child='Humidity9amCat'))
print("P(Humidity3pmCat | Humidity9amCat):", probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
print("P(WindGustSpeedCat):", probs(df, child='WindGustSpeedCat'))
print("P(RainTomorrow | Humidity3pmCat, WindGustSpeedCat):", probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))


# In[ ]:




