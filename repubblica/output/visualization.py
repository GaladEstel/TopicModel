import numpy as np
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

theta = np.load("theta.train.npz")["theta"]

arrTopics = pd.DataFrame(theta).fillna(0).values
arrTopics = arrTopics[np.amax(arrTopics, axis=1) > 0.25]
topicNum = np.argmax(arrTopics, axis=1)

tsneModel = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsneLda = tsneModel.fit_transform(arrTopics)

#output_notebook()
nTopics = 50
myColors = np.array([color for name, color in mcolors.CSS4_COLORS.items()])


#plot = figure(title="tsne clustering LDA", plot_width=900, plot_height=700)
plt.scatter(x=tsneLda[:, 0], y=tsneLda[:, 1], c=myColors[topicNum], linewidths=0.1)
plt.show()

print("End")
