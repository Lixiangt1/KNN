import numpy as np
import matplotlib.pyplot as plt

def kmeans(data,k=3, epoches=500):
    n=0
    centre = data[:k]

    for i in range(epoches):
        data_class = np.argmin(np.sum((data[:,None,:]-centre)**2, axis=2), axis=1)
        new_centre=[]
        for j in range(k):
            new_centre.append(data[data_class == j, :].mean(axis=0))
        new_centre=np.array(new_centre)
        if (new_centre == centre).all():
            break
        else:
            centre = new_centre

    return data_class,centre

data=np.random.rand(500,2)

classification,centres= kmeans(data,k=4)

plt.figure(figsize=(12, 8))
plt.scatter(x=data[:, 0], y=data[:, 1], s=100, c=classification)
plt.scatter(x=centres[:, 0], y=centres[:, 1], s=500, c='r', marker='*')
plt.show()