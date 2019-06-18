
import matplotlib.pyplot as plt


lossFile = open("output.txt").readlines()


loss = lossFile[0][:-2].split(',')

for i in range(0, len(loss)):
    loss[i] = float(loss[i])

plt.plot(range(0, len(loss)), loss)

"""
loss1 = lossFile[1][:-1].split(',')

for i in range(0, len(loss1)):
    loss1[i] = float(loss1[i])

plt.plot(range(0, len(loss1)), loss1)
"""

plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(0, 120)
plt.show()
