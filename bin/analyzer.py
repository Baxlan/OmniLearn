
import matplotlib.pyplot as plt
import numpy as np


content = open("output.txt").readlines()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


trainLoss = content[1][:-2].split(',')
for i in range(0, len(trainLoss)):
    trainLoss[i] = float(trainLoss[i])

lns1 = ax1.plot(range(0, len(trainLoss)), trainLoss, label = "training loss", color="blue")



validLoss = content[2][:-2].split(',')
for i in range(0, len(validLoss)):
    validLoss[i] = float(validLoss[i])

lns2 = ax1.plot(range(0, len(validLoss)), validLoss, label = "validation loss", color="orange")



testAccuracyPerFeature = content[3][:-2].split(',')
for i in range(0, len(testAccuracyPerFeature)):
    testAccuracyPerFeature[i] = float(testAccuracyPerFeature[i])

lns3 = ax2.plot(range(0, len(testAccuracyPerFeature)), testAccuracyPerFeature, label = "test accuracy", color = "green")



testAccuracyPerOutput = content[4][:-2].split(',')
for i in range(0, len(testAccuracyPerOutput)):
    testAccuracyPerOutput[i] = float(testAccuracyPerOutput[i])

lns4 = ax2.plot(range(0, len(testAccuracyPerOutput)), testAccuracyPerOutput, label = "test false positive", color = "red")



optimal = int(content[7])
plt.axvline(optimal, color = "black")



plt.title("Optimal epoch : " + str(optimal) + "\n(Feature acc: " + str(round(testAccuracyPerFeature[optimal-1])) + "%, Output acc: " + str(round(testAccuracyPerOutput[optimal-1])) + "%)", fontsize=18)
plt.grid()
ax1.set_xlabel("epoch", fontsize=16)
ax1.set_ylabel("loss", fontsize=16)
ax2.set_ylabel("Accuracy (%)", fontsize=16)
ax2.set_ylim(0, 105)
ax1.set_yscale("log")

lns = lns1 + lns2 + lns3 + lns4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=14)

plt.show()







acc = content[5][:-2].split(',')
for i in range(0, len(acc)):
    acc[i] = float(acc[i])



fp = content[6][:-2].split(',')
for i in range(0, len(fp)):
    fp[i] = float(fp[i])



width = 0.2
ind = np.arange(len(acc))

ax = plt.figure().add_subplot(111)

rec1 = ax.bar(ind, acc, width, color = "green")
rec2 = ax.bar(ind + width, fp, width, color = "red")

ax.legend((rec1, rec2), ('accuracy', 'false prediction'), fontsize=14)
plt.xticks(ind + width/2, content[0][:-2].split(","), rotation=45)
plt.ylim(0, 105)
plt.axes().yaxis.grid()
plt.ylabel("rate (%)", fontsize=16)
plt.xlabel("label", fontsize=16)
plt.title("Detailed accuracy at optimal epoch", fontsize=18)
plt.show()