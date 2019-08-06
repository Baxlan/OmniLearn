
import matplotlib.pyplot as plt


content = open("output.txt").readlines()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


trainLoss = content[0][:-2].split(',')
for i in range(0, len(trainLoss)):
    trainLoss[i] = float(trainLoss[i])

lns1 = ax1.plot(range(0, len(trainLoss)), trainLoss, label = "training loss")


validLoss = content[1][:-2].split(',')
for i in range(0, len(validLoss)):
    validLoss[i] = float(validLoss[i])

lns2 = ax1.plot(range(0, len(validLoss)), validLoss, label = "validation loss")


testAccuracy = content[2][:-2].split(',')
for i in range(0, len(testAccuracy)):
    testAccuracy[i] = float(testAccuracy[i])


lns3 = ax2.plot(range(0, len(testAccuracy)), testAccuracy, label = "test accuracy", color = "g")


optimal = int(content[3])
plt.axvline(optimal, color = "black")


plt.title("Optimal epoch : " + str(optimal) + " (" + str(round(testAccuracy[optimal-1])) + "% accurate)", fontsize=18)
plt.grid()
ax1.set_xlabel("epoch", fontsize=16)
ax1.set_ylabel("loss", fontsize=16)
ax2.set_ylabel("accuracy (%)", fontsize=16)
ax2.set_ylim(0, 100)

lns = lns1 + lns2 + lns3
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=14)

plt.show()
