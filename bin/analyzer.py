
import matplotlib.pyplot as plt


content = open("output.txt").readlines()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


trainLoss = content[0][:-2].split(',')
for i in range(0, len(trainLoss)):
    trainLoss[i] = float(trainLoss[i])

lns1 = ax1.plot(range(0, len(trainLoss)), trainLoss, label = "training loss", color="blue")


validLoss = content[1][:-2].split(',')
for i in range(0, len(validLoss)):
    validLoss[i] = float(validLoss[i])

lns2 = ax1.plot(range(0, len(validLoss)), validLoss, label = "validation loss", color="orange")


testAccuracyPerFeature = content[2][:-2].split(',')
for i in range(0, len(testAccuracyPerFeature)):
    testAccuracyPerFeature[i] = float(testAccuracyPerFeature[i])


lns3 = ax2.plot(range(0, len(testAccuracyPerFeature)), testAccuracyPerFeature, label = "test accuracy (per feature)", color = "green")


testAccuracyPerOutput = content[3][:-2].split(',')
for i in range(0, len(testAccuracyPerOutput)):
    testAccuracyPerOutput[i] = float(testAccuracyPerOutput[i])


lns4 = ax2.plot(range(0, len(testAccuracyPerOutput)), testAccuracyPerOutput, label = "test accuracy (per output)", color = "red")


optimal = int(content[5])
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


acc = content[4][:-2].split(',')
for i in range(0, len(acc)):
    acc[i] = float(acc[i])

plt.bar(range(len(acc)), acc)
plt.xticks(range(len(acc)), content[6][:-2].split(","), rotation=45)
plt.ylim(0, 105)
plt.axes().yaxis.grid()
plt.show()