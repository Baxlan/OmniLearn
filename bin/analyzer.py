import matplotlib.pyplot as plt
import numpy as np

content = open("output.txt").readlines()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# get metric type
metric_t = content[3][:-1]
metric_t = "test " + metric_t
metric2_t = "test standard deviation"
if metric_t == "test accuracy":
  metric2_t == "false prediction"

# get train loss
trainLoss = content[1][:-2].split(',')
for i in range(0, len(trainLoss)):
    trainLoss[i] = float(trainLoss[i])
lns1 = ax1.plot(range(0, len(trainLoss)), trainLoss, label = "training loss", color="blue")

# get validation loss
validLoss = content[2][:-2].split(',')
for i in range(0, len(validLoss)):
    validLoss[i] = float(validLoss[i])
lns2 = ax1.plot(range(0, len(validLoss)), validLoss, label = "validation loss", color="orange")


# get train metric
testAccuracyPerFeature = content[4][:-2].split(',')
for i in range(0, len(testAccuracyPerFeature)):
    testAccuracyPerFeature[i] = float(testAccuracyPerFeature[i])
lns3 = ax2.plot(range(0, len(testAccuracyPerFeature)), testAccuracyPerFeature, label = metric_t, color = "green")


# get train second metric
testAccuracyPerOutput = content[5][:-2].split(',')
for i in range(0, len(testAccuracyPerOutput)):
    testAccuracyPerOutput[i] = float(testAccuracyPerOutput[i])
lns4 = ax2.plot(range(0, len(testAccuracyPerOutput)), testAccuracyPerOutput, label = metric2_t, color = "red")


# get optimal epoch
optimal = int(content[8])
plt.axvline(optimal, color = "black")


# plot evolution regarding epochs
plt.title("Loss regarding epochs", fontsize=18)
plt.grid()
ax1.set_xlabel("epoch", fontsize=16)
ax1.set_ylabel("loss", fontsize=16)
ax1.set_yscale("log")
ax2.set_ylabel(metric_t, fontsize=16)
if metric_t != "test accuracy":
  ax2.set_yscale("log")


lns = lns1 + lns2 + lns3 + lns4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=14)

plt.show()



# get metric per output
metric = content[6][:-2].split(',')
for i in range(0, len(metric)):
    metric[i] = float(metric[i])


# get second epoch per output
metric2 = content[7][:-2].split(',')
for i in range(0, len(metric2)):
    metric2[i] = float(metric2[i])


# plot detailed metrics per output at optimal epoch
width = 0.3
ind = np.arange(len(metric))

ax = plt.figure().add_subplot(111)

rec1 = ax.bar(ind, metric, width, color = "green")
rec2 = ax.bar(ind + width, metric2, width, color = "red")

ax.legend((rec1, rec2), (metric_t, metric2_t), fontsize=14)
plt.xticks(ind + width/2, content[0][:-2].split(","), rotation=45)
plt.axes().yaxis.grid()
plt.ylabel(metric_t, fontsize=16)
plt.xlabel("label", fontsize=16)
plt.title("Detailed metrics at optimal epoch (=" + str(optimal) + ")", fontsize=18)
if metric_t != "test accuracy":
  plt.yscale("log")
plt.show()