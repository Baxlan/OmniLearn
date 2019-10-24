import matplotlib.pyplot as plt
import numpy as np


content = open("output.out").readlines()

content = [content[i][:-1] for i in range(len(content))]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


# get loss type
loss_t = content[content.index("loss:")+1]


# get train loss
trainLoss = content[content.index("loss:")+2][:-1].split(',')
for i in range(0, len(trainLoss)):
    trainLoss[i] = float(trainLoss[i])
lns1 = ax1.plot(range(0, len(trainLoss)), trainLoss, label = "training loss", color="blue")


# get validation loss
validLoss = content[content.index("loss:")+3][:-1].split(',')
for i in range(0, len(validLoss)):
    validLoss[i] = float(validLoss[i])
lns2 = ax1.plot(range(0, len(validLoss)), validLoss, label = "validation loss", color="orange")


# get metric type
metric_t = content[content.index("metric:")+1]
metricLabel1 = "normalized test metric"
metricLabel2 = "unormalized test metric"
if metric_t == "accuracy":
  metricLabel1 = "accuracy"
  metricLabel2 = "false prediction rate"


# get normalized metric
testAccuracyPerFeature = content[content.index("metric:")+2][:-1].split(',')
for i in range(0, len(testAccuracyPerFeature)):
    testAccuracyPerFeature[i] = float(testAccuracyPerFeature[i])
lns3 = ax2.plot(range(0, len(testAccuracyPerFeature)), testAccuracyPerFeature, label = metricLabel1, color = "green")


# get unormalized metric
testAccuracyPerOutput = content[content.index("metric:")+3][:-1].split(',')
for i in range(0, len(testAccuracyPerOutput)):
    testAccuracyPerOutput[i] = float(testAccuracyPerOutput[i])
lns4 = ax2.plot(range(0, len(testAccuracyPerOutput)), testAccuracyPerOutput, label = metricLabel2, color = "red")


# get optimal epoch
optimal = int(content[content.index("optimal epoch:")+1])
plt.axvline(optimal, color = "black")


# plot evolution regarding epochs
plt.title("Learning regarding epochs", fontsize=18)
plt.grid()
ax1.set_xlabel("epoch", fontsize=16)
ax1.set_ylabel(loss_t + " loss", fontsize=16)
ax1.set_yscale("log")
ax2.set_ylabel(metric_t + " metric", fontsize=16)
if metric_t != "test accuracy":
  ax2.set_yscale("log")


lns = lns1 + lns2 + lns3 + lns4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=14)

plt.show()



# get metric per output
metric = content[content.index("metric:")+4][:-1].split(',')
for i in range(0, len(metric)):
    metric[i] = float(metric[i])


# get second epoch per output
metric2 = content[content.index("metric:")+5][:-1].split(',')
for i in range(0, len(metric2)):
    metric2[i] = float(metric2[i])





metricLabel1 = "test metric"
metricLabel2 = "test metric deviation"
if metric_t == "accuracy":
  metricLabel1 = "accuracy"
  metricLabel2 = "false prediction rate"


# plot detailed metrics per output at optimal epoch for normalized outputs
width = 0.3
ind = np.arange(len(metric))

ax = plt.figure().add_subplot(111)

rec1 = ax.bar(ind, metric, width, color = "green")
rec2 = ax.bar(ind + width, metric2, width, color = "red")

ax.legend((rec1, rec2), (metricLabel1, metricLabel2), fontsize=14)
plt.xticks(ind + width/2, content[content.index("labels:")+1][:-1].split(","), rotation=45)
plt.axes().yaxis.grid()
plt.ylabel(metric_t + " metric", fontsize=16)
plt.xlabel("label", fontsize=16)
plt.title("Metrics for normalized outputs at optimal epoch (=" + str(optimal) + ")", fontsize=18)
if metric_t != "test accuracy":
  plt.yscale("log")
plt.show()



# get min output normalization
min = content[content.index("output normalization:")+1][:-1].split(",")
for i in range(0, len(min)):
    min[i] = float(min[i])


# get max output normalization
max = content[content.index("output normalization:")+2][:-1].split(",")
for i in range(0, len(max)):
    max[i] = float(max[i])


# compute normalized outputs
metric = [metric[i] * (max[i] - min[i]) for i in range(0, len(metric))]
metric2 = [metric2[i] * (max[i] - min[i]) for i in range(0, len(metric2))]


# plot detailed metrics per output at optimal epoch for unormalized outputs
width = 0.3
ind = np.arange(len(metric))

ax = plt.figure().add_subplot(111)

rec1 = ax.bar(ind, metric, width, color = "green")
rec2 = ax.bar(ind + width, metric2, width, color = "red")

ax.legend((rec1, rec2), (metricLabel1, metricLabel2), fontsize=14)
plt.xticks(ind + width/2, content[content.index("labels:")+1][:-1].split(","), rotation=45)
plt.axes().yaxis.grid()
plt.ylabel(metric_t + " metric", fontsize=16)
plt.xlabel("label", fontsize=16)
plt.title("Metrics for unormalized outputs at optimal epoch (=" + str(optimal) + ")", fontsize=18)
if metric_t != "test accuracy":
  plt.yscale("log")
plt.show()