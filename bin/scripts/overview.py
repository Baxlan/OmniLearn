import matplotlib.pyplot as plt
import numpy as np
import math
import sys


content = open(sys.argv[1]).readlines()
content = [content[i][:-1] for i in range(len(content))]



# ===========================================
# ===========================================
# ========== EVOLUTION ======================
# ===========================================
# ===========================================



# get loss type
loss_t = content[content.index("loss:")+1]


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

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
metric_t = "regression"
metricLabel1 = "mae test metric"
metricLabel2 = "mse test metric"
if loss_t in ["cross entropy", "binary cross entropy"]:
  metric_t = "classification"
  metricLabel1 = "accuracy"
  metricLabel2 = "false prediction rate"

# get metric mae or accuracy
mae_acc = content[content.index("metric:")+1][:-1].split(',')
for i in range(0, len(mae_acc)):
    mae_acc[i] = float(mae_acc[i])
lns3 = ax2.plot(range(0, len(mae_acc)), mae_acc, label = metricLabel1, color = "green")

# get metric mse or false prediction rate
mse_fp = content[content.index("metric:")+2][:-1].split(',')
for i in range(0, len(mse_fp)):
    mse_fp[i] = float(mse_fp[i])
lns4 = ax2.plot(range(0, len(mse_fp)), mse_fp, label = metricLabel2, color = "red")

# get optimal epoch
optimal = int(content[content.index("optimal epoch:")+1])
plt.axvline(optimal, color = "black")

# plot evolution regarding epochs
title = "Learning process overview"
if metric_t == "regression":
  title = "Learning process overview\n(metrics are on normalized outputs)"

plt.title(title, fontsize=18)
ax1.grid()
ax1.set_xlabel("epoch", fontsize=16)
ax1.set_ylabel(loss_t + " loss", fontsize=16)
ax1.set_yscale("log")
ax2.set_ylabel("metrics", fontsize=16)
ax2.set_yscale("log")
if metric_t == "classification":
  ax2.set_yscale("linear")
  ax2.set_ylim(0, 101)

lns = lns1 + lns2 + lns3 + lns4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=14)

plt.show()



# ===========================================
# ===========================================
# ========== DETAILED METRICS NORM ==========
# ===========================================
# ===========================================



outLabels = content[content.index("output labels:")+1][:-1].split(",")

# get outputs
predicted = []
expected = []
for lab in outLabels:
  floats = [float(val) for val in content[content.index("label: " + lab)+1][:-1].split(",")]
  expected.append(np.array(floats))
  floats = [float(val) for val in content[content.index("label: " + lab)+2][:-1].split(",")]
  predicted.append(np.array(floats))


# REGRESSION PROBLEM
if metric_t == "regression":

# get min output normalization
  min = content[content.index("output normalization:")+1][:-1].split(",")
  for i in range(0, len(min)):
    min[i] = float(min[i])

# get max output normalization
  max = content[content.index("output normalization:")+2][:-1].split(",")
  for i in range(0, len(max)):
    max[i] = float(max[i])

# normalize outputs
  for lab in range(len(outLabels)):
    predicted[lab] = [(predicted[lab][i] - min[lab]) / (max[lab] - min[lab]) for i in range(0, len(predicted[lab]))]

  metric = []
  dev = []
  err = []
  error = [predicted[i] - expected[i] for i in range(len(predicted))]

  for i in range(len(predicted)):
    if sys.argv[2] == "mae":
      metric.append(np.mean(np.abs(error[i])))
      dev.append(np.std(error[i]))
    elif sys.argv[2] == "mse":
      metric.append(np.mean(np.square(error[i])))
      dev.append(np.std(np.square(error[i])))
    err.append(dev[i]/math.sqrt(len(predicted[i])))

  # plot
  ind = np.arange(len(metric))
  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  width = 0.2
  rects1 = ax1.bar(ind+0.5*width, metric, width, yerr = err, color='red')
  rects2 = ax1.bar(ind+1.5*width, dev, width, color='orange')

  ax1.set_title("Detailed metrics (on normalized outputs)", fontsize=18)
  ax1.set_ylabel("metric value", fontsize=16)
  ax1.set_yscale("log")
  ax1.set_xlabel("labels", fontsize=16)
  plt.legend((rects1[0], rects2[0]), (sys.argv[2], "fluctuation (65%)"), fontsize=14, bbox_to_anchor=(0.8,-0.025))
  ax1.grid(b=True, which='major', color='grey', linestyle='-', axis="y")
  ax1.set_xticks(ind+width)
  xtickNames = ax1.set_xticklabels(outLabels)

  plt.show()


# ===========================================
# ===========================================
# ========== DETAILED METRICS UNORM =========
# ===========================================
# ===========================================


# REGRESSION PROBLEM
if metric_t == "regression":

# denormalize outputs
  for lab in range(len(outLabels)):
    expected[lab] = np.array([(expected[lab][i] * (max[lab] - min[lab])) + min[lab] for i in range(0, len(expected[lab]))])
    predicted[lab] = np.array([(predicted[lab][i] * (max[lab] - min[lab])) + min[lab] for i in range(0, len(predicted[lab]))])

  metric = []
  dev = []
  err = []
  error = [predicted[i] - expected[i] for i in range(len(predicted))]

  for i in range(len(predicted)):
    if sys.argv[2] == "mae":
      metric.append(np.mean(np.abs(error[i])))
      dev.append(np.std(error[i]))
    elif sys.argv[2] == "mse":
      metric.append(np.mean(np.square(error[i])))
      dev.append(np.std(np.square(error[i])))
    err.append(dev[i]/math.sqrt(len(predicted[i])))

  # plot
  ind = np.arange(len(metric))
  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  width = 0.2
  rects1 = ax1.bar(ind+0.5*width, metric, width, yerr = err, color='red')
  rects2 = ax1.bar(ind+1.5*width, dev, width, color='orange')

  ax1.set_title("Detailed metrics (on unormalized outputs)", fontsize=18)
  ax1.set_ylabel("metric value", fontsize=16)
  ax1.set_yscale("log")
  ax1.set_xlabel("labels", fontsize=16)
  plt.legend((rects1[0], rects2[0]), (sys.argv[2], "fluctuation (65%)"), fontsize=14, bbox_to_anchor=(0.8,-0.025))
  ax1.grid(b=True, which='major', color='grey', linestyle='-', axis="y")
  ax1.set_xticks(ind+width)
  xtickNames = ax1.set_xticklabels(outLabels)

  plt.show()



# ===========================================
# ===========================================
# ========== DETAILED ACCURACY ==============
# ===========================================
# ===========================================



# CLASSIFICATION PROBLEM
if metric_t == "classification":
  # get threshold
  threshold = float(content[content.index("classification threshold:")+1])

  acc = []
  fp = []

  for lab in range(len(outLabels)):
    count = 0
    validated = 0
    fp2 = 0
    for i in range(len(predicted[0])):
      if expected[lab][i] == 1:
        count = count + 1
        if(predicted[lab][i] >= threshold):
          validated = validated + 1
      else:
        if predicted[lab][i] >= threshold:
          fp2 = fp2 + 1
    acc.append(100*validated/count)
    fp.append(100*fp2/(validated+fp2))

  # plot
  ind = np.arange(len(acc))
  fig = plt.figure()
  ax1 = fig.add_subplot(111)

  width = 0.2
  rects1 = ax1.bar(ind+0.5*width, acc, width, color='green')
  rects2 = ax1.bar(ind+1.5*width, fp, width, color='red')

  ax1.set_title("Detailed accuracy on all labels", fontsize=18)
  ax1.set_ylabel("rate (%)", fontsize=16)
  ax1.set_xlabel("labels", fontsize=16)
  ax1.set_yticks(np.arange(0, 101, 5))
  ax1.set_ylim(0, 100)
  plt.legend((rects1[0], rects2[0]), ("accuracy", "false prediction rate"), fontsize=14, bbox_to_anchor=(1,-0.025))
  ax1.grid(b=True, which='major', color='grey', linestyle='-', axis="y")
  ax1.set_xticks(ind+width)
  xtickNames = ax1.set_xticklabels(outLabels)

  plt.show()