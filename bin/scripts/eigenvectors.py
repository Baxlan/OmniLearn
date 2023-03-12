import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import json


if len(sys.argv) < 2:
  print("ERROR: the file to be analyzed has to be specified.")
  sys.exit()


content = json.load(open(sys.argv[1], "r"))



# ===========================================
# ===========================================
# ========== EVOLUTION ======================
# ===========================================
# ===========================================


# get loss type
loss_t = content["parameters"]["loss"]

# get metric type
metric_t = "regression"
metricLabel1 = "mae metric (normalized)"
metricLabel2 = "rmse metric (normalized)"
if loss_t in ["cross entropy", "binary cross entropy"]:
  metric_t = "classification"
  metricLabel1 = "likelihood"
  metricLabel2 = "false prediction rate"


# ===========================================
# ===========================================
# ===== UNORMALIZED DETAILED METRICS ========
# ===========================================
# ===========================================



outLabels = content["parameters"]["output labels"]

# get test outputs
predicted = []
expected = []
for lab in outLabels:
  expected.append(content["test data"]["outputs"][lab]["expected"])
  predicted.append(content["test data"]["outputs"][lab]["predicted"])
expected = np.matrix(expected)
predicted = np.matrix(predicted)

# turn real data into processed ones
process = content["preprocess"]["output"]["preprocess"]

for proc in process:

  if proc == "standardize":
    mean = content["preprocess"]["output"]["standardization"][0]
    dev = content["preprocess"]["output"]["standardization"][1]
    for i in range(len(outLabels)):
      expected[i] = (expected[i] - mean[i]) / dev[i]
      predicted[i] = (predicted[i] - mean[i]) / dev[i]

  if proc == "whiten":
    val = content["preprocess"]["output"]["eigenvalues"]
    vec = []
    FAMDmean = content["preprocess"]["output"]["dummyMeans"]
    FAMDscale = content["preprocess"]["output"]["dummyScales"]
    bias = content["preprocess"]["output"]["whitening bias"]
    ZCA = content["parameters"]["output whitening type"] == "ZCA"

    for i in range(len(val)):
      vec.append(content["preprocess"]["output"]["eigenvectors"][i])

    vec = np.matrix(vec)

    for i in range(len(FAMDmean)):
      if(FAMDscale[i] != 0):
        expected[i] = (expected[i] * FAMDscale[i]) - FAMDmean[i]
        predicted[i] = (predicted[i] * FAMDscale[i]) - FAMDmean[i]

    expected = expected.transpose() * vec * np.reciprocal((np.sqrt(np.diag(val)) + bias))
    predicted = predicted.transpose() * vec * np.reciprocal((np.sqrt(np.diag(val)) + bias))

    if ZCA:
      expected = (expected * vec.transpose()).transpose()
      predicted = (predicted * vec.transpose()).transpose()
    else:
      expected = expected.transpose()
      predicted = predicted.transpose()

    for i in range(len(outLabels)):
      outLabels[i] = "eigenvector " + str(i+1)

  if proc == "reduce":
    val = content["preprocess"]["output"]["eigenvalues"]
    threshold = content["preprocess"]["output"]["reduction threshold"]
    tot = np.sum(val)
    rates = [np.sum(val[0:i])/tot for i in range(len(val))]
    optimal = 0
    for i in range(len(val)):
      optimal += val[i]/tot
      if optimal >= threshold:
        expexted = expected[:i+1]
        predicted = predicted[:i+1]
        outLabels = outLabels[:i+1]
        break

  if proc == "normalize":
    min = content["preprocess"]["output"]["normalization"][0]
    max = content["preprocess"]["output"]["normalization"][1]
    for i in range(len(outLabels)):
      expected[i] = (expected[i] - min[i]) / (max[i] - min[i])
      predicted[i] = (predicted[i] - min[i]) / (max[i] - min[i])

# REGRESSION PROBLEM
if metric_t == "regression":

  if len(sys.argv) < 3:
    print("ERROR: the detailed metric type to plot must be entered (mae or rmse).")
    sys.exit()

  metric = []
  dev = []
  err = []
  error = [predicted[i] - expected[i] for i in range(len(predicted))]

  for i in range(len(predicted)):
    if sys.argv[2] == "mae":
      metric.append(np.mean(np.abs(error[i])))
      dev.append(np.std(np.abs(error[i])))
    elif sys.argv[2] == "rmse":
      metric.append(np.sqrt(np.mean(np.square(error[i]))))
      dev.append(np.std(np.square(error[i])))
    err.append(dev[i]/math.sqrt(len(np.asarray(predicted)[i])))

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
  xtickNames = ax1.set_xticklabels(outLabels, rotation=45)
  plt.subplots_adjust(bottom=0.2)

  plt.show()



# ===========================================
# ===========================================
# ========= NORMALIZED METRICS ==============
# ===========================================
# ===========================================



if metric_t == "regression":

# normalize outputs
  min = []
  max = []
  for lab in range(len(outLabels)):
    min.append(np.min(expected[lab]))
    max.append(np.max(expected[lab]))
  for lab in range(len(outLabels)):
    expected[lab] = np.array([(expected[lab][i] - min[lab]) / (max[lab] - min[lab]) for i in range(0, len(expected[lab]))])
    predicted[lab] = np.array([(predicted[lab][i] - min[lab]) / (max[lab] - min[lab]) for i in range(0, len(predicted[lab]))])

  metric = []
  dev = []
  err = []
  error = [predicted[i] - expected[i] for i in range(len(predicted))]

  for i in range(len(predicted)):
    if sys.argv[2] == "mae":
      metric.append(np.mean(np.abs(error[i])))
      dev.append(np.std(np.abs(error[i])))
    elif sys.argv[2] == "rmse":
      metric.append(np.sqrt(np.mean(np.square(error[i]))))
      dev.append(np.std(np.square(error[i])))
    err.append(dev[i]/math.sqrt(len(np.asarray(predicted)[i])))

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
  xtickNames = ax1.set_xticklabels(outLabels, rotation=45)
  plt.subplots_adjust(bottom=0.2)

  plt.show()



# CLASSIFICATION PROBLEM
if metric_t == "classification":
  print("ERROR: this script only works for regression models.")