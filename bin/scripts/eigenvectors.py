import matplotlib.pyplot as plt
import numpy as np
import math
import sys


if len(sys.argv) < 2:
  print("ERROR: the file to be analyzed has to be specified.")
  sys.exit()


content = open(sys.argv[1]).readlines()
content = [content[i][:-1] for i in range(len(content))]



# ===========================================
# ===========================================
# ========== EVOLUTION ======================
# ===========================================
# ===========================================


# get loss type
loss_t = content[content.index("loss:")+1]

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



outLabels = content[content.index("output labels:")+1][:-1].split(",")

# get test outputs
predicted = ""
expected = ""
for lab in outLabels:
  expected = expected + content[content.index("label: " + lab)+1][:-1] + ";"
  predicted = predicted + content[content.index("label: " + lab)+2][:-1] + ";"
expected = np.matrix(expected[:-1])
predicted = np.matrix(predicted[:-1])

# turn real data into processed ones
process = content[content.index("output preprocess:")+1][:-1].split(",")

for proc in process:
  if proc == "center":
    center = [float(val) for val in content[content.index("output center:")+1][:-1].split(",")]
    for i in range(len(outLabels)):
      expected[i] = expected[i] - center[i]
      predicted[i] = predicted[i] - center[i]

  if proc == "decorrelate":
    vectors = ""
    for i in range(len(outLabels)):
      vectors = vectors + content[content.index("output eigenvectors:")+1+i][:-1] + ";"
    Ut = np.matrix(vectors[:-1])
    for i in range(np.size(expected, 1)):
      expected[:,i] = Ut * expected[:,i]
      predicted[:,i] = Ut * predicted[:,i]
    for i in range(len(outLabels)):
      outLabels[i] = "eigenvector " + str(i+1)

  if proc == "reduce":
    eigen = [float(val) for val in content[content.index("output eigenvalues:")+1][:-1].split(",")]
    threshold = float(content[content.index("output eigenvalues:")+2])
    tot = np.sum(eigen)
    rates = [np.sum(eigen[0:i])/tot for i in range(len(eigen))]
    optimal = 0
    for i in range(len(eigen)):
      optimal += eigen[i]/tot
      if optimal >= threshold:
        optimal = i
        break
    for i in range(len(outLabels)):
      if(i > optimal):
        outLabels[i] = outLabels[i] + " (not learned)"

  if proc == "normalize":
    min = [float(val) for val in content[content.index("output normalization:")+1][:-1].split(",")]
    max = [float(val) for val in content[content.index("output normalization:")+2][:-1].split(",")]
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
      dev.append(np.std(error[i]))
    elif sys.argv[2] == "rmse":
      metric.append(sqrt(np.mean(np.square(error[i]))))
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
      dev.append(np.std(error[i]))
    elif sys.argv[2] == "rmse":
      metric.append(sqrt(np.mean(np.square(error[i]))))
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