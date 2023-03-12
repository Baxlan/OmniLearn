import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
from decimal import Decimal
import sys
import json


wantedLabels = ""
if(len(sys.argv) > 2):
  wantedLabels = sys.argv[2]

content = json.load(open(sys.argv[1], "r"))

outLabels = content["parameters"]["output labels"]

# get test outputs
predicted = []
expected = []
for lab in outLabels:
  expected.append(content["test data"]["outputs"][lab]["expected"])
  predicted.append(content["test data"]["outputs"][lab]["predicted"])
expected = np.matrix(expected)
predicted = np.matrix(predicted)

# get metric type
metric_t = content["parameters"]["loss"]
if metric_t in ["L1", "L2"]:
  metric_t = "regression"
else:
  metric_t = "classification"

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
        expexted = expected[:i]
        predicted = predicted[:i]
        outLabels = outLabels[:i]
        break

  if proc == "normalize":
    min = content["preprocess"]["output"]["normalization"][0]
    max = content["preprocess"]["output"]["normalization"][1]
    for i in range(len(outLabels)):
      expected[i] = (expected[i] - min[i]) / (max[i] - min[i])
      predicted[i] = (predicted[i] - min[i]) / (max[i] - min[i])

# REGRESSION PROBLEM
if metric_t == "regression":
  mae = []
  rmse = []

  for lab in range(len(outLabels)):
    if wantedLabels != "" and lab != int(wantedLabels)-1:
      continue

    #np.seterr(invalid='ignore')
    slope, origin, corr, p, err = scipy.stats.linregress(np.asarray(expected)[lab], np.asarray(predicted)[lab])

    # x=y line
    minExpected = np.min(expected[lab])
    maxExpected = np.max(expected[lab])
    minPredicted = np.min(predicted[lab])
    maxPredicted = np.max(predicted[lab])

    min = np.min([minExpected, minPredicted])
    max = np.max([maxExpected, maxPredicted])

    #plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot([min, max], [min, max], label = "y=x", color = "red")
    lns2 = ax1.plot([min, max], [slope*min+origin, slope*max+origin], label="linear regression", color="orange")
    ax1.scatter(list(expected[lab]), list(predicted[lab]), s=1)

    ax1.set_title(outLabels[lab] + "\ncorr=" + str(round(corr, 4)) + "   origin=" + "%.4E"%Decimal(origin) + "   slope=" + str(round(slope, 4)), fontsize=18)
    ax1.set_ylabel("predicted value", fontsize=16)
    ax1.set_xlabel("expected value", fontsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(b=True, which='major', color='grey', linestyle='-')

    plt.show()

# CLASSIFICATION PROBLEM
if metric_t == "classification":
  print("ERROR: this script only works for regression models.")