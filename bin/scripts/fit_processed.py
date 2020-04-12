import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
from decimal import Decimal
import sys


wantedLabels = ""
if(len(sys.argv) > 2):
  wantedLabels = sys.argv[2]

content = open(sys.argv[1]).readlines()
content = [content[i][:-1] for i in range(len(content))]

outLabels = content[content.index("output labels:")+1][:-1].split(",")

# get test outputs
predicted = ""
expected = ""
for lab in outLabels:
  expected = expected + content[content.index("label: " + lab)+1][:-1] + ";"
  predicted = predicted + content[content.index("label: " + lab)+2][:-1] + ";"
expected = np.matrix(expected[:-1])
predicted = np.matrix(predicted[:-1])

# get metric type
metric_t = content[content.index("loss:")+1]
if metric_t in ["mse", "mae"]:
  metric_t = "regression"
else:
  metric_t = "classification"

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
  mae = []
  mse = []

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