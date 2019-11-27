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
predicted = []
expected = []
for lab in outLabels:
  floats = [float(val) for val in content[content.index("label: " + lab)+1][:-1].split(",")]
  expected.append(np.array(floats))
  floats = [float(val) for val in content[content.index("label: " + lab)+2][:-1].split(",")]
  predicted.append(np.array(floats))

# get metric type
metric_t = content[content.index("loss:")+1]
if metric_t in ["mse", "mae"]:
  metric_t = "regression"
else:
  metric_t = "classification"

# REGRESSION PROBLEM
if metric_t == "regression":
  mae = []
  mse = []

  for lab in range(len(outLabels)):
    if wantedLabels != "" and outLabels[lab] != wantedLabels:
      continue

    slope, origin, corr, p, err = scipy.stats.linregress(expected[lab], predicted[lab])

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
    ax1.scatter(expected[lab], predicted[lab], s=1)

    ax1.set_title(outLabels[lab] + " prediction value analysis\ncorr=" + str(round(corr, 4)) + "   origin=" + "%.4E"%Decimal(origin) + "   slope=" + str(round(slope, 4)), fontsize=18)
    ax1.set_ylabel("predicted value", fontsize=16)
    ax1.set_xlabel("expected value", fontsize=16)
    ax1.legend(fontsize=14)
    ax1.grid(b=True, which='major', color='grey', linestyle='-')

    plt.show()

# CLASSIFICATION PROBLEM
if metric_t == "classification":
  print("ERROR: this script only works for regression models.")