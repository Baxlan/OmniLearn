import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
from decimal import Decimal
import sys


wantedLabels = []
if(len(sys.argv) > 2):
  for val in sys.argv[2:]:
    wantedLabels.append(val)

content = open(sys.argv[1]).readlines()
content = [content[i][:-1] for i in range(len(content))]

outLabels = content[content.index("output labels:")+1][:-1].split(",")

# get outputs
predicted = []
expected = []
for lab in outLabels:
  floats = [float(val) for val in content[content.index("label: " + lab)+1][:-1].split(",")]
  expected.append(np.array(floats))
  floats = [float(val) for val in content[content.index("label: " + lab)+2][:-1].split(",")]
  predicted.append(np.array(floats))

"""
# get min output normalization
min = content[content.index("output normalization:")+1][:-1].split(",")
for i in range(0, len(min)):
    min[i] = float(min[i])

# get max output normalization
max = content[content.index("output normalization:")+2][:-1].split(",")
for i in range(0, len(max)):
    max[i] = float(max[i])

# denormalize outputs
for lab in range(len(outLabels)):
  expected[lab] = [(expected[lab][i] * (max[lab] - min[lab])) + min[lab] for i in range(0, len(expected[lab]))]
  #predicted[lab] = [(predicted[lab][i] - min[lab]) / (max[lab] - min[lab])  for i in range(0, len(predicted[lab]))]
"""

# get metric type
metric_t = content[content.index("metric:")+1]

# REGRESSION PROBLEM
if metric_t == "regression":
  mae = []
  mse = []

  for lab in range(len(outLabels)):
    if len(wantedLabels) != 0 and outLabels[lab] not in wantedLabels:
      continue
    #corr = np.corrcoef(expected[lab], predicted[lab])
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

    ax1.set_title(outLabels[lab] + " prediction value analysis (on unormalized outputs)\ncorrelation=" + str(round(corr, 4)) + "\norigin=" + "%.4E"%Decimal(origin) + "   slope=" + str(round(slope, 4)), fontsize=18)
    ax1.set_ylabel("predicted value", fontsize=16)
    ax1.set_xlabel("expected value", fontsize=16)
    #ax1.set_xlim(minExpected, minExpected)
    #ax1.set_ylim(minPredicted, maxPredicted)
    ax1.legend(fontsize=14)
    ax1.grid(b=True, which='major', color='grey', linestyle='-')

    plt.show()

# CLASSIFICATION PROBLEM
if metric_t == "classification":
  print("ERROR: this script only works for regression problems.")