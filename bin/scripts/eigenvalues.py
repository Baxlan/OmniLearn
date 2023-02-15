import sys
import matplotlib.pyplot as plt
import numpy as np
import json

if len(sys.argv) < 2:
  print("ERROR: the file to be analyzed has to be specified.")
  sys.exit()
if len(sys.argv) < 3:
  print("ERROR: the eigenvectors to be ploted must be specified (\"input\" or \"output\").")
  sys.exit()

content = json.load(open(sys.argv[1], "r"))

preprocessList = content["preprocess"][sys.argv[2]]["preprocess"]
if "whiten" not in preprocessList:
  raise Exception(sys.argv[2] + " have not been whitened.")

values = []
threshold = 0
title = sys.argv[2] + " eigenvalue analysis"

values = [float(val) for val in content["preprocess"][sys.argv[2]]["eigenvalues"]]
threshold = float(content["preprocess"][sys.argv[2]]["whitening bias"])

tot = np.sum(values)
rates = [np.sum(values[0:i])/tot for i in range(len(values))]

optimal = 0
for i in range(len(values)):
  optimal += values[i]/tot
  if optimal >= threshold:
    optimal = i
    break

# create axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.suptitle(title.capitalize(), fontsize=18, weight="bold")
plt.title("(log scale)", fontsize=14)

lns1 = ax1.plot(range(len(values)), rates, color="orange", label="cumulative variance")
ax1.set_yscale("log")
ax1.set_ylabel("Cumulative variance (%)", fontsize=14)
ax1.grid(which="both")
ax1.set_xlabel("Eigenvalue rank", fontsize=14)

lns2 = ax2.plot(range(len(values)), values, color="blue", label="eigenvalues")
ax2.set_yscale("log")
ax2.set_ylabel("Eigenvalues", fontsize=14)
ax2.set_ylim(values[len(values)-1], values[0])

ax1.set_xlim(0, len(values))

lns = lns1 + lns2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=14, loc="lower left")
plt.xlim(0, len(values))
plt.axvline(optimal, color = "black")
plt.show()