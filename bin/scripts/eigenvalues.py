import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
  print("ERROR: the file to be analyzed has to be specified.")
  sys.exit()
if len(sys.argv) < 3:
  print("ERROR: the eigenvectors to be ploted must be specified (in or out).")
  sys.exit()

content = open(sys.argv[1]).readlines()
content = [content[i][:-1] for i in range(len(content))]

values = []
threshold = 0
title = ""
if sys.argv[2] == "in":
  values = [float(val) for val in content[content.index("input eigenvalues:")+1][:-1].split(",")]
  threshold = float(content[content.index("input eigenvalues:")+2])
  title = "Input eigenvalue analysis"
if sys.argv[2] == "out":
  values = [float(val) for val in content[content.index("output eigenvalues:")+1][:-1].split(",")]
  threshold = float(content[content.index("output eigenvalues:")+2])
  title = "Output eigenvalue analysis"

if values[0] == 0:
  print("ERROR: decorrelation have not been performed.")
  sys.exit()
if len(values) == 1:
  print("ERROR: there is only one output, plot cannot be performed.")
  sys.exit()

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

plt.title(title, fontsize=18)

lns1 = ax1.plot(range(len(values)), rates, color="orange", label="cumulative variance")
ax1.set_yscale("log")
ax1.set_ylabel("Cumulative variance (%) (log)", fontsize=16)
ax1.grid(which="both")

lns2 = ax2.plot(range(len(values)), values, color="blue", label="eigenvalues")
ax2.set_yscale("log")
ax2.set_ylabel("Eigenvalues (log)", fontsize=16)
ax2.set_ylim(values[len(values)-1], values[0])

ax1.set_xlim(0, len(values))

lns = lns1 + lns2
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, fontsize=15)
plt.xlim(0, len(values))
plt.axvline(optimal, color = "black")
plt.show()