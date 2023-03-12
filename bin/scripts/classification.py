
import omnilearn
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    raise Exception("The .save file must be given")

np.seterr(all="ignore")

postprocessor = omnilearn.Classifier(sys.argv[1])

P, N = postprocessor.likelihood()
prior = postprocessor.prior()
prior["threshold"] = np.nan # to be consistent with other dataframes (for plottng)
P_post, N_post, P_evi, N_evi = postprocessor.posterior(P, N, prior)

# what to plot ?
df = P_post
thresholdIndex = 0
print("Considered threshold: " + str(df["threshold"][thresholdIndex]), flush=True)

print(prior)

# plotting
df = df.drop(["threshold"], axis=1) # delete "threshold" column
df = df.loc[[thresholdIndex], (df > 0).any()] # delete columns inferior to X
ind = np.arange(len(df.iloc[0]))
fig = plt.figure()
ax1 = fig.add_subplot(111)

width = 0.2
rects1 = ax1.bar(ind+0.5*width, df.iloc[0], width, color='green')

ax1.set_title("Model positive likelihoods", fontsize=18)
ax1.set_ylabel("Likelihood", fontsize=16)
ax1.set_xlabel("label", fontsize=16)
#ax1.set_yticks(np.arange(0, 1.05, 0.05))
ax1.set_yscale("log")
ax1.grid(visible=True, which='both', color='grey', linestyle='-', axis="y")
plt.xticks(ind+width, rotation = 20)
xtickNames = ax1.set_xticklabels(df.columns)

plt.show()