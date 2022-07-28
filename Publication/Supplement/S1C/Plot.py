from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
import numpy as np

lines = open(Path(r"Publication\Supplement\S1C\Losses.csv"), "r").read().split("\n")[1:-1]
losses = np.asarray([[float(x) for x in line.split(",")] for line in lines])
epoch = losses[:, 0]
trainingLosses = losses[:, 1]
validationLosses = losses[:, 2]

a = [x / 255 for x in (0, 205, 108)]
b = [x / 255 for x in (0, 154, 222)]

plt.plot(trainingLosses, color=a)
plt.plot(validationLosses, color=b)
plt.scatter(np.arange(len(trainingLosses)), trainingLosses, color=a)
plt.scatter(np.arange(len(validationLosses)), validationLosses, color=b)
plt.legend(["Training set", "Validation set"])
plt.xlabel("Epoch")
plt.ylabel("Loss (binary cross-entropy)")
plt.title("OrganoID neural network training performance")
plt.axvline(x=np.argmin(validationLosses), color="black", linestyle="--")
plt.show()
