import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['svg.fonttype'] = 'none'
data = open(Path(r"Publication\Supplement\S4\ManualInspection.csv"), mode="r").read()
performance = np.asarray([[int(x) for x in y.split(",")] for y in data.split("\n")[1:]])
percentCorrect = performance[:, 0] / (performance[:, 0] + performance[:, 1])
a = [x / 255 for x in (0, 205, 108)]
plt.plot(np.arange(performance.shape[0]) * 4, percentCorrect, color=a)
plt.ylim([0, 1.1])
plt.xlabel("Time (hours)")
plt.ylabel("Tracking accuracy")
plt.axhline(y=min(percentCorrect), color="black", linestyle="--")
print(min(performance[2, :]))
plt.show()
