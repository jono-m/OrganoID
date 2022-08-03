import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import pandas
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'
data = pandas.read_csv("Publication\Supplement\S4\ManualInspection.csv")
percentCorrect = data["Correct Matches"] / (data["Correct Matches"] + data["Incorrect Matches"])
a = [x / 255 for x in (0, 205, 108)]
plt.plot(percentCorrect.index * 2, percentCorrect, color=a)
plt.ylim([0, 1.1])
plt.xlabel("Time (hours)")
plt.ylabel("Tracking accuracy")
plt.axhline(y=min(percentCorrect), color="black", linestyle="--")
plt.show()
