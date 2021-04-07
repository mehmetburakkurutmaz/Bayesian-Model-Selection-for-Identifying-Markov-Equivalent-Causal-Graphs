import pandas as pd
import os
import random

random.seed(42)
folder = "pairs/"

for f in os.listdir("pairs"):
	if f[-3:] == "csv":
		df = pd.read_csv(folder+f, header=None)
		if len(df) > 1000:
			df = df.sample(1000)
			df.to_csv(folder + f, index=False, header=False, sep=",")
