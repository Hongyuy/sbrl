import numpy as np
import pysbrl

train_rules = [("{rule1}", np.array([0, 1, 0, 1, 1])), ("{rule2}", np.array([1, 0, 0, 1, 0]))]
train_labels = [("{label=0}", np.array([1, 0, 0, 1, 0])), ("{label=1}", np.array([0, 1, 1, 0, 1]))]

pysbrl.run(train_rules, train_labels, tnum=1, debug_level=1000, ruleset_size=3, iterations=1)
