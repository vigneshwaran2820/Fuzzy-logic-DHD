import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x_average  = np.arange(0, 101, 1)
average = fuzz.trimf(x_average, [0, 0, 53])

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
ax0.plot(x_average, average, 'b', linewidth=1.5, label='average age')
ax0.set_title('average age')
