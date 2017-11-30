import matplotlib.pyplot as plt

with file('out.txt') as f:
    acc = [float(line) for line in f.read().split('\n')[:-1]]

plt.plot(acc)
plt.show()
