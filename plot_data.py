import matplotlib.pyplot as plt

with open('training_output.txt') as f:
    data = list(map(float, f.read().splitlines()))

# print(data)

plt.plot(data)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.savefig('training.png')
print("\nPlot saved\n")
