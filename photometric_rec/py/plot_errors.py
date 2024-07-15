import matplotlib.pyplot as plt

# Read the error values from the file
file_path = '/Users/ekole/Dev/gut_slam/photometric_rec/py/errors.txt'
with open(file_path, 'r') as file:
    errors = [float(line.strip()) for line in file]

# Generate the number of iterations
iterations = list(range(1, len(errors) + 1))

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(iterations, errors, linestyle='-', color='b')
plt.title('Photometric Error vs. Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Photometric Error (*100)')
plt.grid(True)
plt.show()