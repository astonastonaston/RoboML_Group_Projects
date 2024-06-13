import matplotlib.pyplot as plt

# Dummy data for demonstration purposes
# Replace these with actual data from your training logs
eval_steps1 = list(range(0, 925, 25))  # Evaluation steps/epochs
eval_success_rate = [0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.375, 0.25, 0.125, 0.0, 0.0, 0.5, 0.666, 0.818, 0.444, 0.818, 0.8, 0.636, 0.769, 0.875, 0.875, 0.866, 1.0, 1.0, 0.923, 0.941, 1.0, 1.0, 0.846, 0.933, 1.0, 0.866, 0.937, 0.941, 1.0, 1.0, 0.785]  # Evaluation success rate
eval_steps2 = list(range(0, 950, 25))  # Evaluation steps/epochs
eval_episodic_return = [3.49, 26.54, 28.72, 29.92, 35.75, 32.69, 31.64, 33.48, 33.24, 36.32, 31.28, 23.76, 20.01, 18.61, 28.95, 15.68, 16.07, 15.89, 14.80, 10.52, 12.87, 13.78, 11.69, 9.47, 9.41, 14.54, 11.42, 9.89, 9.80, 15.02, 11.38, 9.63, 13.31, 11.25, 9.74, 10.52, 11.65, 13.00]
# print(len(eval_success_rate), len(eval_episodic_return))
# Plotting the success rate
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(eval_steps1, eval_success_rate, label='Success Rate', marker='o')
plt.xlabel('Training Epochs')
plt.ylabel('Success Rate')
plt.title('Success Rate over Training Steps')
plt.legend()
plt.grid(True)

# Plotting the episodic return
plt.subplot(1, 2, 2)
plt.plot(eval_steps2, eval_episodic_return, label='Episodic Return', marker='o', color='orange')
plt.xlabel('Training Epochs')
plt.ylabel('Episodic Return')
plt.title('Episodic Return over Training Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
