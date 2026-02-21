from IPython.display import Image, display

# Display the training history plot
display(Image(filename='training_history.png'))

# Plot training history from the trainer
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Training loss
ax.plot(trainer.train_history)
ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss (MSE)")
ax.set_title("Training Loss Over Time")
ax.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print(f"Total training steps: {len(trainer.train_history)}")
print(f"Final training loss: {trainer.train_history[-1]:.6f}")
if hasattr(trainer, 'rollout_all_step_error'):
    print(f"Final rollout error: {trainer.rollout_all_step_error:.6f}")