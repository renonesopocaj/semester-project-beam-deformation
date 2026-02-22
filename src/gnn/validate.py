# Final evaluation
model.eval()
target_stats.to("cpu")

true_rollout, pred_rollout = do_rollout(
    model=model,
    test_loader=test_dataloader,
    device=device,
    node_stats=node_stats,
    edge_stats=edge_stats,
    target_stats=target_stats,
    dt=dt,
)

# Compute final metrics
final_rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t)
print(f"\nFinal rollout MAE: {final_rollout_error:.6f}")

# Compute per-timestep errors
timestep_errors = []
for true_graph, pred_graph in zip(true_rollout, pred_rollout):
    error = mae(true_graph.x_pos_t, pred_graph.x_pos_t)
    timestep_errors.append(error.item())

plt.figure(figsize=(10, 6))
plt.plot(timestep_errors)
plt.xlabel("Timestep")
plt.ylabel("Position Error (MAE)")
plt.title("Prediction Error Over Rollout")
plt.grid(True)
plt.savefig("rollout_error_over_time.png")
plt.show()

print(f"Average per-timestep error: {np.mean(timestep_errors):.6f}")

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