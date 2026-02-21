# Training loop using the Trainer class
epoch_bar = tqdm(range(total_epochs), desc="Epochs")
for epoch in epoch_bar:
    # Training phase
    for batch in tqdm(train_dataloader, desc="Batches", leave=False):
        batch = batch.to(device)

        # Train step
        trainer.train(
            graph=batch,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
        )

    # End of epoch logging
    trainer.epoch_end(epoch=epoch)

    # Validation phase
    if epoch % validation_interval == 0:
        trainer.test(
            test_loader=test_dataloader,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            dt=dt,
            epoch=epoch,
        )
        trainer.save_model(epoch=epoch)

        epoch_bar.set_postfix({
            "train_loss": f"{trainer.loss:.6f}",
            "rollout_error": f"{trainer.rollout_all_step_error:.6f}"
        })

        print(f"\nEpoch {epoch}: Train Loss = {trainer.loss:.6f}, Rollout Error = {trainer.rollout_all_step_error:.6f}")
    else:
        epoch_bar.set_postfix({"train_loss": f"{trainer.loss:.6f}"})

print("\nTraining complete!")
print(f"Models saved to: {trainer.model_dir}")