sarsa_agent = SARSAAgent(env, sarsa_config)

print("Starting training...")
train_result, eval_result = sarsa_agent.train()
print("Training completed!")

print_summary(train_result, title = "Training")
print("\n")
print_summary(eval_result, title = "Evaluation")