from stable_baselines3 import PPO

model = PPO.load("bytefight_logs/ppo_bytefight_final.zip")
policy = model.policy

print(policy)  # high‑level summary

# Feature extractor output size
print("Feature extractor output dim:", policy.features_extractor.features_dim)

# Policy/value nets and action head
print("Policy network:", policy.mlp_extractor.policy_net)
print("Action head:", policy.action_net)
print("Value head:", policy.value_net)

# Input/output shapes
print("Observation shape:", policy.observation_space.shape)
print("Action space size:", policy.action_space.n)

# Every parameter name → shape
for name, param in policy.state_dict().items():
    print(f"{name:50} → {tuple(param.shape)}")