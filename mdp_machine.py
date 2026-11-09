
# Experiment: Markov Decision Process using Value Iteration

# Define the machine states
states = ["Good", "Worn", "Broken"]

# Define the available actions and their rewards and transitions
mdp = {
    "Good": {
        "Run": {
            "reward": 10,
            "transitions": {"Good": 0.8, "Worn": 0.2}
        },
        "Service": {
            "reward": -4,
            "transitions": {"Good": 1.0}
        }
    },

    "Worn": {
        "Run": {
            "reward": 6,
            "transitions": {"Worn": 0.6, "Broken": 0.4}
        },
        "Service": {
            "reward": -4,
            "transitions": {"Good": 1.0}
        }
    },

    "Broken": {
        "Run": {
            "reward": 0,
            "transitions": {"Broken": 1.0}
        },
        "Service": {
            "reward": -12,
            "transitions": {"Good": 1.0}
        }
    }
}

# Discount factor
gamma = 0.9

# Initialize state values to zero
V = {state: 0 for state in states}

# Value Iteration
iteration = 0
tolerance = 1e-10

print("MDP USING VALUE ITERATION")
print("-" * 55)

while True:
    new_V = {}
    policy = {}

    for state in states:
        action_values = {}

        # Calculate value for each action
        for action, details in mdp[state].items():
            reward = details["reward"]

            future_value = sum(
                probability * V[next_state]
                for next_state, probability
                in details["transitions"].items()
            )

            action_values[action] = reward + gamma * future_value

        # Select the action with maximum value
        best_action = max(action_values, key=action_values.get)

        new_V[state] = action_values[best_action]
        policy[state] = best_action

    # Calculate maximum change in values
    delta = max(abs(new_V[s] - V[s]) for s in states)

    V = new_V
    iteration += 1

    # Display first three iterations
    if iteration <= 3:
        print(f"\nIteration {iteration}")
        for state in states:
            print(f"{state}: Value = {V[state]:.2f}, "
                  f"Action = {policy[state]}")

    # Stop when values become stable
    if delta < tolerance:
        break

# Display final results
print("\n" + "-" * 55)
print("FINAL OPTIMAL VALUES AND POLICY")
print("-" * 55)

for state in states:
    print(f"{state:10} Value = {V[state]:.2f} | "
          f"Best Action = {policy[state]}")

print("\nTotal iterations:", iteration)
print("\nOptimal Policy:")
print("Good   -> Run")
print("Worn   -> Service")
print("Broken -> Service")
