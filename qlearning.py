import numpy as np
import random


# Create a 4 x 4 grid
grid_size = 4

# Number of states
states = grid_size * grid_size

# Four possible actions
# 0 = Up
# 1 = Down
# 2 = Left
# 3 = Right
actions = 4


# Create Q-table
q_table = np.zeros((states, actions))


# Learning parameters
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2

episodes = 1000


# Goal state
goal_state = 15


# Function to move the agent
def move(state, action):

    row = state // grid_size
    column = state % grid_size

    # Up
    if action == 0:
        row = max(0, row - 1)

    # Down
    elif action == 1:
        row = min(grid_size - 1, row + 1)

    # Left
    elif action == 2:
        column = max(0, column - 1)

    # Right
    elif action == 3:
        column = min(grid_size - 1, column + 1)

    new_state = row * grid_size + column

    return new_state


# Training
for episode in range(episodes):

    state = 0

    while state != goal_state:

        # Exploration or exploitation
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, actions - 1)
        else:
            action = np.argmax(q_table[state])

        # Move to new state
        new_state = move(state, action)

        # Give reward
        if new_state == goal_state:
            reward = 100
        elif new_state == state:
            reward = -10
        else:
            reward = -1

        # Find best future action
        best_future_q = np.max(q_table[new_state])

        # Update Q-value
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward
            + discount_factor * best_future_q
            - q_table[state, action]
        )

        # Move to next state
        state = new_state


# Display Q-table
print("Q-Table:")
print(q_table)


# Test the learned path
state = 0
path = [state]

while state != goal_state:

    action = np.argmax(q_table[state])

    state = move(state, action)

    path.append(state)

    if len(path) > 20:
        break


print("\nLearned Path:")
print(path)


# Display path as grid
print("\nGrid Path:")

for row in range(grid_size):

    for column in range(grid_size):

        state = row * grid_size + column

        if state in path:
            print(" * ", end="")
        else:
            print(" . ", end="")

    print()