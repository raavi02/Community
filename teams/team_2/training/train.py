import numpy as np
from scipy.optimize import linear_sum_assignment
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    from run import run
else:
    from teams.team_2.training.run import run


class TaskScorerNN(nn.Module):
    def __init__(self, task_feature_size, player_state_size, hidden_size):
        super(TaskScorerNN, self).__init__()
        self.fc1 = nn.Linear(task_feature_size + player_state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Outputs a single score for a task

    def forward(self, task_features: torch.Tensor, player_state: torch.Tensor):
        # Concatenate task features and player state
        combined = torch.cat(
            [
                task_features if task_features.ndim > 1 else task_features.view(-1),
                player_state,
            ],
            dim=-1,
        )
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        score = self.fc3(x)  # Outputs score
        return score


class RestDecisionNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RestDecisionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output for rest score

    def forward(self, features):
        x = F.relu(self.fc1(features))
        score = self.fc2(x)
        return score


def evaluate_fitness(task_model: nn.Module, rest_model: nn.Module):
    # for task in task_environment:
    # action = model(task.features, task.state)
    result = run(task_model, rest_model)
    # result = task_environment.run(action)  # Simulates environment behavior
    return result


def crossover(parent1, parent2, is_task):
    if is_task:
        child = TaskScorerNN(task_feature_size, player_state_size, hidden_size)
    else:
        child = RestDecisionNN(player_state_size + 1, hidden_size)

    for param1, param2, child_param in zip(
        parent1.parameters(), parent2.parameters(), child.parameters()
    ):
        child_param.data.copy_((param1.data + param2.data) / 2)  # Weighted average
    return child


def mutate(model, mutation_rate=0.1):
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.01  # Small random noise


def select_parents(population, fitness_scores):
    best = sorted(
        list(zip(population, fitness_scores)), key=lambda x: x[1], reverse=True
    )

    return [b[0] for b in best][: len(population) // 2]


def task_scorer():
    pass


def rest_scorer():
    pass


class Task:
    def __init__(self, features: torch.Tensor, state=None):
        self.features = features
        self.state = state
        if not state:
            self.state = torch.zeros(features.shape[0])


max_generations = 5
hidden_size = 64
task_feature_size = 1
pop_size = 10
player_state_size = 7

population = [
    (
        TaskScorerNN(task_feature_size, player_state_size, hidden_size),
        RestDecisionNN(player_state_size + task_feature_size, hidden_size),
    )
    for _ in range(pop_size)
]

for generation in range(max_generations):
    # Evaluate fitness
    fitness_scores = [
        evaluate_fitness(task_model, rest_model)
        for task_model, rest_model in population
    ]

    # Select parents
    parents = select_parents(population, fitness_scores)

    # Generate offspring
    offspring = []
    # for now, only create offspring from TaskScorerNN
    for _ in range(len(parents) // 2):
        parent1, parent2 = random.sample(parents, 2)
        child_task = crossover(parent1[0], parent2[0], is_task=True)
        child_rest = crossover(parent1[1], parent2[1], is_task=False)
        mutate(child_task)
        mutate(child_rest)
        offspring.append((child_task, child_rest))

        parent1, parent2 = random.sample(parents, 2)
        child_task = crossover(parent1[0], parent2[0], is_task=True)
        child_rest = crossover(parent1[1], parent2[1], is_task=False)
        mutate(child_task)
        mutate(child_rest)
        offspring.append((child_task, child_rest))

    # Replace population
    population = parents + offspring
    print(f"{len(population)} members")

    print(f"{generation}: fitness scores: {sorted(fitness_scores, reverse=True)[:3]}")


best_model = select_parents(population, fitness_scores)[0]

torch.save(best_model, "best_weigths.pth")
print('best model weights saved in "best_weights.pth"')

# task_scores = [task_scorer(task, player_state) for task in tasks]
# rest_score = rest_scorer(player_state)
# combined_scores = torch.cat([task_scores, rest_score.unsqueeze(0)])
# action = torch.argmax(combined_scores).item()
