# Project 4: Community

## Citation and License
This project belongs to Department of Computer Science, Columbia University. It may be used for educational purposes under Creative Commons **with proper attribution and citation** for the author TAs **Raavi Gupta (First Author), Divyang Mittal and the Instructor, Prof. Kenneth Ross**.

## Summary

Course: COMS 4444 Programming and Problem Solving (Fall 2024)  
Problem Description: https://www.cs.columbia.edu/~kar/4444f24/node21.html  
Course Website: https://www.cs.columbia.edu/~kar/4444f24/  
University: Columbia University  
Instructor: Prof. Kenneth Ross  
Project Language: Python

### TA Designer for this project

Raavi Gupta

### Teaching Assistants for Course
1. Divyang Mittal
2. Raavi Gupta

## Installation

To install tkinter on macOS, run the following command:
```bash
brew install python-tk@3.X
```
For Windows, tkinter can be installed using pip:
```bash
pip install tk
```

## Usage
To view all options, use `python community.py -h`.

```bash
python community.py  [--seed] [--num_members] [--num_turns] [--num_abilities] [--seed_ability] [--seed_task_difficulty] [--group_task_distribution] [--group_abilities_distribution] [--log] [--gui] [--gi]
```
### Description of Flags:

- **seed**: The seed for the entire game.
- **num_members**: Number of members in the community.
- **num_turns**: Total number of turns.
- **num_abilities**: Number of abilities present (between 3-8).
- **seed_ability**: Separate seed for ability generation (optional).
- **seed_task_difficulty**: Separate seed for task difficulty generation (optional).
- **group_task_distribution**: Group number whose task difficulty distribution function will be used.
- **task_distribution_difficulty**: Difficulty level of tasks. One of easy, medium or hard.
- **group_abilities_distribution**: Group number whose ability distribution function will be used.
- **abilities_distribution_difficulty**: Difficulty level for the abilities. One of easy, medium or hard.
- **log**: If this flag is present, print statements won't appear on the console and will be saved to the `log-results/` folder.
- **gui**: Activates the simulator with a graphical user interface (GUI).
- **gi**: Specifies the number of players from each group in the community. [i] ranges from 1-10, with $\sum_{i=1}^{10} g_{i} = $`num_members`. If the sum is less than `num_members`, the remaining players will be from team 0 by default. The sum cannot exceed the specified `num_members`.

## Example Usage

```bash
python .\community.py --num_members 40 --num_turns 100 --num_abilities 8 --g1 20 --gui --log
```

## Submission

To submit the code for each deliverable, open a pull request to merge your proposed `preferences.py` and `distributions.py` files for player preferences and task/abilities distributions. These two Python files, named `preferences.py` and `distributions.py`, should be located in the `teams/team_{group-number}` folder (e.g., `teams/team_0`, where the group number corresponds to the allotted group number for this project). The Python files should contain the following functions:

**preferences.py**

1. `phaseIpreferences(player, community, global_random)`: This function defines your strategy for choosing preferences in phase I of the game. It should return a list of lists, where each element is a `[task_index, partner_id]` pair. This represents your bid to form partnerships with players on tasks.
   
2. `phaseIIpreferences(player, community, global_random)`: This function defines your strategy for bidding on tasks in phase II. It should return a list of task indices from `community.tasks` that the player chooses to bid on during phase II in a turn.

**distributions_{easy/medium/hard}.py**

1. `ability_distribution(num_abilities, seed, player_id, global_random)`: This function defines the ability distribution for a player, given the number of abilities and the player ID. For a given seed, the function should output the same list of abilities across runs to ensure reproducibility. Refer to the sample in the `team_0` folder for guidance.

2. `task_difficulty_distribution(num_abilities, seed, task_generation_id, global_random)`: This function defines the task difficulty distribution, given the number of abilities and task generation ID. For a given seed, the function should output the same list of task difficulties across runs to ensure reproducibility. Refer to the sample in the `team_0` folder for guidance.

A sample submission format for team 0 (a non-existent team) has been provided in the `teams` folder for reference.
