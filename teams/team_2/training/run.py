import subprocess


def run(_action):
    result = subprocess.run(
        [
            "python",
            "community.py",
            "--num_members",
            "40",
            "--g2",
            "40",
            "--num_turns",
            "100",
        ],
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().split("\n")
    for line in lines:
        searchstr = "Total tasks completed: "
        if line.startswith(searchstr):
            completed_tasks = int(line[len(searchstr) :])
            print(f"result: {completed_tasks}")
            return completed_tasks

    raise Exception("something broke")


if __name__ == "__main__":
    run()
