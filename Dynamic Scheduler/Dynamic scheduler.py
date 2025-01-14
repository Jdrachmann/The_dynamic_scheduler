from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum, value

# Define the problem
problem = LpProblem("Production_Schedule_Optimization", LpMinimize)

# Decision variables
# Days and exercises
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
exercises = ["Bench_Press", "Deadlift", "Squat", "Pulldown", "Curls", "Tricep"]
muscles = ["Chest", "Back", "Legs", "Arms"]

# Fatigue contribution for each exercise (initial fatigue per muscle group)
fatigue = {
    "Bench_Press": {"Chest": 1, "Back": 0, "Legs": 0, "Arms": 0.5},
    "Deadlift": {"Chest": 0, "Back": 0.5, "Legs": 1, "Arms": 0},
    "Squat": {"Chest": 0, "Back": 0.5, "Legs": 1, "Arms": 0},
    "Pulldown": {"Chest": 0, "Back": 1, "Legs": 0, "Arms": 0.5},
    "Curls": {"Chest": 0, "Back": 0, "Legs": 0, "Arms": 1},
    "Tricep": {"Chest": 0, "Back": 0, "Legs": 0, "Arms": 1},
}


# Target volumes for each muscle group over the schedule period
target_volume = {"Chest": 8, "Back": 10, "Legs": 6, "Arms": 4}

# Fatigue decay rate (percentage of fatigue carried over to the next day)
decay_rate = 0.5

# Absolute deviation variables for each muscle group
deviation = {
    m: LpVariable(f"Deviation_{m}", lowBound=0, cat="Continuous") for m in muscles
}

# Fatigue variables: fatigue level for each muscle on each day
f = {
    (d, m): LpVariable(f"Fatigue_{m}_{d}", lowBound=0, cat="Continuous")
    for d in days
    for m in muscles
}

# Decision variables: whether to perform an exercise on a given day (0 or 1)
x = {
    (d, e): LpVariable(f"Sets_{e}_{d}", lowBound=0, cat="Integer")
    for d in days
    for e in exercises
}

# Achieved total volume for each muscle group
achieved_volume = {
    m: LpVariable(f"Achieved_Volume_{m}", lowBound=0, cat="Continuous") for m in muscles
}

# Fatigue dynamics constraints
for i, d in enumerate(days):
    for m in muscles:
        # Fatigue on day d = (decay from previous day) + (fatigue added by exercises on this day)
        if i == 0:  # First day
            problem += f[d, m] == 0, f"Initial_Fatigue_{m}_{d}"
        else:  # Subsequent days
            previous_day = days[i - 1]
            problem += (
                f[d, m]
                == decay_rate * f[previous_day, m]
                + lpSum(x[previous_day, e] * fatigue[e][m] for e in exercises),
                f"Cumulative_Fatigue_{m}_{d}",
            )

# For each muscle m, Achieved_Volume[m] is the sum of volume from all exercises across all days
for m in muscles:
    problem += (
        achieved_volume[m]
        == lpSum(x[d, e] * fatigue[e][m] for d in days for e in exercises),
        f"Total_Achieved_Volume_{m}",
    )
# Fatigue limit constraints: fatigue on any day for any muscle cannot exceed 100
for d in days:
    for m in muscles:
        problem += f[d, m] <= 20, f"Fatigue_Limit_{m}_{d}"

# Limit exercises per day (e.g., 1 exercise per day)
for d in days:
    problem += lpSum(x[d, e] for e in exercises) <= 10, f"One_Exercise_Per_Day_{d}"


# Objective function: Minimize the sum of squared differences between achieved and target volumes
problem += (
    lpSum((-achieved_volume[m] + target_volume[m]) for m in muscles),
    "Minimize_Volume_Deviation",
)

# Solve the problem
problem.solve()


print("\nOptimal Workout Plan (Sets):")
for d in days:
    day_sets = []
    for e in exercises:
        val = x[d, e].varValue
        if val is not None and val > 0:  # Only print exercises with > 0 sets
            day_sets.append(f"{e} x {val}")
    if day_sets:
        print(f"{d}: {', '.join(day_sets)}")

print("\nFatigue Levels:")
for d in days:
    for m in muscles:
        print(f"{d} - {m}: {f[d, m].varValue}")

print("\nAchieved Volumes:")
for m in muscles:
    print(f"{m}: {achieved_volume[m].varValue}")

print("\nObjective function value:", value(problem.objective))
print("Objective expression:", problem.objective)
