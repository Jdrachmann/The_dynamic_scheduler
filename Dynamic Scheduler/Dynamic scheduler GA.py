import random
import math
import pdb
from deap import base, creator, tools
from functools import partial
import pandas as pd
import os
import random
import itertools 


os.chdir(os.path.dirname(os.path.abspath("Dynamic scheduler GA.py")))
# Input values
POPULATION_SIZE  = 5000
# Fysiological variables

decay_rate = 0.5 # How fast does fatigue decay
alpha = 0.2  # scale factor for how strongly fatigue reduces effective volume
max_sets_per_day = 5 # Initialize for max sets pr exercise
threshold_shuffle = 5 # After 5 tries with no improvement try shuffling the workout days
shuffle_keep = 20 # How many of the best shuffles indivduals should be kept
elite_size = 10  # Keep top 50 individuals in every generation
indpb = 0.1 # probability of chance of sets

# Reading input sheets
file_path_folder = "Dynamic scheduler/"
file_path_results = file_path_folder+"Results/"
file_path_schedule = file_path_folder+"DayAvailability.xlsx"
file_workout_history = file_path_folder+"WorkoutHistory.xlsx"
df_exer = pd.read_excel(file_path_folder+"ExerciseInfo.xlsx", sheet_name=0)

def load_workout_history(file_path):
    """
    Load workout history from an Excel file and convert it into a usable format for fatigue calculation.
    - Returns sets by day and exercises for fatigue calculation.
    """
    # Load the Excel file
    df = pd.read_excel(file_path, header=None)

    # Extract the list of exercises from the column headers (excluding the 'Day' column)
    exercises = df.columns[1:].tolist()

    # Extract the days and sets data
    days = df.iloc[:, 0].dropna().tolist()
    sets_by_day = df.iloc[:, 1:].values.tolist()

    # Convert NaN values to 0 for safety
    sets_by_day =  [[int(sets) if pd.notna(sets) else 0 for sets in row] for row in sets_by_day[1:]]


    return days, exercises, sets_by_day

def load_workout_schedule(file_path):
    """
    Load the workout schedule from an Excel file and structure it for GA optimization.
    Extracts:
    - Time availability per day
    - Desired volume per muscle group
    - Current exercise schedule (days, exercises, sets)
    """
    # Load the Excel file
    df = pd.read_excel(file_path, header=None)
    
    # 1. Extract Time Availability (First Row)
    time_available = dict(zip(df.iloc[0, :7], df.iloc[1, :7]))

    # 2. Extract Desired Volume (Row 3 and 4)
    desired_volume = dict(zip(df.iloc[4, 0:].dropna(), df.iloc[5, 0:].dropna()))

    # 3. Extract Exercises and Sets (Starting from Row 6)
    exercises = df.iloc[7, 1:].tolist()
    days = df.iloc[0, :7].tolist()
    sets_by_day = df.iloc[7:, 1:].fillna(0).values.tolist()
    #Remove first list
    sets_by_day =  [[int(sets) if pd.notna(sets) else 0 for sets in row] for row in sets_by_day[1:]]

    # Ensure the model uses only these exercises
    exercise_data = {
        "time_available": time_available,
        "desired_volume": desired_volume,
        "days": days,
        "exercises": exercises,
        "sets_by_day": sets_by_day
    }

    return exercise_data

def create_individual_from_schedule(current_schedule):
    """
    Convert the current schedule into an individual.
    """
    # Flatten the current schedule into a list (chromosome format)
    return [sets for day in current_schedule for sets in day]

# Load schedule and avalability
workout_data = load_workout_schedule(file_path_schedule)
days_history, exercises_history, sets_by_day_history = load_workout_history(file_workout_history)
time_available = workout_data['time_available']
target_volume = workout_data['desired_volume']
day_names = workout_data['days']
exercises = workout_data['exercises']
sets_by_day = workout_data['sets_by_day']
sets_by_day_ori_sched = sets_by_day 

workout_days = list()
for index, (key, value) in enumerate(time_available.items()):
    if value > 0:
        workout_days.append(key)



current_schedule = create_individual_from_schedule(sets_by_day)

print("Flattened Current Schedule:", current_schedule)

# Creates the size of the problem
NUM_EXERCISES = len(exercises)
NUM_DAYS = len(day_names)
CHROMOSOME_LENGTH = NUM_DAYS * NUM_EXERCISES

# Create relevant dictionaries
muscles = list(df_exer.columns[2:])
time_per_exercise = {}
fatigue = {}
contrib_dict = {}

for i in exercises:
    row = df_exer.loc[df_exer['exercise'] == i]
    time_per_exercise[i] = row['time_per_set'].item()
     # 2) Build a dict for muscle contributions automatically
    contrib_dict = {}

    # Loop over every muscle column
    for m_col in muscles:
        val = row[m_col].item()
        # If it's > 0, add it to the dictionary
        if val > 0:
            contrib_dict[m_col.capitalize()] = val
    fatigue[i] = contrib_dict

# -----------------------------
# Diminishing-Returns Parameters
# -----------------------------
# For example, a saturating exponential:

def diminishing_effective_sets(n_sets):
    """
    If each new set is 2/3 of the previous, 
    we sum: 1.0 + (2/3) + (2/3)^2 + ...
    up to n_sets times.
    """
    if n_sets <= 0:
        return 0.0
    
    r = 2/3
    return 1.0 * (1 - r**n_sets) / (1 - r)
# ----------------------------
# 2) GENETIC ALGORITHM SETUP
# ----------------------------


# Create the fitness and individual classes (if not already done)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
creator.create("Individual", list, fitness=creator.FitnessMin)

# Ensure the toolbox is registered properly
toolbox = base.Toolbox()
toolbox.register("attr_sets", random.randint, 2, 5)  # Valid range for sets
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_sets, n=NUM_DAYS * NUM_EXERCISES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# 2.1 Define fitness as a single objective ("Minimize" or "Maximize" in DEAP)
# We'll define it as Minimizing "Total Deviation" from targets + Penalties


# ----------------------------
# 3) FITNESS EVALUATION
# ----------------------------

def reshape_chromosome(individual):
    """Convert a flat chromosome into a 2D structure (sets per day and exercise)."""
    sets_by_day = []
    idx = 0

    for i in range(NUM_DAYS):
        day_slice = individual[idx : idx + NUM_EXERCISES]
        sets_by_day.append(day_slice)
        idx += NUM_EXERCISES
    return sets_by_day

def calculate_daily_fatigue(sets_by_day):
    """Calculate the fatigue levels for each muscle on each day."""
    
    combined_sets_his_fut = [*sets_by_day_history, *sets_by_day]
    daily_fatigue = [[float(0) for _ in muscles] for _ in range(NUM_DAYS*2)]

    
    for d in range(1, NUM_DAYS*2):  # Day 0 already initialized
        for mi, m in enumerate(muscles):
            # Fatigue decays from the previous day
            daily_fatigue[d][mi] = decay_rate * daily_fatigue[d - 1][mi]

            # Add fatigue contribution from previous day's sets
            for ei, e in enumerate(exercises):
                day_minus_1_sets = combined_sets_his_fut[d - 1][ei]
                daily_fatigue[d][mi] += day_minus_1_sets * fatigue[e].get(m, 0)

    daily_fatigue = daily_fatigue[7:14] # WOrk indexes skal automatiseres
    return daily_fatigue


def calculate_effective_volume(sets_by_day, daily_fatigue):
    """Calculate the effective volume considering fatigue and diminishing returns."""
    achieved_vol = {m: 0.0 for m in muscles}
    effective_vol = {m: 0.0 for m in muscles}

    for d in range(NUM_DAYS):
        for ei, e in enumerate(exercises):

            n_sets = sets_by_day[d][ei]

            # Gather muscle fatigue factors
            muscle_factors = []
            for m in muscles:
                if fatigue[e].get(m, 0) > 0:
                    current_f = daily_fatigue[d][muscles.index(m)]
                    raw_factor = 1.0 - alpha * current_f
                    muscle_factors.append(max(0.0, raw_factor))

            # If no muscle involvement, skip
            if not muscle_factors:
                continue

            # Bottleneck effect from fatigue
            exercise_factor = min(muscle_factors)

            # Distribute volume across muscles
            for m in muscles:
                # Apply diminishing returns
                eff_sets = diminishing_effective_sets(n_sets)
                base_vol = fatigue[e].get(m, 0)
                if base_vol > 0:
                    volume_gained = eff_sets * base_vol * exercise_factor
                    achieved_vol[m] += n_sets * base_vol
                    effective_vol[m] += volume_gained

    return achieved_vol, effective_vol


def calculate_time_penalty(sets_by_day):
    """Calculate the time penalty for exceeding daily availability.
    Setup time is added only when an exercise is performed.
    """
    penalty = 0
    total_time_used = 0

    for d, day_name in enumerate(day_names):
        time_used_today = 0

        # Loop through each exercise and calculate time only if performed
        for ei in range(NUM_EXERCISES):
            sets = sets_by_day[d][ei]
            if sets > 0:
                # Time for sets + additional setup time per performed exercise
                time_used_today += sets * time_per_exercise[exercises[ei]]
                time_used_today += time_per_exercise[exercises[ei]]  # Setup time added only if sets > 0

        # Apply penalty for exceeding available time
        if time_used_today > time_available[day_name]:
            penalty += 100.0 * (time_used_today - time_available[day_name])
        elif time_available[day_name] - time_used_today < 5:
            penalty += 100.0 * (time_available[day_name] - time_used_today)
        

        total_time_used += time_used_today

    return penalty, total_time_used


def calculate_deviation(achieved_vol):
    """Calculate the deviation from the target volume for each muscle."""
    total_deviation = 0.0
    for m in muscles:
        diff = target_volume[m] - achieved_vol[m]
        if diff > 0:
            total_deviation += abs(diff)
    return total_deviation

def calculate_time_used_per_day(sets_by_day):
    """
    Calculate time used per day including setup time only when an exercise is performed.
    Setup time is equal to the time per exercise.
    """
    time_used_per_day = []

    for d, day_name in enumerate(day_names):
        time_used_today = 0
        for ei in range(NUM_EXERCISES):
            sets = sets_by_day[d][ei]
            if sets > 0:  # Only add setup time if the exercise is performed
                time_used_today += sets * time_per_exercise[exercises[ei]]
                # Adding setup time (same as time_per_exercise)
                time_used_today += time_per_exercise[exercises[ei]]

        time_used_per_day.append(time_used_today)

    return time_used_per_day

def evaluate(individual):
    """
    Modularized evaluation function using helper functions.
    """
  
    # Step 1: Reshape chromosome into day-by-day structure
    sets_by_day = reshape_chromosome(individual)
    
    # Step 2: Calculate fatigue levels
    daily_fatigue = calculate_daily_fatigue(sets_by_day)

    # Step 3: Calculate effective and achieved volumes
    achieved_vol, effective_vol = calculate_effective_volume(sets_by_day, daily_fatigue)
    total_effective_volume = sum(effective_vol.values())

    # Step 4: Apply time penalty
    penalty, total_time_used = calculate_time_penalty(sets_by_day)

    # Step 5: Calculate deviation from the target volumes
    total_deviation = calculate_deviation(achieved_vol)

    # Step 6: Fitness = deviation + penalty
    fitness_val = total_deviation**3 + penalty - total_effective_volume**1.5

    # Return only the fitness value for DEAP
    return (fitness_val,)


def generate_combinations(pop,stagnant_generations,threshold,gen):
    """
    Generates all possible unique combinations for a list of lists based on shuffling the lists themselves.
    
    Parameters:
    activities (list of lists): A list of lists where each inner list represents a set of activities.

    Returns:
    list: A list containing all unique combinations of the lists themselves.
    """
    # Trigger shuffle only if the threshold is reached
    
    if stagnant_generations == threshold or gen==1:
        current_best = tools.selBest(pop, 1)[0]
        current_best = reshape_chromosome(current_best)

        shuffled_schedule = list(itertools.permutations(current_best))
        shuffled_individual = create_individual_from_schedule(shuffled_schedule)
        
        # Flatten each sequence into a single list
        flattened = []
    
        # Loop through each sequence in the shuffled individual
        for index in range(int(len(shuffled_individual)/7)):
            sliced_list = shuffled_individual[index*7:(index+1)*7]
            sliced_list=create_individual_from_schedule(sliced_list)
            
            
            flattened.append(sliced_list)
        # Example to convert a raw list to a DEAP individual
        pop_flattened = [creator.Individual(ind) for ind in flattened]
       
        # Calculate fitness after shuffling
        fitnesses = list(map(toolbox.evaluate, flattened))
        for ind, fit in zip(pop_flattened, fitnesses):
            ind.fitness.values = fit

        new_best_individuals = tools.selBest(pop_flattened, shuffle_keep)

        return new_best_individuals
    else:
        return []

def mutUniformSets(individual, workout_days, indpb=0.1):
    """Mutate only workout days by randomly adjusting sets between 2-5."""
    for day_index, day_name in enumerate(day_names):
        if day_name in workout_days:  # Mutate only workout days
            for exercise_index in range(NUM_EXERCISES):
                if random.random() < indpb:
                    individual[day_index * NUM_EXERCISES + exercise_index] = random.randint(0, max_sets_per_day)
        else:
            for exercise_index in range(NUM_EXERCISES):
                individual[day_index * NUM_EXERCISES + exercise_index] = 0
    return (individual,)

# def mutUniformSets(individual, indpb=0.1):
#     """Mutate each gene with probability indpb by assigning a random integer [0..max_sets_per_day]."""
#     for i in range(len(individual)):
#         if random.random() < indpb:
#             individual[i] = random.randint(0, max_sets_per_day)
#     return (individual,)

# ----------------------------
# 4) Genetic Operators
# ----------------------------



# 2.3 Structure initializers
# We can use built-in DEAP tools: mate=tools.cxTwoPoint, mutate=tools.mutUniformInt, etc.
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("evaluate", evaluate)

# number for set for a given day for an exercise
toolbox.register("mutate", partial(mutUniformSets, workout_days=workout_days, indpb=0.1), indpb=0.1)
# toolbox.register("mutate", mutUniformSets, indpb=0.1)    
toolbox.register("select", tools.selTournament, tournsize=3)



def initialize_population_with_schedule(current_schedule, pop_size,indpb):
    """
    Initialize a population where the current schedule is included and others are randomly generated.
    """
    # Convert the provided schedule to a DEAP individual object
    population = [creator.Individual(current_schedule)]  # Convert it here

    # Generate the remaining individuals
    for _ in range(pop_size - 1):
        individual = toolbox.individual()
        for day_index, day_name in enumerate(day_names):
            if day_name in workout_days:  # Mutate only workout days
                for exercise_index in range(NUM_EXERCISES):
                    if random.random() < indpb:
                        individual[day_index * NUM_EXERCISES + exercise_index] = random.randint(0, max_sets_per_day)
            else:
                for exercise_index in range(NUM_EXERCISES):
                    individual[day_index * NUM_EXERCISES + exercise_index] = 0
        population.append(individual)
    
    return population


def main(current_schedule):
    # random.seed(42)
    # Create initial population
    pop = initialize_population_with_schedule(current_schedule, POPULATION_SIZE,indpb)

    # Number of generations
    NGEN = 100
    # Probability of crossover / mutation
    CXPB, MUTPB = 0.7, 0.3

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # ---------------------------
    # Improvement-based stopping
    # ---------------------------
    best_so_far = float("inf")
    no_improvement_count = 0
    IMPROVEMENT_THRESHOLD = 1e-4    # "small" improvement threshold
    MAX_NO_IMPROVE_GENS = 10        # after 10 gens of no improvement, we stop

    for gen in range(NGEN):
        
        # Sort population by fitness and preserve the top individuals
        pop = sorted(pop, key=lambda ind: ind.fitness.values[0])

        # shuffle shuffle

        shuffles = generate_combinations(pop,no_improvement_count,threshold_shuffle, gen)

        elites = pop[:elite_size]
        if shuffles:
            elites += shuffles

        # 1) Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 2) Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 3) Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 4) Evaluate the new offspring
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # 5) Replace population
        pop[:] = offspring + elites 

        # 6) Get current best fitness
        current_best = min(ind.fitness.values[0] for ind in pop)

        # 7) Check for improvement
        if abs(best_so_far - current_best) < IMPROVEMENT_THRESHOLD:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            best_so_far = current_best

       


        # 8) If no improvement for many generations, stop early
        if no_improvement_count >= MAX_NO_IMPROVE_GENS:
            print(f"No improvement for {MAX_NO_IMPROVE_GENS} generations; stopping early.")
            break

        print(f"Gen {gen}: best fitness = {current_best:.6f}, no_improvement_count = {no_improvement_count}")

    # After evolution ends (either we break early or hit NGEN)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is:", best_ind)
    print("Fitness:", best_ind.fitness.values[0])

    # Example: Reshape best solution to day-by-exercise for printing
    best_solution = []
    idx = 0
    for i in range(NUM_DAYS):
        day_slice = best_ind[idx : idx + NUM_EXERCISES]
        best_solution.append(day_slice)
        idx += NUM_EXERCISES

    for d_i, d_name in enumerate(day_names):
        day_plan = []
        for e_i, e_name in enumerate(exercises):
            sets_for_exercise = best_solution[d_i][e_i]
            if sets_for_exercise > 0:
                day_plan.append(f"{e_name} x {sets_for_exercise}")
        if day_plan:
            print(f"{d_name}: {', '.join(day_plan)}")

     # Return the best individual and metrics
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind
    


def save_schedule_to_excel(best_individual, days, exercises, muscles, filename="workout_schedule.xlsx"):
    """
    Save the workout schedule results to an Excel file including:
    - Time used per day next to time availability
    - Target volume under achieved volume
    """

    # Recalculate all metrics for the best individual
    sets_by_day = reshape_chromosome(best_individual)
    daily_fatigue = calculate_daily_fatigue(sets_by_day)
    achieved_volume, effective_volume = calculate_effective_volume(sets_by_day, daily_fatigue)
    penalty, total_time_used = calculate_time_penalty(sets_by_day)
    time_used_per_day = calculate_time_used_per_day(sets_by_day)
    total_deviation = calculate_deviation(effective_volume)

    # Prepare DataFrames for Export
    # 1. Sets Performed
    # Prepare DataFrame for Export (Days as Rows)
    all_data = []

    # Add Original Schedule
    for i, day in enumerate(days):
        all_data.append([f"{day} (Original)"] + sets_by_day_ori_sched[i])

    # Add Optimized Schedule
    for i, day in enumerate(days):
        all_data.append([f"{day} (Optimized)"] + sets_by_day[i])
    
    # Convert to DataFrame
    sets_combined = pd.DataFrame(all_data, columns=["Day"] + exercises)

    # 2. Effective Volume Per Day
    effective_volume_df = pd.DataFrame(effective_volume, index=[days[i] for i in range(len(days))])
    effective_volume_df.index.name = "Day"

    # 3. Fatigue Levels
    fatigue_df = pd.DataFrame(daily_fatigue, index=days, columns=muscles)
    fatigue_df.index.name = "Day"

    # 4. Time Used vs. Time Available
    time_data = {
        "Time Used (min)": time_used_per_day,
        "Time Available (min)": [time_available[day] for day in days]
    }
    time_df = pd.DataFrame(time_data, index=days)
    time_df.index.name = "Day"

    # 5. Achieved Volume and Target Volume
    achieved_volume_df = pd.DataFrame([achieved_volume, target_volume], 
                                      index=["Achieved Volume", "Target Volume"])

    # 6. Penalty and Total Time Used Summary
    summary_df = pd.DataFrame({
        "Penalty": [penalty],
        "Total Time Used": [total_time_used],
        "Total Deviation": [total_deviation]
    })

    # Exporting DataFrames to Excel
    with pd.ExcelWriter(file_path_results+filename) as writer:
        sets_combined.to_excel(writer, sheet_name="Sets_Performed")
        effective_volume_df.to_excel(writer, sheet_name="Effective_Volume")
        fatigue_df.to_excel(writer, sheet_name="Fatigue")
        time_df.to_excel(writer, sheet_name="Time_Used_vs_Available")
        achieved_volume_df.to_excel(writer, sheet_name="Achieved_vs_Target_Volume")
        summary_df.to_excel(writer, sheet_name="Summary")

    print(f"✅ Schedule successfully saved to: {file_path_results+filename}")


if __name__ == "__main__":
    best_ind = main(current_schedule)  # Runs the GA and returns the best solution
    save_schedule_to_excel(
        best_individual=best_ind, 
        days=day_names, 
        exercises=exercises, 
        muscles=muscles
    )


