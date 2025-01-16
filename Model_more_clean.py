import random
import math
import pdb
from deap import base, creator, tools
from functools import partial
import pandas as pd
import os
import random
import itertools 

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
    Load the workout schedule from an Excel file.
    """
    df = pd.read_excel(file_path, sheet_name=0)
    sets_by_day = df.iloc[7:, 1:].fillna(0).values.tolist()

    return sets_by_day

def create_individual_from_schedule(current_schedule):
    """
    Convert the current schedule into an individual.
    """
    return [sets for day in current_schedule for sets in day]


def generate_combinations(pop,stagnant_generations,threshold,gen,NUM_DAYS,NUM_EXERCISES,toolbox,shuffle_keep):
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
        current_best = reshape_chromosome(current_best, NUM_DAYS, NUM_EXERCISES)

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

def reshape_chromosome(individual, NUM_DAYS,NUM_EXERCISES):
    """
    Reshape a flat chromosome into a structured schedule.
    """
    sets_by_day = []
    idx = 0
    for _ in range(NUM_DAYS):
        day_slice = individual[idx: idx + NUM_EXERCISES]
        sets_by_day.append(day_slice)
        idx += NUM_EXERCISES
    return sets_by_day

def diminishing_effective_sets(n_sets):
    """
    Function that calculates the diminishing returns of performing a lot of sets in the same day
    """
    if n_sets <= 0:
        return 0.0
    
    r = 2/3
    return 1.0 * (1 - r**n_sets) / (1 - r)

def calculate_effective_volume(sets_by_day, daily_fatigue,NUM_DAYS,muscles,exercises,fatigue,alpha):
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

def calculate_deviation(achieved_vol, muscles):
    """Calculate the deviation from the target volume for each muscle."""
    total_deviation = 0.0
    for m in muscles:
        diff = target_volume[m] - achieved_vol[m]
        if diff > 0:
            total_deviation += abs(diff)
    return total_deviation

def calculate_time_penalty(sets_by_day,day_names,exercises,NUM_EXERCISES,time_per_exercise,time_available):
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

def calculate_time_used_per_day(sets_by_day,day_names,exercises,NUM_EXERCISES,time_per_exercise):
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

def mutUniformSets(individual, workout_days, day_names, NUM_EXERCISES, max_sets_per_day, indpb=0.1):
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



def initialize_population_with_schedule(current_schedule, pop_size,indpb, toolbox, day_names, workout_days, max_sets_per_day, NUM_EXERCISES):
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

def calculate_daily_fatigue(sets_by_day, sets_by_day_history,NUM_EXERCISES,NUM_DAYS,muscles,decay_rate,exercises,fatigue):
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

def generate_workout_combinations(days_available, workouts_needed):
    """
    Generates all possible unique combinations of workout days based on the available days
    and the number of workouts needed with a rest day between each workout.
    
    Parameters:
    days_available (list): A list of days available for working out.
    workouts_needed (int): The number of workout days desired.

    Returns:
    list: A list containing all unique combinations of workout days.
    """
    
    valid_combinations = []
    day_indices = list(range(len(days_available)))  # Convert days to numerical indices
    for combination in itertools.combinations(day_indices, workouts_needed):
        # Check for rest days between workouts
        if all(combination[i+1] - combination[i] > 1 for i in range(len(combination) - 1)):
            valid_combinations.append([days_available[j] for j in combination])  # Map back to days
    
    if not valid_combinations:
        valid_combinations = [list(days_available[j] for j in combination) for combination in itertools.combinations(day_indices, workouts_needed)]

    return random.choice(valid_combinations) if valid_combinations else None


def generate_workout_plan(time_availability, preferred_exercises, target_volume, days_to_workout):
    """
    Main function for generating a workout plan.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    POPULATION_SIZE = 1000
    decay_rate = 0.5
    alpha = 0.2
    max_sets_per_day = 5
    threshold_shuffle = 5
    shuffle_keep = 20
    elite_size = 10
    indpb = 0.1

    file_path_folder = "Dynamic scheduler/"
    file_path_schedule = "DayAvailability.xlsx"
    file_workout_history =  "WorkoutHistory.xlsx"
    file_workout_exercise_info = "ExerciseInfo.xlsx"

    # ExerciseInfo sheet. Skal laves til SQL based
    df_exer = pd.read_excel(file_workout_exercise_info, sheet_name=0)

    days_history, exercises_history, sets_by_day_history = load_workout_history(file_workout_history)

    time_available = time_availability
    day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    exercises = preferred_exercises
    sets_by_day = load_workout_schedule(file_path_schedule)
    muscles = list(df_exer.columns[2:])

    workout_days = [key for key, value in time_available.items() if value > 0]
    workout_days = generate_workout_combinations(workout_days, days_to_workout)
    current_schedule = create_individual_from_schedule(sets_by_day)

    NUM_EXERCISES = len(exercises)
    NUM_DAYS = len(day_names)

    muscles = list(df_exer.columns[2:])
    time_per_exercise = {}
    fatigue = {}

    for i in exercises:
        row = df_exer.loc[df_exer['exercise'] == i]
        time_per_exercise[i] = row['time_per_set'].item()
        contrib_dict = {}
        for m_col in muscles:
            val = row[m_col].item()
            if val > 0:
                contrib_dict[m_col.capitalize()] = val
        fatigue[i] = contrib_dict

    def evaluate(individual):
        """
        Modularized evaluation function using helper functions.
        """
    
        # Step 1: Reshape chromosome into day-by-day structure
        sets_by_day = reshape_chromosome(individual,NUM_DAYS,NUM_EXERCISES)
        
        # Step 2: Calculate fatigue levels
        daily_fatigue = calculate_daily_fatigue(sets_by_day, sets_by_day_history,NUM_EXERCISES,NUM_DAYS,muscles,decay_rate,exercises,fatigue)

        # Step 3: Calculate effective and achieved volumes
        achieved_vol, effective_vol = calculate_effective_volume(sets_by_day, daily_fatigue,NUM_DAYS,muscles,exercises,fatigue,alpha)
        total_effective_volume = sum(effective_vol.values())

        # Step 4: Apply time penalty
        penalty, total_time_used = calculate_time_penalty(sets_by_day,day_names,exercises,NUM_EXERCISES,time_per_exercise,time_available)

        # Step 5: Calculate deviation from the target volumes
        total_deviation = calculate_deviation(achieved_vol, muscles)

        # Step 6: Fitness = deviation + penalty
        fitness_val = total_deviation**3 + penalty - total_effective_volume**1.5

        # Return only the fitness value for DEAP
        return (fitness_val,)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_sets", random.randint, 2, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_sets, n=NUM_DAYS * NUM_EXERCISES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=max_sets_per_day, indpb=0.1)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def main(current_schedule):
        # random.seed(42)
        # Create initial population
        pop = initialize_population_with_schedule(current_schedule, POPULATION_SIZE,indpb, toolbox, day_names, workout_days, max_sets_per_day, NUM_EXERCISES)

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

            shuffles = generate_combinations(pop,no_improvement_count,threshold_shuffle, gen, NUM_DAYS,NUM_EXERCISES,toolbox,shuffle_keep)

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

    best_ind = main(current_schedule)
    return reshape_chromosome(best_ind, NUM_DAYS, NUM_EXERCISES)

if __name__ == "__main__":
    user_schedule = {"Monday": 60, "Tuesday": 30, "Wednesday":0, "Thursday": 45,"Friday":0,"Saturday":0,"Sunday":0}
    preferred_exercises = ['Chest_press', 'Pulldown', 'Curls', 'Tricep_ext', 'Leg_curls', 'Leg_extensions', 'Calf_raises', 'Lateral_raises', 'chest_supported_row', 'Chest_fly']
    target_volume = {'Chest': 8, 'Back': 8, 'Quads': 4, 'Hamstrings': 4, 'Bicep': 4, 'Tricep': 4, 'Lateral_delts': 4, 'Front_delts': 0, 'Glutes': 0, 'Calf': 4}
    days_to_workout = 3

    result = generate_workout_plan(user_schedule, preferred_exercises, target_volume, days_to_workout)
    print("Optimized Schedule:", result)
