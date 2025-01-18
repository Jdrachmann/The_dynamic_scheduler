"""
MODEL_GA.py - The genetic algorithm workout planner
Added detailed logging for debugging purposes
"""
import random
import math
from deap import base, creator, tools
from functools import partial
import pandas as pd
import os
import random
import itertools
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_workout_history(file_path):
    """
    Load workout history from an Excel file and convert it into a usable format for fatigue calculation.
    - Returns sets by day and exercises for fatigue calculation.
    """
    logger.info(f"Loading workout history from {file_path}")
    try:
        logger.info(file_path)
        df = pd.read_excel(file_path, header=None)
        logger.debug(f"Workout history loaded successfully. Shape: {df.shape}")

        exercises = df.iloc[0].dropna().tolist()

        days = df.iloc[:, 0].dropna().tolist()
        sets_by_day = df.iloc[:, 1:].values.tolist()

        sets_by_day = [[int(sets) if pd.notna(sets) else 0 for sets in row]
                       for row in sets_by_day[1:]]
        logger.debug(
            f"Extracted {len(days)} days, {len(exercises)} exercises, and sets data."
        )
        return days, exercises, sets_by_day
    except FileNotFoundError:
        logger.error(f"Error: Workout history file not found at {file_path}")
        raise
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while loading workout history: {e}")
        raise


def load_workout_schedule(file_path):
    """
    Load the workout schedule from an Excel file.
    """
    logger.info(f"Loading workout schedule from {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        sets_by_day = df.iloc[7:, 1:].fillna(0).infer_objects(
            copy=False).values.tolist()
        logger.debug(
            f"Workout schedule loaded successfully. Shape: {df.shape}")
        return sets_by_day
    except FileNotFoundError:
        logger.error(f"Error: Workout schedule file not found at {file_path}")
        raise
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while loading workout schedule: {e}"
        )
        raise


def create_individual_from_schedule(current_schedule):
    """
    Convert the current schedule into an individual.
    """

    return [sets for day in current_schedule for sets in day]


def generate_combinations(pop, stagnant_generations, threshold, gen, NUM_DAYS,
                          NUM_EXERCISES, toolbox, shuffle_keep):
    """
    Generates all possible unique combinations for a list of lists based on shuffling the lists themselves.

    Parameters:
    activities (list of lists): A list of lists where each inner list represents a set of activities.

    Returns:
    list: A list containing all unique combinations of the lists themselves.
    """
    logger.debug(
        f"Generating combinations. Stagnant generations: {stagnant_generations}, Threshold: {threshold}, Generation: {gen}"
    )
    # Trigger shuffle only if the threshold is reached

    if stagnant_generations == threshold or gen == 1:
        current_best = tools.selBest(pop, 1)[0]
        current_best = reshape_chromosome(current_best, NUM_DAYS,
                                          NUM_EXERCISES)

        shuffled_schedule = list(itertools.permutations(current_best))
        shuffled_individual = create_individual_from_schedule(
            shuffled_schedule)

        # Flatten each sequence into a single list
        flattened = []

        # Loop through each sequence in the shuffled individual
        for index in range(int(len(shuffled_individual) / 7)):
            sliced_list = shuffled_individual[index * 7:(index + 1) * 7]
            sliced_list = create_individual_from_schedule(sliced_list)

            flattened.append(sliced_list)
        # Example to convert a raw list to a DEAP individual
        pop_flattened = [creator.Individual(ind) for ind in flattened]

        # Calculate fitness after shuffling
        fitnesses = list(map(toolbox.evaluate, flattened))
        for ind, fit in zip(pop_flattened, fitnesses):
            ind.fitness.values = fit

        new_best_individuals = tools.selBest(pop_flattened, shuffle_keep)

        logger.debug(
            f"Generated {len(new_best_individuals)} new individuals after shuffling."
        )
        return new_best_individuals
    else:
        return []


def reshape_chromosome(individual, NUM_DAYS, NUM_EXERCISES):
    """
    Reshape a flat chromosome into a structured schedule.
    """

    sets_by_day = []
    idx = 0
    for _ in range(NUM_DAYS):
        day_slice = individual[idx:idx + NUM_EXERCISES]
        sets_by_day.append(day_slice)
        idx += NUM_EXERCISES
    return sets_by_day


def reshape_sets_by_day_to_usable_format(sets_by_day, exercises, day_names):
    """
    Reshape the sets_by_day list into a schedule where the exercises can be easily accessed.
    """
    # Create a DataFrame

    df = pd.DataFrame(sets_by_day).T
    df.columns = day_names
    df.insert(0, "Exercises", exercises)

    return df


def diminishing_effective_sets(n_sets):
    """
    Function that calculates the diminishing returns of performing a lot of sets in the same day
    """

    if n_sets <= 0:
        return 0.0

    r = 2 / 3
    return 1.0 * (1 - r**n_sets) / (1 - r)


def calculate_effective_volume(sets_by_day, daily_fatigue, NUM_DAYS, muscles,
                               exercises, fatigue, alpha):
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


def calculate_deviation(achieved_vol, muscles, target_volume):
    """Calculate the deviation from the target volume for each muscle."""

    total_deviation = 0.0
    for m in muscles:
        diff = target_volume[m] - achieved_vol[m]
        if diff > 0:
            total_deviation += abs(diff)
    return total_deviation


def calculate_time_penalty(sets_by_day, day_names, exercises, NUM_EXERCISES,
                           time_per_exercise, time_available):
    """Calculate the time penalty for exceeding daily availability.
    Setup time is added only when an exercise is performed.
    """
    # logger.debug("Calculating time penalty.")
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
                time_used_today += time_per_exercise[
                    exercises[ei]]  # Setup time added only if sets > 0

        # Apply penalty for exceeding available time
        if time_used_today > time_available[day_name]:
            penalty += 100.0 * (time_used_today - time_available[day_name])
        elif time_available[day_name] - time_used_today < 5:
            penalty += 100.0 * (time_available[day_name] - time_used_today)

        total_time_used += time_used_today

    return penalty, total_time_used


def calculate_time_used_per_day(sets_by_day, day_names, exercises,
                                NUM_EXERCISES, time_per_exercise):
    """
    Calculate time used per day including setup time only when an exercise is performed.
    Setup time is equal to the time per exercise.
    """
    # logger.debug("Calculating time used per day.")
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


def mutUniformSets(individual,
                   workout_days,
                   day_names,
                   NUM_EXERCISES,
                   max_sets_per_day,
                   indpb=0.1):
    """Mutate only workout days by randomly adjusting sets between 2-5."""
    # logger.debug("Mutating individual.")
    for day_index, day_name in enumerate(day_names):
        if day_name in workout_days:  # Mutate only workout days
            for exercise_index in range(NUM_EXERCISES):
                if random.random() < indpb:
                    individual[day_index * NUM_EXERCISES +
                               exercise_index] = random.randint(
                        0, max_sets_per_day)
        else:
            for exercise_index in range(NUM_EXERCISES):
                individual[day_index * NUM_EXERCISES + exercise_index] = 0
    return (individual, )



def initialize_population_with_schedule(current_schedule, pop_size, indpb,
                                        toolbox, day_names, workout_days,
                                        max_sets_per_day, NUM_EXERCISES):
    """
    Initialize a population ensuring a good distribution of sets on workout days.
    """
    logger.debug("Initializing population.")

    # Convert the provided schedule to a DEAP individual object
    population = [creator.Individual(current_schedule)]

    # Generate the remaining individuals
    for _ in range(pop_size - 1):
        individual = creator.Individual([0] * (len(day_names) * NUM_EXERCISES))

        # For each workout day, ensure some exercises have non-zero sets
        for day_index, day_name in enumerate(day_names):
            if day_name in workout_days:
                # Select 3-5 exercises randomly for this day
                num_exercises = random.randint(3, 5)
                selected_exercises = random.sample(range(NUM_EXERCISES),
                                                   num_exercises)

                for exercise_index in selected_exercises:
                    # Assign 2-4 sets for each selected exercise
                    individual[day_index * NUM_EXERCISES +
                               exercise_index] = random.randint(2, 4)

        population.append(individual)

    return population


def calculate_initial_fatigue(sets_by_day_history, NUM_DAYS, muscles, decay_rate, exercises_history, fatigue):
    """
    Calculate the initial fatigue levels for the first 7 days based on history.

    Args:
        sets_by_day_history (list): Historical sets performed.
        NUM_DAYS (int): Number of days to calculate fatigue for.
        muscles (list): List of muscles.
        decay_rate (float): Fatigue decay rate per day.
        exercises (list): List of exercises.
        fatigue (dict): Fatigue contribution by exercise for each muscle.

    Returns:
        list: Initial fatigue levels for each muscle over the 7 days.
    """
    schedule_empty = creator.Individual([0] * (NUM_DAYS * len(exercises_history)))
    sets_by_day_empty = reshape_chromosome(schedule_empty, NUM_DAYS, len(exercises_history))
    combined_sets_his_fut = [*sets_by_day_history, *sets_by_day_empty]

    daily_fatigue = [[float(0) for _ in muscles] for _ in range(NUM_DAYS * 2)]

    for d in range(1, NUM_DAYS * 2):  # Day 0 already initialized
        for mi, m in enumerate(muscles):
            # Fatigue decays from the previous day
            daily_fatigue[d][mi] = decay_rate * daily_fatigue[d - 1][mi]

            # Add fatigue contribution from previous day's sets
            for ei, e in enumerate(exercises_history):
                day_minus_1_sets = combined_sets_his_fut[d - 1][ei]
                daily_fatigue[d][mi] += day_minus_1_sets * fatigue[e].get(m, 0)

    daily_initial_fatigue = daily_fatigue[7:14]  # WOrk indexes skal automatiseres

    return daily_initial_fatigue


def calculate_daily_fatigue(sets_by_day, daily_initial_fatigue, NUM_EXERCISES,
                            NUM_DAYS, muscles, decay_rate, exercises, fatigue):
    """Calculate the fatigue levels for each muscle on each day."""
    # logger.debug("Calculating daily fatigue.")

    daily_fatigue = [[float(0) for _ in muscles] for _ in range(NUM_DAYS * 2)]
    for d in range(1, NUM_DAYS):  # Day 0 already initialized
        for mi, m in enumerate(muscles):
            # Fatigue decays from the previous day
            daily_fatigue[d][mi] = decay_rate * daily_fatigue[d - 1][mi]

            # Add fatigue contribution from previous day's sets
            for ei, e in enumerate(exercises):
                day_minus_1_sets = sets_by_day[d - 1][ei]
                daily_fatigue[d][mi] += day_minus_1_sets * fatigue[e].get(m, 0)

    daily_fatigue_total = daily_initial_fatigue + daily_fatigue

    return daily_fatigue_total


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
    logger.debug(
        f"Generating workout combinations. Days available: {days_available}, Workouts needed: {workouts_needed}"
    )
    valid_combinations = []
    day_indices = list(range(
        len(days_available)))  # Convert days to numerical indices
    for combination in itertools.combinations(day_indices, workouts_needed):
        # Check for rest days between workouts
        if all(combination[i + 1] - combination[i] > 1
               for i in range(len(combination) - 1)):
            valid_combinations.append([days_available[j] for j in combination
                                       ])  # Map Back to days

    if not valid_combinations:
        valid_combinations = [
            list(days_available[j] for j in combination) for combination in
            itertools.combinations(day_indices, workouts_needed)
        ]

    logger.debug(
        f"Generated {len(valid_combinations)} valid workout combinations.")
    return random.choice(valid_combinations) if valid_combinations else None


def generate_workout_plan(time_availability, preferred_exercises,
                          target_volume, days_to_workout):
    """
    Main function for generating a workout plan.
    """
    POPULATION_SIZE = 10000
    decay_rate = 0.5
    alpha = 0.2
    max_sets_per_day = 5
    min_sets_per_day = 2  # Added minimum sets constraint
    threshold_shuffle = 5
    shuffle_keep = 20
    elite_size = 10
    indpb = 0.2  # Increased mutation probability for better exploration

    file_path_folder = "Dynamic scheduler/"
    file_path_schedule = "DayAvailability.xlsx"
    file_workout_history = "WorkoutHistory.xlsx"

    day_names = [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
        'Sunday'
    ]
    exercises = preferred_exercises

    NUM_EXERCISES = len(exercises)
    NUM_DAYS = len(day_names)

    time_available = time_availability
    available_days = sum(1 for duration in time_availability.values() if duration > 0)
    days_to_workout = available_days  # Needs to be set by the user
    workout_days = [
        key for key, value in time_available.items() if value > 0
    ]
    workout_days = generate_workout_combinations(workout_days,
                                                 days_to_workout)

    logger.info("Starting generate_workout_plan with inputs:")
    logger.info(f"Time availability: {time_availability}")
    logger.info(f"Preferred exercises: {preferred_exercises}")
    logger.info(f"Target volume: {target_volume}")
    logger.info(f"Days to workout: {days_to_workout}")

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_sets", random.randint, 0, max_sets_per_day)
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_sets,
                     n=NUM_DAYS * NUM_EXERCISES)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    toolbox.register("mutate",
                     mutUniformSets,
                     workout_days=workout_days,
                     day_names=day_names,
                     NUM_EXERCISES=NUM_EXERCISES,
                     max_sets_per_day=max_sets_per_day)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # sets_by_day = load_workout_schedule(file_path_schedule)
    current_schedule = creator.Individual(
        [0] * (len(day_names) * NUM_EXERCISES))  # Supposed to by user preffered schedule as input
    sets_by_day = reshape_chromosome(current_schedule, NUM_DAYS, NUM_EXERCISES)

    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load workout history
        days_history, exercises_history, sets_by_day_history = load_workout_history(
            file_workout_history)
        # Process exercises and build fatigue dictionary
        df_exer = pd.read_excel("ExerciseInfo.xlsx", sheet_name=0)
        muscles = list(df_exer.columns[2:])
        logger.info(f"Detected muscle groups: {muscles}")

        time_per_exercise = {}
        fatigue = {}

        for exercise in preferred_exercises:
            logger.info(f"\nProcessing exercise: {exercise}")

            # Find exercise in database
            row = df_exer.loc[df_exer['exercise'] == exercise]
            if len(row) == 0:
                logger.error(f"Exercise '{exercise}' not found in database!")
                raise ValueError(
                    f"Exercise '{exercise}' not found in database")

            logger.info(f"Found exercise data: {row.to_dict('records')[0]}")

            # Get time per set
            time_per_exercise[exercise] = row['time_per_set'].iloc[0]
            logger.info(f"Time per set: {time_per_exercise[exercise]}")

            # Build muscle contributions dictionary
            contrib_dict = {}
            for muscle in muscles:
                val = row[muscle].iloc[0]
                if val > 0:
                    contrib_dict[muscle.capitalize()] = val

            fatigue[exercise] = contrib_dict
            logger.info(f"Fatigue contributions: {contrib_dict}")

        logger.info("\nExercise processing complete")
        logger.info(f"Time per exercise: {time_per_exercise}")
        logger.info(f"Fatigue dictionary: {fatigue}")

        # Calculate initial fatigue levels

        daily_initial_fatigue = calculate_initial_fatigue(
            sets_by_day_history, NUM_DAYS, muscles, decay_rate, exercises_history, fatigue)

        def evaluate(individual):
            """
            Modularized evaluation function using helper functions.
            """

            # Step 1: Reshape chromosome into day-by-day structure
            sets_by_day = reshape_chromosome(individual, NUM_DAYS,
                                             NUM_EXERCISES)

            # Step 2: Calculate fatigue levels
            daily_fatigue = calculate_daily_fatigue(sets_by_day,
                                                    daily_initial_fatigue,
                                                    NUM_EXERCISES, NUM_DAYS,
                                                    muscles, decay_rate,
                                                    exercises, fatigue)

            # Step 3: Calculate effective and achieved volumes
            achieved_vol, effective_vol = calculate_effective_volume(
                sets_by_day, daily_fatigue, NUM_DAYS, muscles, exercises,
                fatigue, alpha)
            total_effective_volume = sum(effective_vol.values())

            # Step 4: Apply time penalty
            penalty, total_time_used = calculate_time_penalty(
                sets_by_day, day_names, exercises, NUM_EXERCISES,
                time_per_exercise, time_available)

            # Step 5: Calculate deviation from target volumes
            total_deviation = calculate_deviation(
                achieved_vol, muscles, target_volume)

            # Step 6: Add penalties for workout day distribution
            distribution_penalty = 0
            for day_index, day_name in enumerate(day_names):
                day_sets = sets_by_day[day_index]
                if day_name in workout_days:
                    # Penalize workout days with too few exercises
                    active_exercises = sum(1 for sets in day_sets if sets > 0)
                    if active_exercises < 3:
                        distribution_penalty += 1000 * (3 - active_exercises)
                else:
                    # Penalize non-workout days with exercises
                    active_exercises = sum(1 for sets in day_sets if sets > 0)
                    if active_exercises > 0:
                        distribution_penalty += 1000 * active_exercises

            # Step 7: Add penalties for sets outside the allowed range
            sets_penalty = 0
            for day in sets_by_day:
                for sets in day:
                    if sets > 0 and sets < min_sets_per_day:
                        sets_penalty += 100 * (min_sets_per_day - sets)
                    elif sets > max_sets_per_day:
                        sets_penalty += 100 * (sets - max_sets_per_day)

            # Step 8: Calculate final fitness with adjusted weights
            fitness_val = (
                (total_deviation**2) +  # Volume deviation
                penalty * 0.5 +  # Time penalty (reduced weight)
                sets_penalty +  # Set range penalties
                distribution_penalty -  # Workout distribution penalty
                (total_effective_volume * 2)  # Increased reward for volume
            )

            return (fitness_val, )

        toolbox.register("evaluate", evaluate)

        def main(current_schedule):
            # Create initial population
            pop = initialize_population_with_schedule(
                current_schedule, POPULATION_SIZE, indpb, toolbox, day_names,
                workout_days, max_sets_per_day, NUM_EXERCISES)

            # Number of generations and probabilities
            NGEN = 100
            CXPB, MUTPB = 0.7, 0.3

            # Evaluate initial population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            # Evolution parameters
            best_so_far = float("inf")
            no_improvement_count = 0
            IMPROVEMENT_THRESHOLD = 1e-4
            MAX_NO_IMPROVE_GENS = 10

            for gen in range(NGEN):
                # Sort population by fitness and preserve elites
                pop = sorted(pop, key=lambda ind: ind.fitness.values[0])

                # Generate new combinations if needed
                shuffles = generate_combinations(pop, no_improvement_count,
                                                 threshold_shuffle, gen,
                                                 NUM_DAYS, NUM_EXERCISES,
                                                 toolbox, shuffle_keep)

                elites = pop[:elite_size]
                if shuffles:
                    elites += shuffles

                # Selection and reproduction
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))

                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < CXPB:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate new offspring
                invalid_inds = [
                    ind for ind in offspring if not ind.fitness.valid
                ]
                fitnesses = map(toolbox.evaluate, invalid_inds)
                for ind, fit in zip(invalid_inds, fitnesses):
                    ind.fitness.values = fit

                # Replace population
                pop[:] = offspring + elites

                # Track improvement
                current_best = min(ind.fitness.values[0] for ind in pop)
                if abs(best_so_far - current_best) < IMPROVEMENT_THRESHOLD:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                    best_so_far = current_best

                # Early stopping
                if no_improvement_count >= MAX_NO_IMPROVE_GENS:
                    logger.info(
                        f"No improvement for {MAX_NO_IMPROVE_GENS} generations; stopping early."
                    )
                    break

                logger.info(
                    f"Gen {gen}: best fitness = {current_best:.6f}, no_improvement_count = {no_improvement_count}"
                )

            # Get best solution
            best_ind = tools.selBest(pop, 1)[0]
            return best_ind

        best_ind = main(current_schedule)
        result_df = reshape_sets_by_day_to_usable_format(
            reshape_chromosome(best_ind, NUM_DAYS, NUM_EXERCISES), preferred_exercises, day_names)

        # Convert DataFrame to dictionary format
        result_dict = {
            "Exercises": result_df["Exercises"].tolist(),
            "Monday": result_df["Monday"].tolist(),
            "Tuesday": result_df["Tuesday"].tolist(),
            "Wednesday": result_df["Wednesday"].tolist(),
            "Thursday": result_df["Thursday"].tolist(),
            "Friday": result_df["Friday"].tolist(),
            "Saturday": result_df["Saturday"].tolist(),
            "Sunday": result_df["Sunday"].tolist()
        }
        return result_dict

    except Exception as e:
        logger.error(f"Error in generate_workout_plan: {str(e)}")
        logger.exception("Stack trace:")
        raise


if __name__ == "__main__":
    result = generate_workout_plan(time_available, selected_exercises,
                                   target_volume, 4)
    logger.info(f"Generated workout plan: {result}")