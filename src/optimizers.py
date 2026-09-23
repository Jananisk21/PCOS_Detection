"""
Three metaheuristic optimizers for the classifier-head hyperparameter
search, all operating on the shared encoding defined in
src/hyperparam_space.py so their performance can be fairly compared:

    - Genetic Algorithm (GA)             -- the base paper's proposed method
    - Particle Swarm Optimization (PSO)  -- named as future work in the paper
    - Grey Wolf Optimizer (GWO)          -- named as future work in the paper

Each `*_search` function has the same signature and returns the same
tuple: (best_individual: dict, best_fitness: float, history: list of
(iteration, best_fitness_so_far)) -- so run_ablation_study.py can drive
all three identically and plot their convergence on one chart.
"""

import numpy as np

from src.hyperparam_space import DIM, LOWER, UPPER, clip_vector, decode, random_vector


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
def ga_search(fitness_fn, population_size=8, generations=4, elitism=2,
              seed=42, verbose=True, log_callback=None):
    rng = np.random.default_rng(seed)
    population = [random_vector(rng) for _ in range(population_size)]
    history = []
    best_vec, best_fitness = None, -np.inf

    for gen in range(generations):
        fitnesses = [fitness_fn(decode(ind)) for ind in population]

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_vec = population[gen_best_idx].copy()

        history.append((gen, best_fitness))
        if verbose:
            print(f"[GA]  iter {gen + 1}/{generations} | best={best_fitness:.4f} | "
                  f"{decode(best_vec)}")
        if log_callback:
            log_callback(gen, decode(best_vec), best_fitness)

        elite_idx = np.argsort(fitnesses)[::-1][:elitism]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < population_size:
            idx_a = rng.choice(population_size, size=3, replace=False)
            parent_a = population[idx_a[np.argmax([fitnesses[i] for i in idx_a])]]
            idx_b = rng.choice(population_size, size=3, replace=False)
            parent_b = population[idx_b[np.argmax([fitnesses[i] for i in idx_b])]]

            mask = rng.random(DIM) < 0.5
            child = np.where(mask, parent_a, parent_b)

            mutate_mask = rng.random(DIM) < 0.25
            noise = rng.normal(0, 0.15, DIM) * (UPPER - LOWER)
            child = np.where(mutate_mask, child + noise, child)
            child = clip_vector(child)
            new_population.append(child)

        population = new_population

    return decode(best_vec), best_fitness, history


# ---------------------------------------------------------------------------
# Particle Swarm Optimization
# ---------------------------------------------------------------------------
def pso_search(fitness_fn, population_size=8, generations=4, seed=42,
               inertia=0.6, cognitive=1.5, social=1.5,
               verbose=True, log_callback=None):
    rng = np.random.default_rng(seed)
    positions = np.array([random_vector(rng) for _ in range(population_size)])
    velocities = np.zeros_like(positions)

    personal_best_pos = positions.copy()
    personal_best_fit = np.array([fitness_fn(decode(p)) for p in positions])

    global_best_idx = int(np.argmax(personal_best_fit))
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_fit = personal_best_fit[global_best_idx]

    history = [(0, global_best_fit)]
    if verbose:
        print(f"[PSO] iter 1/{generations} | best={global_best_fit:.4f} | "
              f"{decode(global_best_pos)}")
    if log_callback:
        log_callback(0, decode(global_best_pos), global_best_fit)

    for it in range(1, generations):
        r1 = rng.random((population_size, DIM))
        r2 = rng.random((population_size, DIM))
        velocities = (inertia * velocities
                      + cognitive * r1 * (personal_best_pos - positions)
                      + social * r2 * (global_best_pos - positions))
        positions = clip_vector(positions + velocities)

        fitnesses = np.array([fitness_fn(decode(p)) for p in positions])

        improved = fitnesses > personal_best_fit
        personal_best_pos[improved] = positions[improved]
        personal_best_fit[improved] = fitnesses[improved]

        it_best_idx = int(np.argmax(personal_best_fit))
        if personal_best_fit[it_best_idx] > global_best_fit:
            global_best_fit = personal_best_fit[it_best_idx]
            global_best_pos = personal_best_pos[it_best_idx].copy()

        history.append((it, global_best_fit))
        if verbose:
            print(f"[PSO] iter {it + 1}/{generations} | best={global_best_fit:.4f} | "
                  f"{decode(global_best_pos)}")
        if log_callback:
            log_callback(it, decode(global_best_pos), global_best_fit)

    return decode(global_best_pos), global_best_fit, history


# ---------------------------------------------------------------------------
# Grey Wolf Optimizer
# ---------------------------------------------------------------------------
def gwo_search(fitness_fn, population_size=8, generations=4, seed=42,
               verbose=True, log_callback=None):
    rng = np.random.default_rng(seed)
    positions = np.array([random_vector(rng) for _ in range(population_size)])

    def rank_wolves(positions):
        fits = np.array([fitness_fn(decode(p)) for p in positions])
        order = np.argsort(fits)[::-1]
        return positions[order[0]].copy(), fits[order[0]], \
               positions[order[1]].copy(), fits[order[1]], \
               positions[order[2]].copy(), fits[order[2]], fits

    alpha_pos, alpha_fit, beta_pos, beta_fit, delta_pos, delta_fit, fits = rank_wolves(positions)
    best_pos_ever, best_fit_ever = alpha_pos.copy(), alpha_fit
    history = [(0, best_fit_ever)]
    if verbose:
        print(f"[GWO] iter 1/{generations} | best={best_fit_ever:.4f} | {decode(best_pos_ever)}")
    if log_callback:
        log_callback(0, decode(best_pos_ever), best_fit_ever)

    for it in range(1, generations):
        a = 2.0 - 2.0 * it / max(generations - 1, 1)  # linearly decreases 2 -> 0

        new_positions = np.zeros_like(positions)
        for i in range(population_size):
            leaders = [alpha_pos, beta_pos, delta_pos]
            updates = []
            for leader in leaders:
                r1, r2 = rng.random(DIM), rng.random(DIM)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - positions[i])
                updates.append(leader - A * D)
            new_positions[i] = clip_vector(np.mean(updates, axis=0))

        positions = new_positions
        alpha_pos, alpha_fit, beta_pos, beta_fit, delta_pos, delta_fit, fits = rank_wolves(positions)

        if alpha_fit > best_fit_ever:
            best_fit_ever = alpha_fit
            best_pos_ever = alpha_pos.copy()

        history.append((it, best_fit_ever))
        if verbose:
            print(f"[GWO] iter {it + 1}/{generations} | best={best_fit_ever:.4f} | {decode(best_pos_ever)}")
        if log_callback:
            log_callback(it, decode(best_pos_ever), best_fit_ever)

    return decode(best_pos_ever), best_fit_ever, history


OPTIMIZERS = {
    "ga": ga_search,
    "pso": pso_search,
    "gwo": gwo_search,
}


def run_search(name: str, fitness_fn, **kwargs):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(OPTIMIZERS)}.")
    return OPTIMIZERS[name](fitness_fn, **kwargs)
