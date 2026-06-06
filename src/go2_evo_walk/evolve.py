from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from deap import algorithms, base, creator, tools

from .sim import QuadrupedEvaluator, SimulationConfig


def build_toolbox(evaluator: QuadrupedEvaluator, seed: int) -> base.Toolbox:
    random.seed(seed)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.gauss, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=evaluator.gene_count)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluator.evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.4)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.35, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def run_evolution(args: argparse.Namespace) -> dict[str, object]:
    config = SimulationConfig(
        urdf_path=args.urdf,
        gui=args.gui,
        episode_seconds=args.episode_seconds,
        settle_seconds=args.settle_seconds,
    )
    evaluator = QuadrupedEvaluator(config)
    try:
        toolbox = build_toolbox(evaluator, args.seed)
        population = toolbox.population(n=args.population)
        elitism_k = max(0, int(args.elitism))
        hall_of_fame = tools.HallOfFame(max(1, elitism_k))
        stats = tools.Statistics(lambda individual: individual.fitness.values[0])
        stats.register("avg", lambda values: sum(values) / len(values))
        stats.register("max", max)
        stats.register("min", min)

        # Arranco evaluando la población inicial
        invalid_ind = [ind for ind in population if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        if invalid_ind:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # Registro la evolución generación por generación con elitismo
        print("gen     avg     max     min    ")
        for gen in range(0, args.generations + 1):
            record = stats.compile(population)
            if gen == 0:
                print(f"{gen:<3}     {record['avg']:.4f} {record['max']:.4f} {record['min']:.4f}")
            else:
                print(f"{gen:<3}     {record['avg']:.4f} {record['max']:.4f} {record['min']:.4f}")

            if gen == args.generations:
                break

            # Selecciono y clono a los padres de la siguiente generación
            parents = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, parents))

            # Aplico cruce y mutación
            for i in range(1, len(offspring), 2):
                if random.random() < args.crossover:
                    toolbox.mate(offspring[i - 1], offspring[i])
                    try:
                        del offspring[i - 1].fitness.values
                        del offspring[i].fitness.values
                    except Exception:
                        pass

            for i in range(len(offspring)):
                if random.random() < args.mutation:
                    toolbox.mutate(offspring[i])
                    try:
                        del offspring[i].fitness.values
                    except Exception:
                        pass

            # Evalúo a los individuos que quedaron inválidos
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = list(map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # Conservo a los mejores de la población actual
            elites = tools.selBest(population, elitism_k) if elitism_k > 0 else []

            # Completo el resto con los mejores hijos
            needed = len(population) - len(elites)
            new_pop = elites + tools.selBest(offspring, needed)
            population[:] = new_pop

            hall_of_fame.update(population)
            # La visualización en vivo se hace con --gui durante cada evaluación.

        best = hall_of_fame[0]
        result = {
            "fitness": best.fitness.values[0],
            "genome": list(best),
            "gene_count": evaluator.gene_count,
            "urdf": args.urdf,
            "generations": args.generations,
            "population": args.population,
        }

        if args.output:
            output_path = Path(args.output).expanduser()
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["output"] = str(output_path)

        print(json.dumps(result, indent=2))
        return result
    finally:
        evaluator.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimiza la marcha de un cuadrúpedo simulado con algoritmos evolutivos")
    parser.add_argument("--urdf", type=str, default=None, help="Ruta al URDF del Unitree Go2")
    parser.add_argument("--gui", action="store_true", help="Abro la GUI de PyBullet")
    parser.add_argument("--population", type=int, default=24, help="Tamaño de la población")
    parser.add_argument("--generations", type=int, default=20, help="Número de generaciones")
    parser.add_argument("--episode-seconds", type=float, default=6.0, help="Tiempo de simulación por evaluación")
    parser.add_argument("--settle-seconds", type=float, default=1.0, help="Tiempo de asentamiento antes de medir")
    parser.add_argument("--crossover", type=float, default=0.5, help="Probabilidad de cruce")
    parser.add_argument("--mutation", type=float, default=0.35, help="Probabilidad de mutación")
    parser.add_argument("--elitism", type=int, default=2, help="Cantidad de individuos top que conservo")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--output", type=str, default="best_run.json", help="Ruta para guardar el mejor genoma")
    parser.add_argument("--vis-slowdown", type=float, default=1.0, help="Factor para hacer más lenta la visualización (>1 más lento)")
    parser.add_argument("--vis-hold", action="store_true", help="Si lo activo, dejo la GUI abierta y espero Enter después de cada individuo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_evolution(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
