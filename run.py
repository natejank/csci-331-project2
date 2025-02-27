import argparse
import copy
import pprint
import signal
import time

import numpy as np
from search import Problem
import search

# replace this with an import of your code
from teams import TeamState, TeamAction


# modify this to use your state class
class GroupProblem(Problem):
    def actions(self, state: TeamState):
        return state.get_moves()

    def result(self, state: TeamState, action: TeamAction):
        return state.apply(action)

    def goal_test(self, state: TeamState):
        return state.is_goal()

    def value(self, state: TeamState):
        return state.n - state.conflict_weight


def validate_moves(teams: list[list[int]], n):
    """Verify that a computed solution doesn't produce conflicts

    :param teams: resulting teams
    :param n: number of people in class
    """

    assignments = np.zeros((n, n), dtype=np.int16)  # adjacency matrix

    for project in teams:
        for team in project:
            for i, p1 in enumerate(team):
                for p2 in team[i + 1 :]:
                    a = p1
                    b = p2
                    if p2 > p1:
                        a, b = b, a
                    assignments[a, b] += 1
                    if assignments[a, b] != 1:
                        print(f"Conflict between {p1} and {p2}!")
                        pprint.pp(teams, compact=False)
                        assert False


def heuristic_dist(node):
    """Function to provide a weight to states for astar."""
    state = node.state
    # Replace this with a state which works for your state class
    return state.conflict_weight


def hill_climbing_wrapper(problem):
    """Wrapper to hill climbing to run until it succeeds"""
    while True:
        state = search.hill_climbing(problem)
        if state.is_goal():
            return state


def simulated_annealing_wrapper(problem):
    """Wrapper to simulated annealing to increase the limit until it finds a solution.
    Some problems require a higher limit, and the easiest approach to work for
    all is to just make the limit progressively higher."""
    limit = 1000

    while True:
        state = search.simulated_annealing(
            problem, schedule=search.exp_schedule(lam=0.05, limit=limit)
        )
        if state and state.is_goal():
            return state
        limit = limit + 1000


def run(n, k, p, algorithm):
    """Run a search given an algorithm

    :param n: number of people
    :param k: number of people in a group
    :param p: number of projects to assign for
    :param algorithm: name of algorithm to use
    :return: tuple (3d array of groupings, elapsed time, space complexity, time complexity)
    """

    # returns a node
    search_dict = {
        "depth_first_tree_search": (True, search.depth_first_tree_search),
        "depth_first_graph_search": (True, search.depth_first_graph_search),
        "breadth_first_graph_search": (True, search.breadth_first_graph_search),
        "uniform_cost_search": (True, search.uniform_cost_search),
        "best_first_graph_search": (
            True,
            lambda p: search.best_first_graph_search(p, heuristic_dist),
        ),
        "astar_search": (True, lambda p: search.astar_search(p, h=heuristic_dist)),
        "hill_climbing": (False, hill_climbing_wrapper),
        "simulated_annealing": (False, simulated_annealing_wrapper),
        # "genetic_algorithm": genetic_algorithm,  # not doing genetic algorithm
    }
    state = TeamState(n, k, p)
    # print(initial)

    # run
    projects = []
    total_space_complexity = 0
    total_time_complexity = 0
    returns_node, algorithm = search_dict[algorithm]
    start_time = time.time()
    for _ in range(p):
        goal = algorithm(GroupProblem(state))
        goal, space_complexity, time_complexity = algorithm(GroupProblem(state))
        if returns_node:
            state = goal.state
        else:
            state = goal
        projects.append(copy.deepcopy(state.teams))
        state.reset()
        total_space_complexity += space_complexity
        total_time_complexity += time_complexity
    end_time = time.time()
    # print(goal.state)
    validate_moves(projects, n)
    return (
        projects,
        end_time - start_time,
        total_space_complexity,
        total_time_complexity,
    )


def benchmark_timeout_handler(signum, frame):
    """Function to be used by signal.signal to throw an exception after a set time"""
    raise Exception("timeout!!")


def benchmark(
    output_file,
    splits=(
        (6, 2, 1),
        (9, 2, 3),
        (20, 4, 3),
        (40, 8, 1),
        (40, 4, 5),
        (80, 4, 5),
    ),
    iterations=20,
    algorithms=(
        "depth_first_tree_search",
        "depth_first_graph_search",
        "breadth_first_graph_search",
        "uniform_cost_search",
        "best_first_graph_search",
        "astar_search",
        "hill_climbing",
        "simulated_annealing",
    ),
    timeout=120,
):
    """Function to benchmark algorithm performance"""
    header = "algorithm,n,k,p,run_number,out_of,elapsed_time,dnf,space_complexity,time_complexity"

    print(f"writing data to {output_file}")
    signal.signal(signal.SIGALRM, benchmark_timeout_handler)
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")

        print("starting benchmark")
        for algorithm in algorithms:
            print(f"benchmarking {algorithm}")
            for n, k, p in splits:
                for i in range(iterations):
                    # set a timer to throw an exception in {timeout} seconds
                    signal.alarm(timeout)
                    try:
                        _, elapsed, space_cpx, time_cpx = run(n, k, p, algorithm)
                        signal.alarm(0)  # cancel the alarm

                        message = f"{algorithm},{n},{k},{p},{i},{iterations},{elapsed},{False},{space_cpx},{time_cpx}"
                        fh.write(message + "\n")
                        print(message)

                    except Exception as exc:
                        if exc.args[0] != "timeout!!":
                            raise exc
                        print("timeout on execution!")
                        message = f"{algorithm},{n},{k},{p},{i},{iterations},{timeout},{True},{-1},{-1}"
                        fh.write(message + "\n")

                    fh.flush()
                    # clear the timer
                    signal.alarm(0)


def is_feasable(n, k, p):
    return p <= (n - 1) / (k - 1)


def main():
    parser = argparse.ArgumentParser(
        prog="tree_search", description="runs various tree search algorithms"
    )

    parser.add_argument(
        "-n", "--people", default=10, help="The number of people", type=int
    )
    parser.add_argument("-k", "--teams", default=3, help="The team size", type=int)
    parser.add_argument(
        "-p", "--projects", default=3, help="The number projects", type=int
    )
    parser.add_argument(
        "-s",
        "--search",
        default="depth_first_tree_search",
        help="The search algorithm to use",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Switch to enable benchmarking",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--benchmarking_output",
        help="Benchmarking output file",
        default=f"benchmarking_{int(time.time())}.csv",
    )

    args = parser.parse_args()

    if not is_feasable(args.people, args.teams, args.projects):
        print("not feasable")
        return

    if args.benchmark:
        benchmark(
            args.benchmarking_output,
            splits=((4, 2, 2), (8, 3, 2), (20, 4, 4), (40, 4, 5)),
            algorithms=(
                "breadth_first_graph_search",
                "astar_search",
                "hill_climbing",
                "simulated_annealing",
            ),
            iterations=40,
        )
    else:
        projects, elapsed, _, _ = run(
            args.people, args.teams, args.projects, args.search
        )
        # display teams
        for project_num, teams in enumerate(projects):
            print()
            print(f"Project {project_num + 1}")
            for team_num, team in enumerate(teams):
                print(
                    f'Team P{project_num + 1}_{team_num + 1}: {", ".join(map(str, team))}'
                )
        print(f"elapsed time: {elapsed} seconds")


if __name__ == "__main__":
    main()
