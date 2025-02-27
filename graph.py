import argparse
import sys
from dataclasses import dataclass

from matplotlib import pyplot as plt
from numpy import require


@dataclass
class BenchmarkIteration:
    algorithm: str
    n: int
    k: int
    p: int
    run_number: int
    iterations: int
    elapsed_time: float
    dnf: bool
    space_complexity: int
    time_complexity: int

    def nkp(self):
        return (self.n, self.k, self.p)


class BenchmarkData:
    def __init__(self):
        self.space_complexity: list[int] = []
        self.time_complexity: list[int] = []
        self.elapsed_time: list[float] = []

    def add_data(self, bi: BenchmarkIteration):
        self.space_complexity.append(bi.space_complexity)
        self.time_complexity.append(bi.time_complexity)
        self.elapsed_time.append(bi.elapsed_time)


def parse_data(lines, graph_dnf=False):
    """Parse data from iterator of lines
    Assumes data is in the following format:
    algorithm, n, k, p, run_number, iterations, elapsed_time, did_not_finish?, space_complexity, time_complexity
    """

    raw_data = []

    # dict by values of (n, k, p), by algorithm
    by_value = {}

    for line in lines:
        line = line.strip().split(",")
        bi = BenchmarkIteration(
            line[0],
            int(line[1]),
            int(line[2]),
            int(line[3]),
            int(line[4]),
            int(line[5]),
            float(line[6]),
            line[7] == "True",
            int(line[8]),
            int(line[9]),
        )
        raw_data.append(bi)

        if by_value.get(bi.nkp()) is None:
            by_value[bi.nkp()] = {}
        if by_value[bi.nkp()].get(bi.algorithm) is None:
            by_value[bi.nkp()][bi.algorithm] = BenchmarkData()

        if (not bi.dnf) or graph_dnf:
            by_value[bi.nkp()][bi.algorithm].add_data(bi)

    return raw_data, by_value


def main():
    GRAPH_VALUES = ("time_complexity", "space_complexity", "time_elapsed")

    parser = argparse.ArgumentParser(
        prog="tree_search", description="runs various tree search algorithms"
    )
    parser.add_argument("-f", "--file", help="file to parse", type=str, required=True)
    parser.add_argument(
        "-n", "--people", help="The number of people", type=int, required=True
    )
    parser.add_argument("-k", "--teams", help="The team size", type=int, required=True)
    parser.add_argument(
        "-p", "--projects", help="The number projects", type=int, required=True
    )
    parser.add_argument(
        "-l", "--list", help="List possible n, k, p values", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--graph_dnf",
        help='Include 120s "DNF" values',
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--metric",
        help="Metric to graph",
        type=str,
        choices=GRAPH_VALUES,
        required=True,
    )

    args = parser.parse_args()

    data = None
    with open(args.file) as fp:
        fp.readline()
        _, data = parse_data(fp, args.graph_dnf)

    if args.list:
        print(data.keys())
        return

    key = (args.people, args.teams, args.projects)
    labels = []
    graph_data = []
    for label in data[key]:
        labels.append(label)
        if args.metric == "time_complexity":
            graph_data.append(data[key][label].time_complexity)
        if args.metric == "space_complexity":
            graph_data.append(data[key][label].space_complexity)
        if args.metric == "time_elapsed":
            graph_data.append(data[key][label].elapsed_time)

    plt.boxplot(graph_data, tick_labels=labels)
    if args.metric == "time_complexity":
        plt.title(
            f"Time Complexity (n={args.people}, k={args.teams}, p={args.projects})"
        )
        plt.ylabel("nodes explored")
    if args.metric == "space_complexity":
        plt.title(
            f"Space Complexity (n={args.people}, k={args.teams}, p={args.projects})"
        )
        plt.ylabel("data structure size")
    if args.metric == "time_elapsed":
        plt.title(f"Time Elapsed (n={args.people}, k={args.teams}, p={args.projects})")
        plt.ylabel("time (seconds)")

    if not args.graph_dnf:
        print("Note: not including entries which timed out")
        print("Datapoints by type:")
        for i, dps in enumerate(graph_data):
            print(f"{labels[i]}: {len(dps)}")
    plt.show()


if __name__ == "__main__":
    main()
