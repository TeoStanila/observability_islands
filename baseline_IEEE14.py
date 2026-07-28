import sys
import os
import networkx as nx
import pprint
import json
import pandapower as pp
from pathlib import Path
import numpy as np
import pandapower.plotting as pplot
from collections import deque
import matplotlib.pyplot as plt

from islands_IEEE14 import get_subnetwork
from generation_IEEE14 import observability_analysis

def prepare_merged_graph(net):
    def update_measurements(lines, u, v):
        new_list = []
        for (x, y) in lines:
            new_x = u if x == v else x
            new_y = u if y == v else y
            if new_x != new_y:
                new_list.append((new_x, new_y))

        return new_list

    lines = set()
    buses =set()
        
    for measurement in net.measurement.index:
            if (net.measurement.element_type[measurement] == "line"
                    and net.measurement.measurement_type[measurement] in ("p", "q")):
                line = net.measurement.element[measurement]
                from_bus = net.line.from_bus[line]
                to_bus = net.line.to_bus[line]
                lines.add((np.int64(from_bus), np.int64(to_bus)))

            if (net.measurement.element_type[measurement] == "bus"
                    and net.measurement.measurement_type[measurement] in ("p", "q")):
                bus = net.measurement.element[measurement]
                buses.add(np.int64(bus))

    graph = pplot.create_nxgraph(net)
    merges = {key: [] for key in graph.nodes}
    new_graph = graph.copy()

    while lines:
        (u, v) = lines.pop()
        (u, v) = (min(u, v), max(u, v))
        
        merges[u].extend([v])
        try:
            merges[u].extend(merges[v])
            merges.pop(v)
        except KeyError:
            pass

        new_graph = nx.contracted_nodes(new_graph, u, v, self_loops=False)
        lines = update_measurements(lines, u, v)

    return new_graph, buses, merges

def get_loop(forest, u, v):
    try:
        path = nx.shortest_path(forest, u, v)
        return [(min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:])]
    except nx.NetworkXNoPath:
        return None
    
def get_merge(merges, bus):
        for superbus, buses in merges.items():
            if len(buses) == 0:
                continue
            if bus in buses:
                return superbus
        return bus

def augmented_injections(reference, buses, merges): 
    def find_augmenting_sequence(forest, bus, assigned):
        queue = deque()

        for neighbor in reference.neighbors(bus):
            edge = (min(bus, neighbor), max(bus, neighbor))

            loop_edges = get_loop(forest, bus, neighbor)
            if loop_edges is not None:
                sim_forest = forest.copy()
                sim_forest.add_edge(*edge)
                queue.append((edge, [edge], sim_forest, {edge}))
            else:
                return [edge]

        while queue:
            free_edge, path, sim_forest, visited = queue.popleft()
            u, v = free_edge

            loop_edges = get_loop(sim_forest, u, v)

            if loop_edges is None:
                return path

            for loop_edge in loop_edges:
                loop_edge = (min(loop_edge), max(loop_edge))
                if loop_edge not in assigned:
                    continue

                assigned_bus = assigned.get(loop_edge)

                for neighbor in reference.neighbors(get_merge(merges, assigned_bus)):
                    candidate = (min(get_merge(merges, assigned_bus), neighbor), max(get_merge(merges, assigned_bus), neighbor))
                    if candidate == loop_edge:
                        continue
                    if sim_forest.has_edge(*candidate):
                        continue
                    if candidate in visited:
                        continue

                    new_visited = visited.copy()
                    new_visited.add(candidate)

                    new_sim = sim_forest.copy()
                    new_sim.remove_edge(*loop_edge)

                    u2, v2 = candidate
                    new_loop_edges = get_loop(new_sim, u2, v2)
                    if new_loop_edges is None:
                        return path + [loop_edge, candidate]
                    else:
                        new_sim.add_edge(*candidate)
                        new_path = path + [loop_edge, candidate]
                        queue.append((candidate, new_path, new_sim, new_visited))

        return None

    forest = nx.Graph()
    forest.add_nodes_from(reference.nodes)

    assigned = {}

    for bus in buses:
        neighbors = list(reference.neighbors(get_merge(merges, bus)))
        assigned_directly = False

        for node in neighbors:
            edge = (min(get_merge(merges, bus), node), max(get_merge(merges, bus), node))
            if get_loop(forest, get_merge(merges, bus), node) is None:
                    forest.add_edge(get_merge(merges, bus), node)
                    assigned[edge] = bus
                    assigned_directly = True
                    break

        if not assigned_directly:
            seq = find_augmenting_sequence(forest, get_merge(merges, bus), assigned)
            if seq is not None:
                for i, edge in enumerate(seq):
                    if i % 2 == 0:
                        forest.add_edge(*edge)
                    else:
                        forest.remove_edge(*edge)

                for i in range(1, len(seq), 2):
                    removed_edge = seq[i]
                    old_owner = assigned.pop(removed_edge)
                    new_edge = seq[i + 1]
                    assigned[new_edge] = old_owner

                assigned[seq[0]] = bus

    return forest, assigned

def recreate_graph(graph: nx.Graph, merges):
        no_nodes = 0
        for superbus, buses in merges.items():
            for bus in buses:
                if bus not in graph.nodes:
                    graph.add_node(np.int64(bus))
                    no_nodes += 1
                graph.add_edge(np.int64(bus), np.int64(superbus))
        return graph

def remove_unusable_injections(maximal_forest, assigned, old_graph, merges):
    return_forest = maximal_forest.copy()
    changed = True
    while changed:
        changed = False
        for edge, bus in list(assigned.items()):
            superbus = get_merge(merges, bus)
            group = [superbus] + merges.get(superbus, [])
            unusable = False

            for member in group:
                for neighbor in old_graph.neighbors(member):
                    neighbor_super = get_merge(merges, neighbor)
                    if neighbor_super == superbus:
                        continue
                    if return_forest.has_edge(superbus, neighbor_super):
                        continue
                    if get_loop(return_forest, superbus, neighbor_super) is not None:
                        continue
                    unusable = True
                    break
                if unusable:
                    break    

            if unusable:
                if return_forest.has_edge(*edge):
                    return_forest.remove_edge(*edge)
                assigned.pop(edge)
                changed = True
                break

    return return_forest, assigned


def predict_islands_baseline(net):
    graph, buses, merges = prepare_merged_graph(net)
    old_graph = pplot.create_nxgraph(net)

    maximal_forest, assigned = augmented_injections(graph, buses, merges)
    maximal_forest = recreate_graph(maximal_forest, merges)
    shrunk_forest, assigned = remove_unusable_injections(maximal_forest, assigned, old_graph, merges)

    components = [c for c in nx.connected_components(shrunk_forest) if len(c) >= 2]
    return [sorted(int(n) for n in c) for c in components]

def evaluate_baseline(paths):
    total_islands_predicted = 0
    total_observable_islands = 0
    total_rank_deficiency = 0
    total_island_size = 0

    network_coverage = 0
    total_network_coverage = 0

    for idx, path in enumerate(paths, 1):
        pkl_path = Path(path)
        json_path = pkl_path.with_suffix('.json')

        if not json_path.exists():
            print(f"[{idx:03d}/{len(paths):03d}] Warning: Missing JSON file for {pkl_path.name}. Skipping.")
            continue

        try:
            with open(json_path) as f:
                record = json.load(f)
                net = pp.from_json_string(record["net_json"])
        except Exception as e:
            print(f"[{idx:03d}/{len(paths):03d}] Error loading {json_path.name}: {e}")
            continue

        predicted_islands = predict_islands_baseline(net)

        file_obs_count = 0
        for island_buses in predicted_islands:
            total_islands_predicted += 1
            total_island_size += len(island_buses)
            network_coverage += len(island_buses)

            island_net = get_subnetwork(net, island_buses)
            if len(island_net.ext_grid) == 0:
                voltage_meas = island_net.measurement[
                    (island_net.measurement.measurement_type == "v")
                    & (island_net.measurement.element_type == "bus")
                ]
                if len(voltage_meas) == 0:
                    continue
                ref_bus = voltage_meas.iloc[0].element
                ref_vm = voltage_meas.iloc[0].value
                pp.create_ext_grid(island_net, bus=ref_bus, vm_pu=ref_vm, va_degree=0.0)
            try:
                obs_result = observability_analysis(island_net)
                total_rank_deficiency += obs_result.rank_deficiency

                if obs_result.observable:
                    total_observable_islands += 1
                    file_obs_count += 1
            except Exception as e:
                print(f"  Error running observability analysis on island {island_buses}: {e}")

        if len(predicted_islands) > 0:
            total_network_coverage += network_coverage / len(net.bus)
        network_coverage = 0
        print(f"[{idx:03d}/{len(paths):03d}] {Path(path).name} | Islands: {len(predicted_islands)} | Observable: {file_obs_count}")

    accuracy = (total_observable_islands / total_islands_predicted * 100) if total_islands_predicted > 0 else 0.0
    avg_deficiency = (total_rank_deficiency / total_islands_predicted) if total_islands_predicted > 0 else 0.0
    avg_island_size = (total_island_size / total_islands_predicted) if total_islands_predicted > 0 else 0.0
    avg_network_coverage = (total_network_coverage / len(paths)) if paths else 0.0

    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS")
    print("="*50)
    print(f"Total Graphs Evaluated:      {len(paths)}")
    print(f"Total Predicted Islands:     {total_islands_predicted}")
    print(f"Verified Observable Islands: {total_observable_islands} ({accuracy:.2f}%)")
    print(f"Average Rank Deficiency:     {avg_deficiency:.4f}")
    print(f"Average Island Size:     {avg_island_size:.4f}")
    print(f"Average Network Coverage:     {avg_network_coverage*100:.4f}%")
    print("="*50)