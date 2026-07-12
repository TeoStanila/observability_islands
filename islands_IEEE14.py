import os
import sys
import copy
import json
import pprint
import random
import logging
from collections import defaultdict
from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.plotting as plot
import tqdm
import pandas as pd
from pandapower.toolbox import drop_buses
from scipy.sparse.linalg import MatrixRankWarning

from visualization_IEEE14 import load_record
from generation_IEEE14 import observability_analysis

filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
filterwarnings("ignore", category=MatrixRankWarning)
logging.getLogger("pandapower").setLevel(logging.CRITICAL)

class UnionFind:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def same(self, x, y):
        return self.find(x) == self.find(y)
    
    def islands(self, nodes):
        groups = defaultdict(list)
        for n in nodes:
            groups[self.find(n)].append(n)
        return {root: frozenset(members) for root, members in groups.items()}

def measurement_cleanup(net):
    net.measurement = net.measurement[
        ~(
            (net.measurement.element_type=="bus")
            &
            (~net.measurement.element.isin(net.bus.index))
        )
    ]

    net.measurement = net.measurement[
        ~(
            (net.measurement.element_type=="line")
            &
            (~net.measurement.element.isin(net.line.index))
        )
    ]

    net.measurement = net.measurement[
        ~(
            (net.measurement.element_type=="trafo")
            &
            (~net.measurement.element.isin(net.trafo.index))
        )
    ]
    
def get_subnetwork(net, bus_ids, lines_to_drop=None, trafos_to_drop=None):
    bus_set = set(bus_ids)
    buses_to_drop = [b for b in net.bus.index if b not in bus_set]
    
    subnet = copy.deepcopy(net)
    drop_buses(subnet, buses_to_drop)
    measurement_cleanup(subnet)

    if lines_to_drop:
        valid_lines = [l for l in lines_to_drop if l in subnet.line.index]
        subnet.line.drop(valid_lines, inplace=True)
    
    if trafos_to_drop:
        valid_trafos = [t for t in trafos_to_drop if t in subnet.trafo.index]
        subnet.trafo.drop(valid_trafos, inplace=True)

    measurement_cleanup(subnet)

    return subnet

def check_island(net, bus_ids, lines_to_drop=None, trafos_to_drop=None):
    if len(bus_ids) < 2:
        return False, None
    try:
        subnet = get_subnetwork(net, bus_ids, lines_to_drop, trafos_to_drop)
        voltage_meas = subnet.measurement[subnet.measurement.element_type == "bus"]

        if subnet.ext_grid is None:
            subnet.ext_grid = pp.create_empty_network().ext_grid
        if len(subnet.ext_grid) == 0:
            if len(voltage_meas) == 0:
                return False, None
            ref_bus = voltage_meas.iloc[0].element
            ref_vm = voltage_meas.iloc[0].value
            pp.create_ext_grid(subnet, bus=ref_bus, vm_pu=ref_vm, va_degree=0.0)


        if len(subnet.bus) < 2:
            return False, None

        result = observability_analysis
        return result.observable, subnet
    
    except(Exception, UserWarning):
        return False, None
    
def save_network_drawing(subnet, config_path):
    img_path = os.path.splitext(config_path)[0] + ".png"
    try:
        has_coords = hasattr(subnet, "bus_geodata") and len(subnet.bus_geodata) > 0
        if not has_coords:
            plot.create_generic_coordinates(subnet)

        ax = plot.simple_plot(subnet, show_plot=False)
        fig = getattr(ax, "figure", None) or plt.gcf()
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close("all")
        return None

    return img_path
    
def sample_forest(net, samples_per_island=3, record_id="unknown", isl_registry=None, seen_configs=None):
    if isl_registry is None:
        isl_registry = {}
    if seen_configs == None:
        seen_configs = set()

    buses = net.bus.index.tolist()
    uf = UnionFind(buses)
    meas = net.measurement

    island_snapshots = []
    

    flow_meas = meas[meas.element_type.isin(["line", "trafo"])]
    for _, m in flow_meas.iterrows():
        eid = m.element
        if m.element_type == "line":
            fb = net.line.at[eid, "from_bus"]
            tb = net.line.at[eid, "to_bus"]
        else:
            fb = net.trafo.at[eid, "hv_bus"]
            tb = net.trafo.at[eid, "lv_bus"]
        if uf.union(fb, tb):
            island_snapshots.append(copy.deepcopy(uf))

    adjacency = defaultdict(list)
    for lid, row in net.line.iterrows():
        adjacency[row.from_bus].append(row.to_bus)
        adjacency[row.to_bus].append(row.from_bus)
    for tid, row in net.trafo.iterrows():
        adjacency[row.hv_bus].append(row.lv_bus)
        adjacency[row.lv_bus].append(row.hv_bus)

    inj_buses = meas[meas.element_type == "bus"].element.tolist()
    random.shuffle(inj_buses)

    for bus in inj_buses:
        candidates = [nb for nb in adjacency[bus] if not uf.same(bus, nb)]
        if not candidates:
            continue
        chosen = random.choice(candidates)
        if uf.union(bus, chosen):
            island_snapshots.append(copy.deepcopy(uf))

    island_snapshots.append(uf)

    
    raw_islands = {}
    for snapshot in island_snapshots:
        for members in snapshot.islands(buses).values():
            raw_islands[members] = True
            
    valid = []
    measured_lines = set(flow_meas[flow_meas.element_type == "line"].element.tolist())
    measured_trafos = set(flow_meas[flow_meas.element_type == "trafo"].element.tolist())

    candidate_islands = [isl for isl in raw_islands.keys() if len(isl) >= 2]
    base_results = {isl: check_island(net, isl) for isl in candidate_islands}
    successful_islands = [isl for isl in candidate_islands if base_results[isl][0]]

    def redundant_island(island_buses):
        return any(
            island_buses < other
            for other in successful_islands
            if other != island_buses
        )
    
    islands_to_process = {
        isl for isl in candidate_islands
        if not (base_results[isl][0] and redundant_island(isl))
    }
    
    for island_buses in islands_to_process:
        if island_buses in isl_registry:
            continue

        island_id = len(isl_registry)
        isl_registry[island_buses] = island_id

        start_sampling = 0
        save_path = os.path.join("islands", dataset_dir, f"record_{record_id}", f"island_{island_id}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        result, subnet = check_island(net, island_buses)
        if result:
            lines = [f"{f}-{t}" for f, t in zip(subnet.line.from_bus, subnet.line.to_bus)]
            trafos = [f"{h}-{l}" for h, l in zip(subnet.trafo.hv_bus, subnet.trafo.lv_bus)]

            config = (frozenset(island_buses), frozenset(lines), frozenset(trafos))
            if config not in seen_configs:
                seen_configs.add(config)
                config_path = os.path.join(save_path, f"config_{start_sampling}.json")
                pp.to_json(subnet, config_path)
                image_path = save_network_drawing(subnet, config_path)
                valid.append({"subnet_path": config_path, "buses": list(island_buses), "lines": lines, "trafos": trafos})
                start_sampling += 1


        bus_set = set(island_buses)
        internal_lines = net.line[net.line.from_bus.isin(bus_set) & net.line.to_bus.isin(bus_set)].index.tolist()
        internal_trafos = net.trafo[net.trafo.hv_bus.isin(bus_set) & net.trafo.lv_bus.isin(bus_set)].index.tolist()

        unmeas_lines = list(set(internal_lines) - measured_lines)
        unmeas_trafos = list(set(internal_trafos) - measured_trafos)
        total_unmeas = len(unmeas_lines) + len(unmeas_trafos)

        # Start trimming forest
        if total_unmeas > 0:
            config_id = start_sampling
            target_configs = start_sampling + samples_per_island
            max_attempts = samples_per_island * 3
            attempts = 0

            while config_id < target_configs and attempts < max_attempts:
                attempts += 1
                config_path = os.path.join(save_path, f"config_{config_id}.json")
                n_drop = random.randint(1, max(1, total_unmeas // 2))
                pool = [('line', l) for l in unmeas_lines] + [('trafo', t) for t in unmeas_trafos]
                to_drop = random.sample(pool, min(n_drop, len(pool)))

                lines_to_drop = [eid for etype, eid in to_drop if etype == 'line']
                trafos_to_drop = [eid for etype, eid in to_drop if etype == 'trafo']

                result, subnet = check_island(net, island_buses, lines_to_drop=lines_to_drop, trafos_to_drop=trafos_to_drop)
                if result:
                    observed_buses = [b for b in subnet.bus.index.tolist() if b in island_buses]

                    if len(observed_buses) >= 2:
                        lines = [f"{f}-{t}" for f, t in zip(subnet.line.from_bus, subnet.line.to_bus)]
                        trafos = [f"{h}-{l}" for h, l in zip(subnet.trafo.hv_bus, subnet.trafo.lv_bus)]
                        
                        config = (frozenset(observed_buses), frozenset(lines), frozenset(trafos))
                        if config not in seen_configs:
                            seen_configs.add(config)
                            pp.to_json(subnet, config_path)
                            # image_path = save_network_drawing(subnet, config_path)
                            valid.append({
                                "subnet_path": config_path,
                                "buses": observed_buses,
                                "lines": lines,
                                "trafos": trafos,
                            })
                            config_id += 1
                            

    return valid


def sample_configurations(net, n_samples=50, record_id="unknown"):
    configs = []
    isl_registry = {}
    seen_configs = set()

    for _ in range(n_samples):
        islands = sample_forest(net, record_id=record_id, isl_registry=isl_registry, seen_configs=seen_configs)
        if len(islands) > 0:
            configs.extend(islands)

    return configs

if __name__ == "__main__":
    all_results = {}
    successes = 0
    failures = 0

    if len(sys.argv) < 3:
        print("Specify number of files to parse and the dataset directory.")
        sys.exit()
    n_files = int(sys.argv[1])
    dataset_dir = sys.argv[2]
    save_dir = os.path.join("islands", sys.argv[2])

    if not os.path.exists("islands"):
        os.makedirs("islands")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for record in range(n_files):
        if not os.path.exists(os.path.join(save_dir, f"record_{record}")):
            os.makedirs(os.path.join(save_dir, f"record_{record}"))

    for record_id in tqdm.tqdm(range(n_files), desc="Sampling island configurations"):
        record = load_record(dataset_dir, record_id)
        if record["observable"]:
            continue
        net = pp.from_json_string(record["net_json"])

        configs = sample_configurations(net, n_samples=50, record_id=str(record_id))
        if len(configs) > 0:
            all_results[int(record_id)] = {
                "observable": False,
                "n_unique_configs": len(configs),
                "configurations": configs
            }

            with open(os.path.join(save_dir, "islands_records.json"), "w") as f:
                json.dump(all_results, f, indent=2)

    

    for current, _, _ in os.walk(save_dir, topdown=False):
        if current == dataset_dir:
            continue

        if not os.listdir(current):
            os.rmdir(current)