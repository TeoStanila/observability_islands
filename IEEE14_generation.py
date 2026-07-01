import os
import sys
import json
import copy
import random
import logging
import warnings
import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt
import pandapower.networks as nw
import pandapower.converter as pc

from tqdm import tqdm
from dataclasses import dataclass, field
from pandapower.estimation import estimate
from pandapower.plotting import simple_plot
from scipy.sparse.linalg import MatrixRankWarning
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra

warnings.filterwarnings("ignore", category=MatrixRankWarning)
logging.getLogger("pandapower").setLevel(logging.CRITICAL)
logging.getLogger("pandapower.estimation").setLevel(logging.CRITICAL)

@dataclass
class ObservabilityResult:
    observable: bool
    rank: int
    no_states: int
    rank_deficiency: int
    unobservable_buses: list = field(default_factory=list)
    unobservable_islands: list = field(default_factory=list)

def build_jacobian(net):
    net2, ppc, eppci = pp2eppci(net, v_start="flat", delta_start="flat")
    sem = BaseAlgebra(eppci)
    H = sem.create_hx_jacobian(eppci.E)

    return np.asarray(H.todense()), eppci

def null_space_vectors(H, rank_tol=1e-8):
    u, s, vh = np.linalg.svd(H)
    tol = rank_tol * max(H.shape) * (s[0] if len(s) else 1.0)
    null_mask = np.zeros(vh.shape[0], dtype=bool)
    null_mask[len(s):] = True
    null_mask[:len(s)] = s < tol

    return vh[null_mask, :].T

def reset_network():
    net = nw.case14()
    pp.runpp(net)

    return net

def candidate_measurements(net):
    candidates = []

    for bus in net.bus.index:
        candidates.append(dict(meas_type="v", element_type="bus", element=int(bus), side=None))

    for line in net.line.index:
        for side in ("from", "to"):
            candidates.append(dict(meas_type="p", element_type="line", element=int(line), side=side))
            candidates.append(dict(meas_type="q", element_type="line", element=int(line), side=side))
    
    for trafo in net.trafo.index:
        for side in ("hv", "lv"):
            candidates.append(dict(meas_type="p", element_type="trafo", element=int(trafo), side=side))
            candidates.append(dict(meas_type="q", element_type="trafo", element=int(trafo), side=side))

    return candidates

def get_measurement_value(net, measurement):
    if measurement["element_type"] == "bus":
        val = net.res_bus.vm_pu[measurement["element"]]
    elif measurement["element_type"] == "line":
        col = f"{measurement["meas_type"]}_{measurement["side"]}_{'mw' if measurement["meas_type"] == "p" else 'mvar'}"
        val = net.res_line[col][measurement["element"]]
    elif measurement["element_type"] == "trafo":
        col = f"{measurement['meas_type']}_{measurement['side']}_{'mw' if measurement["meas_type"] == 'p' else 'mvar'}"
        val = net.res_trafo[col][measurement["element"]]
    else:
        raise ValueError(measurement["element_type"])
    
    return val

def apply_measurements(net, measurements, std_dev_v=1e-2, std_dev_pq=3e-2, sample_measurement_configuration=True):
    if not measurements:
        net.measurement = net.measurement.iloc[0:0]
        return
    
    rows = []

    for m in measurements:
        std_dev = std_dev_v if m["meas_type"] == "v" else std_dev_pq
        value = get_measurement_value(net, m)
        rows.append(dict(
            name=None,
            measurement_type=m["meas_type"],
            element_type=m["element_type"],
            element=m["element"],
            value=value,
            std_dev=std_dev,
            side=m["side"],
        ))
    
    df = pd.DataFrame(rows, columns=["name", "measurement_type", "element_type",
                                     "element", "value", "std_dev", "side"])
    df["element"] = df["element"].astype('uint32')
    df.index = np.arange(len(df))
    net.measurement = df

def sample_measurement_configuration(candidates, keep_prob_range=(0.2, 1.0)):
    p = random.uniform(*keep_prob_range)

    groups = {}
    for m in candidates:
        key = (m["element_type"], m["element"], m["side"])
        groups.setdefault(key, []).append(m)


    selected = []
    for group in groups.values():
        if random.random() < p:
            selected.extend(group)

    return selected
    

def observability_analysis(net, rank_tol=1e-8):
    H, eppci = build_jacobian(net)
    no_states = H.shape[1]

    rank = np.linalg.matrix_rank(H, tol=rank_tol)
    deficiency = int(no_states - rank)
    observable = bool(deficiency == 0)

    unobservable_buses = []
    unobservable_islands = []

    if not observable:
        non_slack_buses = eppci.non_slack_buses
        no_angles = len(non_slack_buses)

        ns = null_space_vectors(H, rank_tol=rank_tol)

        bus_islands = {}
        next_island_id = 0

        for col in range(ns.shape[1]):
            vec = ns[:, col]
            angle_support = set(non_slack_buses[np.where(np.abs(vec[:no_angles]) > 1e-6)[0]].tolist())
            voltage_support = set(np.where(np.abs(vec[no_angles:]) > 1e-6)[0].tolist())
            support = angle_support | voltage_support
            if not support:
                continue 
            
            touched_ids = {bus_islands[b] for b in support if b in bus_islands}
            if touched_ids:
                island_id = min(touched_ids)
                for b, iid in bus_islands.items():
                    if iid in touched_ids:
                        bus_islands[b] = island_id
            else:
                island_id = next_island_id
                next_island_id += 1

            for b in support:
                bus_islands[b] = island_id

        island_map = {}
        for b, iid in bus_islands.items():
            island_map.setdefault(iid, set()).add(b)
        unobservable_islands = [sorted(s) for s in island_map.values()]
        unobservable_buses = sorted(bus_islands.keys())

    return ObservabilityResult(
        observable=observable,
        rank=int(rank),
        no_states=int(no_states),
        rank_deficiency=int(deficiency),
        unobservable_buses=unobservable_buses,
        unobservable_islands=unobservable_islands,
    )

def generate_dataset(no_samples=1000, keep_prob_range=(0.15, 1),
                                dataset_balance=0.5):
    base_net = reset_network()
    candidates = candidate_measurements(base_net)

    records = []
    no_observable = 0
    no_unobservable = 0
    
    with tqdm(total=no_samples, desc="Generating observability dataset") as pbar: 
        while len(records) < no_samples:
            measurements = sample_measurement_configuration(candidates, keep_prob_range)
            if len(measurements) == 0:
                continue

            net = copy.deepcopy(base_net)
            apply_measurements(net, measurements)

            result = observability_analysis(net)

            if not result.observable:
                try:
                    success = estimate(net, init="flat")
                except UserWarning:
                    continue
                if success["success"]:
                    continue

            if len(records) > no_samples / 5:
                observable_ratio = no_observable / len(records)
                if result.observable and observable_ratio > dataset_balance:
                    continue
                elif not result.observable and observable_ratio < dataset_balance:
                    continue

            records.append(dict(
                net_json=pp.to_json(net),
                measurement_config=measurements,
                no_measurements=len(measurements),
                observable=result.observable,
                no_states=result.no_states,
                unobservable_buses=result.unobservable_buses,
                unobservable_islands=result.unobservable_islands,
                rank_deficiency=result.rank_deficiency,
            ))
            if result.observable:
                no_observable += 1
            else:
                no_unobservable += 1

            pbar.update(1)

            if len(measurements) == 0:
                print("si s-a si salvat ce dracu")

    print(f"{no_observable} observable / {no_unobservable} unobservable islands.")
        
    return records, candidates

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Specify args: no_samples and save_dict.")
        sys.exit()
    else:
        try:
            no_samples = int(sys.argv[1])
        except:
            print("First arg must be integer.")
            sys.exit()

    folder_name = "IEEE14_datasets"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    save_name = sys.argv[2]

    records, candidates = generate_dataset(
        no_samples=no_samples,
        keep_prob_range=(0.15, 1.0),
        dataset_balance=0.5,
    )

    if not os.path.exists(os.path.join(folder_name, save_name)):
        os.mkdir(os.path.join(folder_name, save_name))

    if not os.path.exists(os.path.join(folder_name, save_name, "observable")):
        os.mkdir(os.path.join(folder_name, save_name, "observable"))

    if not os.path.exists(os.path.join(folder_name, save_name, "unobservable")):
        os.mkdir(os.path.join(folder_name, save_name, "unobservable"))

    with open(os.path.join(folder_name, save_name, "candidates.json"), "w") as f:
        json.dump(candidates, f, indent=2)

    for i, record in enumerate(records):
        if record["observable"]:
            record_path = os.path.join(folder_name, save_name, "observable", f"record_{i:04d}.json")
        else:
            record_path = os.path.join(folder_name, save_name, "unobservable", f"record_{i:04d}.json")
        with open(record_path, "w") as f:
            json.dump(record, f)