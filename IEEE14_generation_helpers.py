import json
import copy
import random
import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt
import pandapower.networks as nw
import pandapower.converter as pc

from dataclasses import dataclass, field
from pandapower.estimation import estimate
from pandapower.plotting import simple_plot
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.estimation.algorithm.matrix_base import BaseAlgebra

@dataclass
class ObservabilityResult:
    observable: bool
    rank: int
    no_states: int
    rank_deficiency: int
    unobservanle_buses: list = field(default_factory=list)
    unobservable_islands: list = field(default_factory=list)

def build_jacobian(net):
    net2, ppc, eppci = pp2eppci(net, v_start="flat", delta_start="flat")
    sem = BaseAlgebra(eppci)
    H = sem.create_hx_jacobian(eppci.E)

    return np.asarray(H.todense()), eppci

def null_space(H, rank_tol=1e-8):
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
    else:
        raise ValueError(measurement["element_type"])
    
    return val

def apply_measurements(net, measurements, std_dev_v=1e-2, std_dev_pq=3e-2, noisy=True):
    if not measurements:
        net.measurement = net.measurement.iloc[0:0]
        return
    
    rows = []

    for measurement in measurements:
        std_dev = std_dev_v if measurement["meas_type"] == "v" else std_dev_pq
        value = get_measurement_value(net, measurement)
        rows.append(dict(
            name=None,
            measurement_type=measurement["meas_type"],
            element_type=measurement["element_type"],
            element=measurement["element"],
            value=value,
            std_dev=std_dev,
            side=measurement["side"],
        ))
    
    df = pd.DataFrame(rows, columns=["name", "measurement_type", "element_type",
                                     "element", "value", "std_dev", "side"])
    df["element"] = df["element"].astype('uint32')
    df.index = np.arange(len(df))
    net.measurement = df

def sample_measurement_configuration(candidates, keep_prob_range=(0.2, 1.0), rng=random):
    p = rng.uniform(*keep_prob_range)

    return [measurement for measurement in candidates if rng.random() < p] 
    

def check_observability(net, rank_tol=1e-8):
    H, eppci = build_jacobian(net)
    no_states = H.shape[1]

    rank = np.linalg.matrix_rank(H, tol=rank_tol)
    deficiency = int(no_states - rank)
    observable = bool(deficiency == 0)

    unobs_buses = []
    islands = []

    if not observable:
        non_slack_buses = eppci.non_slack_buses
        no_angles = len(non_slack_buses)

        ns = null_space(H, rank_tol=rank_tol)

        bus_islands = {}
        next_island_id = 0

        for col in range(ns.shape[1]):
            vec = ns[:, col]
            angle_support = set(non_slack_buses[np.where(np.abs(vec[:no_angles]) > 1e-6)[0]].tolist())
            voltage_support = set(np.where(np.abs(vec[no_angles:]) > 1e-6)[0].tolist())
            support = angle_support | voltage_support
            if not support:
                continue 
            
            rouched

if __name__ == "__main__":
    net = reset_network()
    print(net.bus)