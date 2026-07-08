import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandapower as pp
import pandapower.plotting as pplot
from pandapower.estimation import estimate

DATASET_PATH = "IEEE14_datasets"

def load_record(dataset, record_id):
    path = os.path.join(DATASET_PATH, dataset, "observable", f"record_{record_id:04d}.json")
    if not os.path.exists(path):
        path = os.path.join(DATASET_PATH, dataset, "unobservable", f"record_{record_id:04d}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}.")
    with open(path) as f:
        return json.load(f)
    
def get_measuremed_elements(record):
    net = pp.from_json_string(record["net_json"])

    meas_buses = set()
    meas_lines = set()
    meas_trafos = set()

    for m in record["measurement_config"]:
        element = int(m["element"])
        if m["element_type"] == "bus":
            meas_buses.add(element)
        if m["element_type"] == "line":
            meas_lines.add(element)
        if m["element_type"] == "trafo":
            meas_trafos.add(element)

    return net, meas_buses, meas_lines, meas_trafos

def visualize(dataset, record):
    record = load_record(dataset, record)
    net, meas_buses, meas_lines, meas_trafos = get_measuremed_elements(record)

    buses = set(net.bus.index.tolist())
    unmeas_buses = buses - meas_buses
    lines = set(net.line.index.tolist())
    unmeas_lines = lines - meas_lines
    trafos = set(net.trafo.index.tolist())
    unmeas_trafos = trafos - meas_trafos

    collections = []

    if unmeas_buses:
        collections.append(pplot.create_bus_collection(
            net, buses=sorted(unmeas_buses), color="#90ee90", size=0.08
        ))
    if meas_buses:
        collections.append(pplot.create_bus_collection(
            net, buses=sorted(meas_buses), color="#d62728", size=0.08
        ))
    if unmeas_lines:
        collections.append(pplot.create_line_collection(
            net, lines=sorted(unmeas_lines), color="#90ee90", linewidth=1.5
        ))
    if meas_lines:
        collections.append(pplot.create_line_collection(
            net, lines=sorted(meas_lines), color="#d62728", linewidth=1.5
        ))
    if unmeas_trafos:
        collections.append(pplot.create_trafo_collection(
            net, trafos=sorted(unmeas_trafos), color="#90ee90", size=0.2
        ))
    if meas_trafos:
        collections.append(pplot.create_trafo_collection(
            net, trafos=sorted(meas_trafos), color="#d62728", size=0.2
        ))

    print(f"Observable: {record["observable"]}")
    print(f"Rank deficiency: {record["rank_deficiency"]}")
    success = estimate(net, init="flat")
    print(f"Success: {success["success"]}")
    ax = pplot.draw_collections(collections, figsize=(12, 9), draw=False)

    for bus_id in net.bus.index.tolist():
        geo = net.bus.at[bus_id, "geo"]
        if geo is None or not isinstance(geo, str):
            continue
        coords = json.loads(geo)["coordinates"]
        ax.annotate(str(bus_id), xy=(coords[0], coords[1]), fontsize=15, color="blue",
                    xytext=(4, 4), textcoords="offset points")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Specify args: dataset_name and record_id.")
        sys.exit(1)
 
    dataset = sys.argv[1]
    try:
        record_id = int(sys.argv[2])
    except ValueError:
        print("Second arg must be an integer.")
        sys.exit(1)
 
    visualize(dataset, record_id)

        