import os
import sumolib
import xml.etree.ElementTree as ET

import os
import sumolib
import argparse

# Argument parser for dynamic route generation
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=str, default="Geary")
parser.add_argument("--end", type=str, default="Mission")
args = parser.parse_args()

START_STREET = args.start
END_STREET = args.end
NUM_VEHICLES = 1  # 1 standard, 1 AI car per run for precise comparison

net_path = "data/san_francisco_city.net.xml"
net = sumolib.net.readNet(net_path)

print(f"--> Selected Origin: {START_STREET}")
print(f"--> Selected Destination: {END_STREET}")

def find_edge_by_street(net, street_name):
    """Finds the longest edge matching a street name."""
    matching_edges = [
        e for e in net.getEdges() 
        if e.allows("passenger") and e.getName() and street_name.lower() in e.getName().lower()
    ]
    if not matching_edges:
        return None
    # Sort by length to get a main segment rather than a tiny intersection stub
    matching_edges.sort(key=lambda e: e.getLength(), reverse=True)
    return matching_edges[0]

def generate_comparative_routes():
    net_path = "C:\\Users\\saros\\Downloads\\best-route-vs-emissions\\data\\san_francisco_city.net.xml"
    out_route_path = "C:\\Users\\saros\\Downloads\\best-route-vs-emissions\\data\\san_francisco_city.rou.xml"

    if not os.path.exists(net_path):
        print(f"Error: Could not find {net_path}. Make sure netconvert ran successfully.")
        return

    print("--> Reading San Francisco SUMO Network...")
    net = sumolib.net.readNet(net_path)

    # Find edges matching requested street names
    start_edge = find_edge_by_street(net, START_STREET)
    end_edge = find_edge_by_street(net, END_STREET)

    # Fallback to defaults if specific street names aren't found
    passenger_edges = [e for e in net.getEdges() if e.allows("passenger")]
    if not start_edge:
        print(f"--> Warning: Could not find street '{START_STREET}'. Using default start edge.")
        start_edge = passenger_edges[0]
    if not end_edge:
        print(f"--> Warning: Could not find street '{END_STREET}'. Using default end edge.")
        end_edge = passenger_edges[-1]

    print(f"--> Selected Origin: {start_edge.getID()} ({start_edge.getName()})")
    print(f"--> Selected Destination: {end_edge.getID()} ({end_edge.getName()})")

    # Compute Standard Shortest Path
    standard_route_edges, _ = net.getShortestPath(start_edge, end_edge)
    if not standard_route_edges:
        print("--> Error: No connected route exists between these two points. Try different streets.")
        return

    standard_edge_ids = " ".join([e.getID() for e in standard_route_edges])

    root = ET.Element("routes")

    # Define Vehicle Type
    vtype = ET.SubElement(root, "vType", {
        "id": "gas_car",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "5.0",
        "minGap": "2.5",
        "maxSpeed": "15.0",
        "emissionClass": "HBEFA3/PC_G_EU4"
    })

    # Define standard base route
    ET.SubElement(root, "route", {
        "id": "standard_route",
        "edges": standard_edge_ids
    })

    # Spawn Standard Vehicles (stay on standard static path)
    for i in range(NUM_VEHICLES):
        ET.SubElement(root, "vehicle", {
            "id": f"standard_car_{i}",
            "type": "gas_car",
            "route": "standard_route",
            "depart": str(i * 5),
            "color": "1,0,0"  # Red
        })

    # Spawn AI Vehicles (dynamically rerouted away from red lights via TraCI)
    for i in range(NUM_VEHICLES):
        ET.SubElement(root, "vehicle", {
            "id": f"ai_car_{i}",
            "type": "gas_car",
            "route": "standard_route",
            "depart": str(i * 5 + 2),
            "color": "0,1,0"  # Green
        })

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_route_path, encoding="utf-8", xml_declaration=True)

    print(f"--> Route file successfully created at: {out_route_path}")

if __name__ == "__main__":
    generate_comparative_routes()