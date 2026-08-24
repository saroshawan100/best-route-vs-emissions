import os
import subprocess
import traci
import matplotlib.pyplot as plt
import numpy as np


TRAFFIC_SCALE = 1.2

def run_simulation_for_route(start_street, end_street, scale=TRAFFIC_SCALE):
    print(f"\n--> Evaluating Route: {start_street} -> {end_street} (Traffic Scale: {scale})")
    
    subprocess.run([
        "python", "src/routing/build_routes.py", 
        "--start", start_street, 
        "--end", end_street
    ], check=True)
    
    sumo_cmd = ["sumo", "-c", "data/san_francisco.sumocfg", "--no-warnings", "--scale", str(scale)]
    traci.start(sumo_cmd)
    
    step = 0
    max_steps = 1500
    was_stopped = {}
    
    metrics = {
        "standard": {"co2": 0.0, "fuel": 0.0, "time_s": 0, "distance_m": 0.0, "stops": 0},
        "ai": {"co2": 0.0, "fuel": 0.0, "time_s": 0, "distance_m": 0.0, "stops": 0}
    }

    while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_vehicles = set(traci.vehicle.getIDList())
        
        for veh_id in current_vehicles:
            try:
                co2 = traci.vehicle.getCO2Emission(veh_id) / 1000.0  
                fuel = traci.vehicle.getFuelConsumption(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)  
                distance_increment = max(0.0, speed) * 1.0  
                
                is_stopped = speed < 0.1
                prev_stopped = was_stopped.get(veh_id, False)
                new_stop = (not prev_stopped) and is_stopped
                was_stopped[veh_id] = is_stopped
                
                if "ai" in veh_id:
                    metrics["ai"]["co2"] += co2
                    metrics["ai"]["fuel"] += fuel
                    metrics["ai"]["distance_m"] += distance_increment
                    metrics["ai"]["time_s"] += 1
                    if new_stop:
                        metrics["ai"]["stops"] += 1
                    if step % 10 == 0:
                        traci.vehicle.rerouteTraveltime(veh_id)
                else:
                    metrics["standard"]["co2"] += co2
                    metrics["standard"]["fuel"] += fuel
                    metrics["standard"]["distance_m"] += distance_increment
                    metrics["standard"]["time_s"] += 1
                    if new_stop:
                        metrics["standard"]["stops"] += 1
            except:
                pass
                
        for v in list(was_stopped.keys()):
            if v not in current_vehicles:
                del was_stopped[v]
        step += 1

    traci.close()
    
    return {
        "std_time": metrics["standard"]["time_s"] / 60.0,
        "ai_time": metrics["ai"]["time_s"] / 60.0,
        "std_dist": metrics["standard"]["distance_m"] / 1609.344,
        "ai_dist": metrics["ai"]["distance_m"] / 1609.344,
        "std_stops": metrics["standard"]["stops"],
        "ai_stops": metrics["ai"]["stops"],
        "std_co2": metrics["standard"]["co2"],
        "ai_co2": metrics["ai"]["co2"]
    }

def generate_diverse_presentation_assets(res_r1, res_r2, scale=TRAFFIC_SCALE):
    print(f"\n--> Generating Diverse Presentation Visuals (.png) for Scale {scale}...")
    labels = ['Route 1 (Geary -> Mission)', 'Route 2 (Market -> Van Ness)']
    width = 0.35
    x = np.arange(len(labels))
    
    # 1. Vertical Bar Charts (CO2, Time, Stops)
    for metric_name, std_val, ai_val, filename, ylabel in [
        ('CO2 Emissions', [res_r1['std_co2'], res_r2['std_co2']], [res_r1['ai_co2'], res_r2['ai_co2']], 'co2_comparison.png', 'CO2 (g)'),
        ('Travel Time', [res_r1['std_time'], res_r2['std_time']], [res_r1['ai_time'], res_r2['ai_time']], 'time_comparison.png', 'Time (min)'),
        ('Stop Events', [res_r1['std_stops'], res_r2['std_stops']], [res_r1['ai_stops'], res_r2['ai_stops']], 'stops_comparison.png', 'Stop Count')
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, std_val, width, label='Standard', color='#e74c3c')
        ax.bar(x + width/2, ai_val, width, label='AI-Optimized', color='#2ecc71')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison (Scale {scale})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    # 2. Horizontal Percentage Improvement Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_list = ['Time Saved', 'Distance Optimized', 'Stops Reduced', 'CO2 Reduced']
    
    r1_improvements = [
        ((res_r1['std_time'] - res_r1['ai_time']) / res_r1['std_time'] * 100) if res_r1['std_time'] > 0 else 0,
        ((res_r1['std_dist'] - res_r1['ai_dist']) / res_r1['std_dist'] * 100) if res_r1['std_dist'] > 0 else 0,
        ((res_r1['std_stops'] - res_r1['ai_stops']) / res_r1['std_stops'] * 100) if res_r1['std_stops'] > 0 else 0,
        ((res_r1['std_co2'] - res_r1['ai_co2']) / res_r1['std_co2'] * 100) if res_r1['std_co2'] > 0 else 0,
    ]
    
    y_pos = np.arange(len(metrics_list))
    ax.barh(y_pos, r1_improvements, color='#3498db', align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_list, fontsize=11, fontweight='bold')
    ax.invert_yaxis()  
    ax.set_xlabel('Percentage Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'AI Performance Gains: Route 1 (Scale {scale})', fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(r1_improvements):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    plt.savefig('percentage_gains_route1.2.png', dpi=300)
    plt.close()

    # 3. NEW: Comprehensive Table Image (table.png)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('tight')
    ax.axis('off')
    
    cell_text = [
        ["Route 1 (Geary -> Mission)", "Time (min)", f"{res_r1['std_time']:.2f}", f"{res_r1['ai_time']:.2f}", f"{((res_r1['std_time'] - res_r1['ai_time'])/res_r1['std_time']*100):+.1f}%"],
        ["Route 1 (Geary -> Mission)", "Distance (mi)", f"{res_r1['std_dist']:.2f}", f"{res_r1['ai_dist']:.2f}", f"{((res_r1['std_dist'] - res_r1['ai_dist'])/res_r1['std_dist']*100):+.1f}%"],
        ["Route 1 (Geary -> Mission)", "Stop Events", f"{res_r1['std_stops']}", f"{res_r1['ai_stops']}", f"{((res_r1['std_stops'] - res_r1['ai_stops'])/res_r1['std_stops']*100):+.1f}%"],
        ["Route 1 (Geary -> Mission)", "CO2 (g)", f"{res_r1['std_co2']:.2f}", f"{res_r1['ai_co2']:.2f}", f"{((res_r1['std_co2'] - res_r1['ai_co2'])/res_r1['std_co2']*100):+.1f}%"],
        
        ["Route 2 (Market -> Van Ness)", "Time (min)", f"{res_r2['std_time']:.2f}", f"{res_r2['ai_time']:.2f}", f"{((res_r2['std_time'] - res_r2['ai_time'])/res_r2['std_time']*100):+.1f}%"],
        ["Route 2 (Market -> Van Ness)", "Distance (mi)", f"{res_r2['std_dist']:.2f}", f"{res_r2['ai_dist']:.2f}", f"{((res_r2['std_dist'] - res_r2['ai_dist'])/res_r2['std_dist']*100):+.1f}%"],
        ["Route 2 (Market -> Van Ness)", "Stop Events", f"{res_r2['std_stops']}", f"{res_r2['ai_stops']}", f"{((res_r2['std_stops'] - res_r2['ai_stops'])/res_r2['std_stops']*100):+.1f}%"],
        ["Route 2 (Market -> Van Ness)", "CO2 (g)", f"{res_r2['std_co2']:.2f}", f"{res_r2['ai_co2']:.2f}", f"{((res_r2['std_co2'] - res_r2['ai_co2'])/res_r2['std_co2']*100):+.1f}%"],
    ]
    
    columns = ["Route", "Metric", "Standard", "AI-Optimized", "Improvement"]
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f8f9f9' if key[0] % 2 == 0 else '#ffffff')
            
    plt.title(f"Comprehensive Evaluation Summary Table (Scale {scale})", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('table1.2.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[SUCCESS] All presentation graphics & table.png successfully generated!")

if __name__ == "__main__":
    print("="*50)
    print(f" SF TRAFFIC AI: DIVERSE ASSET GENERATION (SCALE {TRAFFIC_SCALE}) ")
    print("="*50)
    
    res_r1 = run_simulation_for_route("Geary", "Mission", scale=TRAFFIC_SCALE)
    res_r2 = run_simulation_for_route("Market", "Van Ness", scale=TRAFFIC_SCALE)
    
    generate_diverse_presentation_assets(res_r1, res_r2, scale=TRAFFIC_SCALE)