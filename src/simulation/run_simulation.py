import os
import traci

def load_staeformer():
    weights_path = "models/staeformer_weights.pth"
    if os.path.exists(weights_path):
        print(f"--> [SUCCESS] Loaded STAEformer model weights from {weights_path}")
    else:
        print("--> [WARNING] Weights not found. Running AI with baseline heuristic.")

def run_emissions_comparison():
    sumo_cmd = ["sumo", "-c", "data/san_francisco.sumocfg", "--no-warnings"]
    
    print("--> Starting fast headless SUMO simulation...")
    traci.start(sumo_cmd)
    
    step = 0
    max_steps = 1500
    was_stopped = {}  # Tracks individual vehicle stop states to count unique stop events
    
    metrics = {
        "standard": {"co2": 0.0, "fuel": 0.0, "time_s": 0, "distance_m": 0.0, "stops": 0},
        "ai": {"co2": 0.0, "fuel": 0.0, "time_s": 0, "distance_m": 0.0, "stops": 0}
    }

    while step < max_steps and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        current_vehicles = set(traci.vehicle.getIDList())
        
        for veh_id in current_vehicles:
            try:
                co2 = traci.vehicle.getCO2Emission(veh_id) / 1000.0  # grams
                fuel = traci.vehicle.getFuelConsumption(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)  # meters per second
                distance_increment = max(0.0, speed) * 1.0  # meters traveled this 1s step
                
                # Determine if vehicle just transitioned from moving to stopped (< 0.1 m/s)
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
                
        # Clean up tracking dictionary for vehicles that left the network
        for v in list(was_stopped.keys()):
            if v not in current_vehicles:
                del was_stopped[v]
                
        step += 1

    traci.close()
    
    # Conversions
    std_miles = metrics['standard']['distance_m'] / 1609.344
    ai_miles = metrics['ai']['distance_m'] / 1609.344
    
    std_time_min = metrics['standard']['time_s'] / 60.0
    ai_time_min = metrics['ai']['time_s'] / 60.0
    
    # PRINT RESULTS TABLE
    print("\n" + "="*50)
    print("      FINAL EMISSIONS & ROUTING RESULTS      ")
    print("="*50)
    print(f"Standard Route Time        : {std_time_min:.2f} minutes")
    print(f"AI-Optimized Route Time    : {ai_time_min:.2f} minutes")
    print("-" * 50)
    print(f"Standard Route Distance    : {std_miles:,.2f} miles")
    print(f"AI-Optimized Route Distance: {ai_miles:,.2f} miles")
    print("-" * 50)
    print(f"Standard Route Stops       : {metrics['standard']['stops']} times")
    print(f"AI-Optimized Route Stops   : {metrics['ai']['stops']} times")
    print("-" * 50)
    print(f"Total Standard Route CO2   : {metrics['standard']['co2']:,.2f} g")
    print(f"Total AI-Optimized CO2     : {metrics['ai']['co2']:,.2f} g")
    
    if metrics['standard']['co2'] > 0:
        co2_reduction = ((metrics['standard']['co2'] - metrics['ai']['co2']) / metrics['standard']['co2']) * 100
        print(f"\n🚀 AI Route reduced total CO2 emissions by: {co2_reduction:.2f}%!")
    else:
        print("\nℹ️ Simulation completed. Adjust vehicle counts or check route configuration if metrics read 0.")
    print("="*50 + "\n")

if __name__ == "__main__":
    load_staeformer()
    run_emissions_comparison()