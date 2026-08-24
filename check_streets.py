import sumolib

def check_streets():
    print("Reading network map...")
    net = sumolib.net.readNet("C:\\Users\\saros\\Downloads\\best-route-vs-emissions\\data\\san_francisco_city.net.xml")
    
    # Grab all unique street names that allow passenger cars
    streets = set(e.getName() for e in net.getEdges() if e.getName() and e.allows("passenger"))
    
    street_list = sorted(list(streets))
    
    print(f"\n--- Found {len(street_list)} Named Streets ---")
    if len(street_list) == 0:
        print("Whoops! It looks like no street names were saved in the map.")
    else:
        print("Here is a sample of 30 streets you can use in your routes:")
        # Skipping the first few to bypass numbered streets/alleys and get to the major names
        for street in street_list[30:60]: 
            print(f"- {street}")

if __name__ == "__main__":
    check_streets()