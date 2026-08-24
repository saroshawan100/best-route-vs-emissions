import requests
import os
'''
this file connects to a free, open-source mapping database called OpenStreetMap (OSM)—think of it as a crowdsourced version of 
Google Maps. It uses a specific endpoint called the Overpass API, which is a powerful tool that lets developers query and download very specific chunks of map data.
the end goal for this script is to download a .osm file that contains the entire road network of San Francisco. 

The .osm file is just a massive text document full of raw GPS coordinates and road types. 
It doesn't know what a "traffic light" or a "lane" physically looks like yet
run this command in your terminal after .osm file is downloaded to convert it into a more usable format for SUMO

netconvert --osm-files data/san_francisco_city.osm -o data/san_francisco_city.net.xml --geometry.remove --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
'''

import requests
import os
import sys

def download_city_emissions_map():
    print("Downloading San Francisco City Network...")
    
    # Using HTTPS
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # BOUNDING BOX: San Francisco (approx 7x7 miles)
    overpass_query = """
    [out:xml][timeout:300];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link"](37.70,-122.51,37.81,-122.37);
    );
    (._;>;);
    out body;
    """
    
    # The crucial fix: Overpass requires a User-Agent to prevent bot blocking
    headers = {
        'User-Agent': 'SUMO-Emissions-Project/1.0 (Student Research)'
    }
    
    response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers)
    
    # Safety check so we don't save HTML error pages again
    if response.status_code != 200:
        print(f"Server Error {response.status_code}: The map server is currently busy or blocking us.")
        print("Please wait 30 seconds and try running the script again.")
        sys.exit(1)
        
    os.makedirs("data", exist_ok=True)
    output_path = "data/san_francisco_city.osm"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    print(f"Map successfully downloaded and saved to: {output_path}")

if __name__ == "__main__":
    download_city_emissions_map()