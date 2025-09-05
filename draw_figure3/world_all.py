import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

# Extract latitude and longitude information from TXT file
def extract_coordinates_from_txt(metadata_file):
    latitudes = []
    longitudes = []

    # Read the file and extract latitude and longitude
    with open(metadata_file, 'r') as f:
        content = f.read()

        # Use regular expressions to extract the latitude and longitude of the four corner points
        lat_pattern = r"CORNER_(UL|UR|LL|LR)_LAT_PRODUCT = ([\d.-]+)"
        lon_pattern = r"CORNER_(UL|UR|LL|LR)_LON_PRODUCT = ([\d.-]+)"
        
        latitudes = [float(match[1]) for match in re.findall(lat_pattern, content)]
        longitudes = [float(match[1]) for match in re.findall(lon_pattern, content)]

    return latitudes, longitudes

# Extract latitude and longitude information from an XML file
def extract_coordinates_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the latitude and longitude of the four corner points
    top_left_lat = float(root.find('TopLeftLatitude').text)
    top_left_lon = float(root.find('TopLeftLongitude').text)
    top_right_lat = float(root.find('TopRightLatitude').text)
    top_right_lon = float(root.find('TopRightLongitude').text)
    bottom_right_lat = float(root.find('BottomRightLatitude').text)
    bottom_right_lon = float(root.find('BottomRightLongitude').text)
    bottom_left_lat = float(root.find('BottomLeftLatitude').text)
    bottom_left_lon = float(root.find('BottomLeftLongitude').text)

    # Return latitude and longitude information
    coordinates = {
        "TopLeft": (top_left_lat, top_left_lon),
        "TopRight": (top_right_lat, top_right_lon),
        "BottomRight": (bottom_right_lat, bottom_right_lon),
        "BottomLeft": (bottom_left_lat, bottom_left_lon)
    }

    return coordinates

# Iterate through the folder to read all TXT metadata files and extract latitude and longitude.
def extract_all_coordinates_from_txt(root_folder):
    all_coordinates = []

    # Recursively scan all subfolders
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".txt"):
                metadata_file = os.path.join(root, file)
                latitudes, longitudes = extract_coordinates_from_txt(metadata_file)
                if latitudes and longitudes:
                    all_coordinates.append((latitudes, longitudes))

    return all_coordinates

# Walk through the folder, read all XML metadata files, and extract latitude and longitude
def extract_all_coordinates_from_xml(root_folder):
    all_coordinates = []

    # Recursively scan all subfolders
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".xml"):
                xml_file = os.path.join(root, file)
                coordinates = extract_coordinates_from_xml(xml_file)
                all_coordinates.append(coordinates)

    return all_coordinates

# Render polygons for all scenes
# def plot_scenes_on_map(txt_coordinates, xml_coordinates, spars_coordinates, L38_coordinates, output_path):
def plot_scenes_on_map(txt_coordinates, xml_coordinates, output_path):
    # Create a map object
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

    # 绘制地图
    m.drawcountries(linewidth=0.2)
    m.drawcoastlines(linewidth=0.5)

    for latitudes, longitudes in L38_coordinates:
        # Coordinate order: top-left, top-right, bottom-right, bottom-left
        x = [longitudes[0], longitudes[1], longitudes[3], longitudes[2]]
        y = [latitudes[0], latitudes[1], latitudes[3], latitudes[2]]

        # Convert latitude and longitude to map coordinates
        x_map, y_map = m(x, y)

        # Create a polygon object and fill the scene area
        polygon = Polygon(list(zip(x_map, y_map)), closed=True, color='#EEBE00', alpha=1, edgecolor='y', linewidth=0)
        plt.gca().add_patch(polygon)


    plt.title("Biome Scenes on World Map")

    plt.savefig(output_path, dpi=500, bbox_inches='tight') 
    plt.close() 


if __name__ == "__main__":
    Biome = "/data/cloud/biome/BC/"
    spars = "/home/ljs/code/sending/"
    GF1 = "/data/cloud/GF1/GF1_WHU/" 
    L38 = "/home/ljs/code/38-Cloud_Training_Metadata_Files"
    output_path = "/home/ljs/code/world_img/biome.png"

    L38_coordinates = extract_all_coordinates_from_txt(L38)
    spars_coordinates = extract_all_coordinates_from_txt(spars)
    Biome_coordinates = extract_all_coordinates_from_txt(Biome)
    GF1_coordinates = extract_all_coordinates_from_xml(GF1)

    plot_scenes_on_map(Biome_coordinates, GF1_coordinates, output_path)
    print(f"map save to {output_path}")