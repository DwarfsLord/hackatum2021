import math

#Default unit is km
earth_radius = 6378

plane_max_cruising_altitude = 13
plane_min_angle = math.radians(10) #rad


#Inputs
orbit_height = 19000


#Compute the radial distance between two reachable extremes in radians
rad = 2 * (math.pi/2 - plane_min_angle - math.asin(math.sin(math.pi/2 + plane_min_angle)*(earth_radius+plane_max_cruising_altitude)/(earth_radius+orbit_height)))

#Max length (diagonal) of a sector in radians
sector_max_length = rad / 5

#Max horizontal spacing between two sectors
sector_max_width = sector_max_length * 0.75

#sectors around the equator
sector_equator_count = math.ceil(math.pi / sector_max_width)

#Determine actual sector size
sector_width = math.pi / sector_equator_count
sector_length = sector_width * 4/3

#Diamater of inscribed circle (a.k.a. h)
sector_max_h = sector_length * math.sqrt(0.75)

#note the 6* (from 6*pi/3) outside of the ceil(). This guarantees a Number divisible by 6.
sector_belt_count = 6 * math.ceil(math.pi / (sector_max_h*3))

sector_h = 2 * math.pi / sector_belt_count

print(f"Distinct Belts: {sector_equator_count}")
print(f"Sector size efficiency: {sector_width/sector_max_width}")
print(f"Sectors per Belt: {sector_belt_count}")
print("sector_h:", sector_h)
y_radiant_per_belt = sector_h
x_radiant_per_belt = 2*sector_width
print("2 x sector_width: ", 2*sector_width)
print(f"Sector height efficiency: {sector_h/sector_max_h}")
print(f"Total Satellites: {sector_equator_count * sector_belt_count/2:.0f}")
print("sector_max_width:", sector_max_width)

########
import numpy as np


def cell(left, right):
    cell_center = np.array([left, right])
    return cell_center


def cell_corners(cell_center):
    cell_center
    return cell_corners


# Alpha Cell
alpha = np.array([0, 0])
beta = np.array([0, 0])
print("SEC: ", sector_equator_count)
# number_belt= sector_equator_count/2
list_position = [[0, 0]]
for i in range(int(sector_equator_count / 2.0)):
    beta = np.add(beta, np.array([2 * sector_width, 0]))
    # print("Beta: ", beta)
    list_position = np.append(list_position, [beta], axis=0)
    # print("Alpha:", alpha)
    ##list_position = np.append(alpha, beta, axis=0)
    # print("Beta: ", beta)
    alpha = beta
    # list_position = list_position[0]
print(list_position)

from collections import defaultdict

mean_coordinates = []
mean_coordinates_dict = defaultdict(lambda: np.empty((0, 2), float))
for y_belt_count in range(int(sector_belt_count / 4)):
    #     for x_belt_count in range(int(sector_equator_count/2.0)):
    # #         beta = np.add(beta, np.array([2*sector_width, 0]))
    # #         list_position= np.append(list_position, [beta] , axis=0)
    # #         alpha = beta
    #         x_radiant = (x_belt_count +(y_belt_count%2)) * x_radiant_per_belt
    #         y_radiant = y_belt_count * y_radiant_per_belt
    #         mean_coordinates.append([x_radiant, y_radiant])
    # #         mean_coordinates_dict[f"layer {y_belt_count}"].append([x_radiant, y_radiant])
    #         np.hstack((mean_coordinates_dict[f"layer {y_belt_count}"],[x_radiant, y_radiant]))
    mean_coordinates_dict[f"layer {y_belt_count}"] = np.vstack(
        [[(x_belt_count + (y_belt_count % 2) / 2) * x_radiant_per_belt,
          y_belt_count * y_radiant_per_belt]]
        for x_belt_count in range(int(sector_equator_count / 2.0)))

# print(2*sector_max_width)
print(mean_coordinates_dict)
print(mean_coordinates_dict["layer 0"].shape)

from sklearn.metrics import pairwise_distances

# remove the overlapping centroids


for y_belt in range(int(sector_belt_count / 4) - 1):
    # within the current belt
    within_belt_distances = pairwise_distances(mean_coordinates_dict[f"layer {y_belt}"],
                                               mean_coordinates_dict[f"layer {y_belt}"], metric='haversine')
    to_delete = []
    for row_idx, row in enumerate(within_belt_distances):
        for col, element in enumerate(row):
            if element < y_radiant_per_belt and row_idx != col:
                to_delete.append(col)
    np.delete(mean_coordinates_dict[f"layer {y_belt}"], to_delete, 0)
    # inside out, thus in neighboring belts
    pw_distances = pairwise_distances(mean_coordinates_dict[f"layer {y_belt}"],
                                      mean_coordinates_dict[f"layer {y_belt + 1}"], metric='haversine')
    print(f"layer {y_belt} {pw_distances}")
    print("smaller", pw_distances < y_radiant_per_belt)
    for row in pw_distances:
        for col, element in enumerate(row):
            if element < y_radiant_per_belt:
                np.delete(mean_coordinates_dict[f"layer {y_belt + 1}"], col, 0)
point = mean_coordinates_dict[f"layer {1}"][0]
# print(point.shape)
print("pw_distances: ", pw_distances)
# for centroid in mean_coordinates_dict[f"layer {1}"]:
# print(pairwise_distances(mean_coordinates_dict["layer 1"], mean_coordinates_dict["layer 1"], metric='haversine'))
print(mean_coordinates_dict)


result = mean_coordinates_dict.items()
data = list(result)
numpyArray = np.array(data)
print(numpyArray)

true_data_array = numpyArray[0:4,1]
#np.concatenate((true_data_array[0], true_data_array[1], true_data_array[2], true_data_array[3]), 1)
true_data_array= np.concatenate((true_data_array[0], true_data_array[1], true_data_array[2], true_data_array[3]), 0)

print("true:", true_data_array)
print("Dim True_Data:", true_data_array.shape)
# import pdb
# test = np.hstack((true_data_array[:,0],-true_data_array[:,1]))
# pdb.set_trace()

earth_array = np.vstack((true_data_array,
                         -true_data_array,
                         np.vstack((true_data_array[:,0],-true_data_array[:,1])).T,
                         np.vstack((-true_data_array[:,0],true_data_array[:,1])).T
))
