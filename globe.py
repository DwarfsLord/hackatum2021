from math import pi, sin, asin, cos, radians, degrees, sqrt, ceil

import math
import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from scipy.spatial import Voronoi

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import AmbientLight, Point3, loadPrcFileData, NodePath, LineSegs 
from direct.gui.DirectGui import *

loadPrcFileData('', 'win-size 1280 720')
loadPrcFileData('', 'window-title hackaTUM')


class Globe(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # Load the earth model.
        self.earth = self.loader.loadModel("earth.bam")
        self.earth.setScale(2,2,2)
        self.earth.reparentTo(self.render)

        self.satellite = self.loader.loadModel("satellite.bam")
        self.satellite.setScale(0.02, 0.02, 0.02)
        self.satellite.setPos(0, 0, 0)

        self.satelite_turner = NodePath('satelite_turner')
        self.satelite_rotation = self.satelite_turner.hprInterval(1,Point3(0, 360, 0))
        self.satelite_rotation.loop()

        self.satelite_turner.reparentTo(self.render)
        self.satellite.reparentTo(self.satelite_turner)
        self.belt = self.render.attachNewNode("belt")
        self.belt_holder = self.render.attachNewNode("belt_holder")
        self.np = self.render.attachNewNode("np")

        self.setBackgroundColor(0,0,0)
        alight = AmbientLight('alight')
        # alight.setColor((0.2, 0.2, 0.2, 1))
        alight.setColor((200,200,200, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")



        self.elevation_slider = DirectSlider(range=(-90, 90), pageSize=10, orientation="vertical", pos=Point3(1.5, -0.95, 0), scale=0.8)


        self.altitude_label = DirectLabel(text_bg=(0,0,0,1), text_fg=(1,1,1,1), text = "Orbital Altitude", pos=Point3(-1.4, 0, 0.9), scale=0.09)
        self.altitude_number = DirectLabel(text_bg=(0,0,0,1), text_fg=(1,1,1,1), text = "", pos=Point3(-1.4, 0, 0.7), scale=0.09)

        self.altitude_slider = DirectSlider(range=(0,5), pageSize = 1, pos=Point3(-1.4, 0, 0.85), scale=0.3, command=self.recalculate)

        self.information_count = DirectLabel(text_bg=(0,0,0,1), text_fg=(1,1,1,1), text = "Aha", pos=Point3(-1.35, 0, -0.7), scale=0.09)

        self.rotation_chkbox = DirectCheckButton(text_bg=(0,0,0,1), text_fg=(1,1,1,1), text = "Rotation", pos=Point3(-1.35, 0, -0.55), scale=0.09, indicatorValue = 1, command=self.recalculate_rot)
        

    def recalculate_rot(self,a):
        if self.rotation_chkbox['indicatorValue']:
            self.satelite_rotation.loop(0,1,1/self.satellite_t)
        else:
            self.satelite_rotation.finish()


    def recalculate(self):
        self.altitude = 2**self.altitude_slider['value']

        self.altitude_number['text'] = f"{self.altitude*1000:.0f} km"
        self.altitude_number.resetFrameSize()

        scale = (((self.altitude-1)/31)*0.1)+0.02
        self.satellite.setScale(scale, scale, scale)

        self.calculate_belts(self.altitude*1000)

        self.distance = max(3.5*(self.earth_radius/1000 + self.altitude + 5), 42)
        self.satellite.setPos(0, self.earth_radius/1000 + self.altitude, 0)

        self.information_count['text'] = f"Total Satellites: {self.sector_equator_count * self.sector_belt_count/2:.0f}"

        self.get_orbital_period()
        if self.rotation_chkbox['indicatorValue']:
            self.satelite_rotation.loop(0,1,1/self.satellite_t)
        else:
            self.satelite_rotation.finish()

        self.instantiate_satellites()
        self.instantiate_belts()

        self.draw_sectors()


    def instantiate_satellites(self):
        self.destroy_node(self.belt)
        self.belt = self.render.attachNewNode("belt")
        for i in range(int(self.sector_belt_count/6)):
            for j in range(3):
                instance = self.belt.attachNewNode("satellite")
                instance.setHpr(0,degrees(self.sector_h) * (6*i+j),0)
                self.satelite_turner.instanceTo(instance)

    def instantiate_belts(self):
        self.destroy_node(self.belt_holder)
        self.belt_holder = self.render.attachNewNode("belt_holder")
        for i in range(int(self.sector_equator_count/2)):
            instance = self.belt_holder.attachNewNode("belt")
            instance.setHpr(degrees(self.sector_width)*2*i,degrees(self.sector_h) *0.1*i,0)
            self.belt.instanceTo(instance)

            instance = self.belt_holder.attachNewNode("belt")
            instance.setHpr(degrees(self.sector_width)*(2*i+1),degrees(self.sector_h) *(0.5+0.1*i),0)
            self.belt.instanceTo(instance)




    def get_orbital_period(self):
        #http://hyperphysics.phy-astr.gsu.edu/hbase/orbv3.html --> js
        bigg = 6.67259 * 10**-11
        re = 6.38 * 10**6
        ms = 5.98 * 10**24
        ghg = bigg * ms / ((re + self.altitude * 10**6) ** 2)
        vorb = sqrt(ghg * (re + self.altitude * 10**6))
        self.satellite_t = 2* pi * (re + self.altitude * 10**6) / (vorb*1440)

    def spinCameraTask(self, task):
        if self.rotation_chkbox['indicatorValue']:
            self.angleDegrees = task.time *-6.0
        
        self.angleRadians = self.angleDegrees *(pi/180)
        z_deg = self.elevation_slider['value']
        z_rad = z_deg * (pi/180)
        self.camera.setPos(self.distance*sin(self.angleRadians)*cos(z_rad), -1*self.distance*cos(self.angleRadians)*cos(z_rad), self.distance*sin(z_rad))
        self.camera.setHpr(self.angleDegrees,-1*z_deg,0)

        return Task.cont


    def calculate_belts(self, orbit_height):
        #Default unit is km
        self.earth_radius = 6378

        plane_max_cruising_altitude = 13
        plane_min_angle = radians(10) #rad

        #Compute the radial distance between two reachable extremes in radians
        rad = 2 * (pi/2 - plane_min_angle - asin(sin(pi/2 + plane_min_angle)*(self.earth_radius+plane_max_cruising_altitude)/(self.earth_radius+orbit_height)))

        #Max length (diagonal) of a sector in radians
        sector_max_length = rad / 5

        #Max horizontal spacing between two sectors
        sector_max_width = sector_max_length * 0.75

        #sectors around the equator
        self.sector_equator_count = ceil(pi / (2*sector_max_width)) *2

        #Determine actual sector size
        self.sector_width = pi / self.sector_equator_count
        self.sector_length = self.sector_width * 4/3

        #Diamater of inscribed circle (a.k.a. h)
        sector_max_h = self.sector_length * sqrt(0.75)

        #note the 6* (from 6*pi/3) outside of the ceil(). This guarantees a Number divisible by 6.
        self.sector_belt_count = 6 * ceil(pi / (sector_max_h*3))

        self.sector_h = 2 * pi / self.sector_belt_count

        self.sectorization_efficiency = self.sector_width* self.sector_h/(sector_max_h*sector_max_width )

        # print()

        # print(f"Distinct Belts: {self.sector_equator_count}")
        # print(f"Sector size efficiency: {self.sector_width/sector_max_width}")
        # print(f"Sectors per Belt: {self.sector_belt_count}")
        # print(f"Sector height efficiency: {self.sector_h/sector_max_h}")
        # print(f"Total Satellites: {self.sector_equator_count * self.sector_belt_count/2:.0f}")

    def destroy_node(self, node):
        for m in node.get_children():
            m.remove_node()
        node.remove_node()

    def draw_sectors(self):
        r = self.earth_radius/1000+0.4
        self.d_lines = self.call_me(self.altitude*1000)

        self.destroy_node(self.np)

        lines = LineSegs()
        for d_line in self.d_lines:
            lines.moveTo(self.polar2cart(r, d_line[0][0], d_line[0][1]))
            lines.drawTo(self.polar2cart(r, d_line[1][0], d_line[1][1]))

        node = lines.create()
        self.np = NodePath(node)
        self.np.reparentTo(self.render)

    def polar2cart(self, r, phi, theta):
        return Point3(
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta)
        )

    def call_me(self, orbit_height=20000):
        #Default unit is km
        earth_radius = 6378

        plane_max_cruising_altitude = 13
        plane_min_angle = math.radians(10) #rad


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
        sector_h = sector_length * math.sqrt(0.75)

        #note the 2* (from 2*pi) outside of the ceil(). This guarantees an even Number.
        sector_belt_count = 2 * math.ceil(math.pi / sector_h)

        #Default unit is km
        earth_radius = 6378

        plane_max_cruising_altitude = 13
        plane_min_angle = math.radians(10) #rad



        #Compute the radial distance between two reachable extremes in radians
        rad = 2 * (math.pi/2 - plane_min_angle - math.asin(math.sin(math.pi/2 + plane_min_angle)*(earth_radius+plane_max_cruising_altitude)/(earth_radius+orbit_height)))

        #Max length (diagonal) of a sector in radians
        sector_max_length = rad / 5

        #Max horizontal spacing between two sectors
        sector_max_width = sector_max_length * 0.75

        #sectors around the equator
        sector_equator_count = math.ceil(math.pi / (sector_max_width *2))*2

        #Determine actual sector size 
        sector_width = math.pi / sector_equator_count
        sector_length = sector_width * 4/3

        #Diamater of inscribed circle (a.k.a. h)
        sector_max_h = sector_length * math.sqrt(0.75)

        #note the 6* (from 6*pi/3) outside of the ceil(). This guarantees a Number divisible by 6.
        sector_belt_count =  6*math.ceil(math.pi / (sector_max_h*3))

        sector_h =  math.pi / sector_belt_count


        y_radiant_per_belt = sector_h
        x_radiant_per_belt = 2*sector_width

        def cell(left, right):
            cell_center = np.array([left, right])
            return cell_center

        def cell_corners(cell_center):
            cell_center
            return cell_corners

        # Alpha Cell
        alpha = np.array([0,0])
        beta = np.array([0,0])
        #print("SEC: ", sector_equator_count)
        #number_belt= sector_equator_count/2
        list_position = [[0,0]]
        for i in range(int(sector_equator_count/2.0)):
            beta = np.add(beta, np.array([2*sector_width, 0]))
            list_position= np.append(list_position, [beta] , axis=0)
            alpha = beta

        mean_coordinates = []
        mean_coordinates_dict = defaultdict(lambda:np.empty((0,2),float))
        for y_belt_count in range(int(sector_belt_count)):
                mean_coordinates_dict[f"layer {y_belt_count}"] = np.vstack([[(x_belt_count +(y_belt_count%2)/2) * x_radiant_per_belt,
                                                                            y_belt_count * y_radiant_per_belt]]
                                                                        for x_belt_count in range(int(sector_equator_count/2.0)))



        #remove the overlapping centroids
        def delete_within_belt(y_belt):
            #y_belt is the layer to delete in
            #in order for this to work we need mean_coordinates_dict as a global dict
            within_belt_distances = pairwise_distances(mean_coordinates_dict[f"layer {y_belt}"], mean_coordinates_dict[f"layer {y_belt}"],metric='haversine')
            to_delete=[]
            for row_idx,row in enumerate(within_belt_distances):
                for col, element in enumerate(row):
                    if element < y_radiant_per_belt and row_idx!=col:
                        to_delete.append(col)
            mean_coordinates_dict[f"layer {y_belt}"] = np.delete(mean_coordinates_dict[f"layer {y_belt}"], to_delete, 0)



        for y_belt in range(sector_belt_count*2-1):

            delete_within_belt(y_belt)


            # inside out, thus in neighboring belts
            pw_distances = pairwise_distances(mean_coordinates_dict[f"layer {y_belt}"], mean_coordinates_dict[f"layer {y_belt+1}"],metric='haversine')
            #print(f"layer {y_belt} {pw_distances}")
            to_delete=[]
            for row in pw_distances:
                for col, element in enumerate(row):
                    if element < y_radiant_per_belt:
                        #mean_coordinates_dict[f"layer {y_belt+1}"]= np.delete(mean_coordinates_dict[f"layer {y_belt+1}"], col, 0)
                        to_delete.append(col)
            mean_coordinates_dict[f"layer {y_belt+1}"]= np.delete(mean_coordinates_dict[f"layer {y_belt+1}"], to_delete, 0)#print(point.shape)

            #also delete in the final ball
            y_belt =sector_belt_count*2
            delete_within_belt(y_belt)

        result = mean_coordinates_dict.items()
        data = list(result)
        numpyArray = np.array(data)

        true_data_array = np.vstack([x for x in numpyArray[:,1]])


        earth_array = np.vstack((true_data_array,
                                -true_data_array,
                                np.vstack((true_data_array[:,0],-true_data_array[:,1])).T,
                                np.vstack((-true_data_array[:,0],true_data_array[:,1])).T
        ))





        points = earth_array
        vor = Voronoi(points)


        
        edges2 = []
        #print(vor.ridge_vertices)
        for [v1,v2] in vor.ridge_vertices:
            if v1!=-1 and v2!=-1:
                edges2.append([list(vor.vertices[v1]),list(vor.vertices[v2])])
        return edges2

        
app = Globe()
app.run()