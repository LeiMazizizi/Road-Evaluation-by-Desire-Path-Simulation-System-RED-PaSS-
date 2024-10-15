import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import features,Affine

import numpy as np

from shapely.geometry import Polygon,Point,LineString
import random
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

import copy
import jenkspy
from operator import itemgetter

def ABM_simulation(rasterized_world, road_rasterized, rasterized_world_shape,
                   roadvector,OD_list,OD_flowquantity_list,num_of_agents,cognition_angle,
                   cognition_depth,speed,ticks, Is_OD_given_as_xy=True):
    # RASTERIZED_WORLD = copy.deepcopy(rasterized_world)
    RASTERIZED_ROADS = copy.deepcopy(road_rasterized)
    GEOM_ROADS = copy.deepcopy(roadvector)

    # Construct a dictionary for OD locations and if a OD is in the dictionary, add the flow quantity to the corresponding OD position:
    OD_distribution = {}
    for i in range(len(OD_list)):
        if OD_distribution.get(str(OD_list[i][0][0]) + ',' + str(OD_list[i][0][1]), None) is None:
            OD_distribution[str(OD_list[i][0][0]) + ',' + str(OD_list[i][0][1])] = OD_flowquantity_list[i]
        else:
            OD_distribution[str(OD_list[i][0][0]) + ',' + str(OD_list[i][0][1])] += OD_flowquantity_list[i]

        if OD_distribution.get(str(OD_list[i][1][0]) + ',' + str(OD_list[i][1][1]), None) is None:
            OD_distribution[str(OD_list[i][1][0]) + ',' + str(OD_list[i][1][1])] = OD_flowquantity_list[i]
        else:
            OD_distribution[str(OD_list[i][1][0]) + ',' + str(OD_list[i][1][1])] += OD_flowquantity_list[i]

    OD_location_list=[]
    OD_location_quantity_list=[]
    for key, value in OD_distribution.items():
        OD_location_list.append(eval(key))
        OD_location_quantity_list.append(value)

    OD_location_quantity_list=list(np.array(OD_location_quantity_list)/np.sum(np.array(OD_location_quantity_list)))
    # OD_locations_list=[]
    # for _ in OD_distribution_list:
    #     OD_locations_list.append([eval(_[0]),_[1]])

    np.random.seed(0)
    global current_O_D_list_for_agents_to_pick_pointer,O_D_list_for_agents_to_pick
    O_D_list_for_agents_to_pick_index =np.random.choice([i for i in range(len(OD_location_list))], 10000, p=OD_location_quantity_list)

    O_D_list_for_agents_to_pick = list(itemgetter(*O_D_list_for_agents_to_pick_index)(OD_location_list))

    current_O_D_list_for_agents_to_pick_pointer=0

    # rasterized_world_shape=rasterized_world.shape

    DISTANCE_DIFFERENCE = []
    ANGLE_CHANGE_DIFFERENCE = []

    def OD_destination_geterator():
        global current_O_D_list_for_agents_to_pick_pointer, O_D_list_for_agents_to_pick
        # print(current_O_D_list_for_agents_to_pick_pointer)
        value=O_D_list_for_agents_to_pick[current_O_D_list_for_agents_to_pick_pointer]
        current_O_D_list_for_agents_to_pick_pointer+=1
        if current_O_D_list_for_agents_to_pick_pointer>=len(O_D_list_for_agents_to_pick):
            current_O_D_list_for_agents_to_pick_pointer=0
        return value
    def visionShed_twoTriPoints(originPoint, destinationPoint, cognition_angle, depthofview):

        angle = math.radians(cognition_angle / 2)
        trianglePointA = None
        trianglePointB = None
        if destinationPoint[0] == originPoint[0]:
            if destinationPoint[1] > originPoint[1]:
                trianglePointA = [originPoint[0] + depthofview * math.cos(math.pi / 2.0 - angle),
                                  originPoint[1] + depthofview * math.sin(math.pi / 2.0 + angle)]
                trianglePointB = [originPoint[0] - depthofview * math.cos(math.pi / 2.0 - angle),
                                  originPoint[1] + depthofview * math.sin(math.pi / 2.0 + angle)]
            elif destinationPoint[1] < originPoint[1]:
                trianglePointA = [originPoint[0] + depthofview * math.cos(math.pi / 2.0 - angle),
                                  originPoint[1] - depthofview * math.sin(math.pi / 2.0 + angle)]
                trianglePointB = [originPoint[0] - depthofview * math.cos(math.pi / 2.0 - angle),
                                  originPoint[1] - depthofview * math.sin(math.pi / 2.0 + angle)]

        else:

            beta = math.atan(abs(destinationPoint[1] - originPoint[1]) / abs(destinationPoint[0] - originPoint[0]))

            # right-up
            if destinationPoint[0] > originPoint[0] and destinationPoint[1] >= originPoint[1]:
                trianglePointA = [originPoint[0] + depthofview * math.cos(beta + angle),
                                  originPoint[1] + depthofview * math.sin(beta + angle)]
                trianglePointB = [originPoint[0] + depthofview * math.cos(beta - angle),
                                  originPoint[1] + depthofview * math.sin(beta - angle)]
            # right-down
            elif destinationPoint[0] > originPoint[0] and destinationPoint[1] < originPoint[1]:
                trianglePointA = [originPoint[0] + depthofview * math.cos(beta - angle),
                                  originPoint[1] - depthofview * math.sin(beta - angle)]
                trianglePointB = [originPoint[0] + depthofview * math.cos(beta + angle),
                                  originPoint[1] - depthofview * math.sin(beta + angle)]
            # left-up
            elif destinationPoint[0] < originPoint[0] and destinationPoint[1] >= originPoint[1]:
                trianglePointA = [originPoint[0] - depthofview * math.cos(beta - angle),
                                  originPoint[1] + depthofview * math.sin(beta - angle)]
                trianglePointB = [originPoint[0] - depthofview * math.cos(beta + angle),
                                  originPoint[1] + depthofview * math.sin(beta + angle)]
            # left-down
            elif destinationPoint[0] < originPoint[0] and destinationPoint[1] < originPoint[1]:
                trianglePointA = [originPoint[0] - depthofview * math.cos(beta + angle),
                                  originPoint[1] - depthofview * math.sin(beta + angle)]
                trianglePointB = [originPoint[0] - depthofview * math.cos(beta - angle),
                                  originPoint[1] - depthofview * math.sin(beta - angle)]

        return trianglePointA, trianglePointB

    def endpoint_of_line(originPoint_ij,destinationPoint_ij,move_distance=1):
        import math
        # angle = math.radians(cognition_angle / 2)
        originPoint=ij_to_xy(originPoint_ij,rasterized_world_shape[0],rasterized_world_shape[1])
        destinationPoint=ij_to_xy(destinationPoint_ij,rasterized_world_shape[0],rasterized_world_shape[1])

        p = None
        # trianglePointB = None
        if destinationPoint[0] == originPoint[0]:
            if destinationPoint[1] > originPoint[1]:
                p=[originPoint[0],originPoint[1]+move_distance]
                # trianglePointA = [originPoint[0] + depthofview * math.cos(math.pi / 2.0 - angle),
                #                   originPoint[1] + depthofview * math.sin(math.pi / 2.0 + angle)]
                # trianglePointB = [originPoint[0] - depthofview * math.cos(math.pi / 2.0 - angle),
                #                   originPoint[1] + depthofview * math.sin(math.pi / 2.0 + angle)]
            elif destinationPoint[1] < originPoint[1]:
                p=[originPoint[0],originPoint[1]-move_distance]
                # trianglePointA = [originPoint[0] + depthofview * math.cos(math.pi / 2.0 - angle),
                #                   originPoint[1] - depthofview * math.sin(math.pi / 2.0 + angle)]
                # trianglePointB = [originPoint[0] - depthofview * math.cos(math.pi / 2.0 - angle),
                #                   originPoint[1] - depthofview * math.sin(math.pi / 2.0 + angle)]

        else:

            beta = math.atan(abs(destinationPoint[1] - originPoint[1]) / abs(destinationPoint[0] - originPoint[0]))

            # right-up
            if destinationPoint[0] > originPoint[0] and destinationPoint[1] >= originPoint[1]:


                p = [originPoint[0] + move_distance * math.cos(beta ),
                                  originPoint[1] + move_distance * math.sin(beta )]
                # trianglePointB = [originPoint[0] + depthofview * math.cos(beta - angle),
                #                   originPoint[1] + depthofview * math.sin(beta - angle)]
            # right-down
            elif destinationPoint[0] > originPoint[0] and destinationPoint[1] < originPoint[1]:
                p = [originPoint[0] + move_distance * math.cos(beta ),
                                  originPoint[1] - move_distance * math.sin(beta )]
                # trianglePointB = [originPoint[0] + depthofview * math.cos(beta + angle),
                #                   originPoint[1] - depthofview * math.sin(beta + angle)]
            # left-up
            elif destinationPoint[0] < originPoint[0] and destinationPoint[1] >= originPoint[1]:
                p = [originPoint[0] - move_distance * math.cos(beta ),
                                  originPoint[1] + move_distance * math.sin(beta )]
                # trianglePointB = [originPoint[0] - depthofview * math.cos(beta + angle),
                #                   originPoint[1] + depthofview * math.sin(beta + angle)]
            # left-down
            elif destinationPoint[0] < originPoint[0] and destinationPoint[1] < originPoint[1]:
                p = [originPoint[0] - move_distance * math.cos(beta ),
                                  originPoint[1] - move_distance * math.sin(beta )]
                # trianglePointB = [originPoint[0] - depthofview * math.cos(beta - angle),
                #                   originPoint[1] - depthofview * math.sin(beta - angle)]

        p_ij=xy_to_ij(p,rasterized_world_shape[1],rasterized_world_shape[0])
        return p_ij


    def ij_to_xy(pixel,shape_i=100,shape_j=100):
        y=shape_i-1-pixel[0]
        x=pixel[1]
        return (x,y)
    def xy_to_ij(cor,shape_x=100,shape_y=100):
        i=int(shape_y-1-cor[1])
        j=int(cor[0])
        return (i,j)
    def visionShed_gridPathes(self_location_ij,trianglePointO_xy, trianglePointA_xy, trianglePointB_xy, this_world=rasterized_world):
        # from shapely.geometry import Polygon,Point
        # trianglePointO_xy=ij_to_xy(trianglePointO,rasterized_world_shape[0],rasterized_world_shape[1])
        # trianglePointA_xy=ij_to_xy(trianglePointA,rasterized_world_shape[0],rasterized_world_shape[1])
        # trianglePointB_xy=ij_to_xy(trianglePointB,rasterized_world_shape[0],rasterized_world_shape[1])
        try:
            pp = Polygon([trianglePointO_xy, trianglePointA_xy, trianglePointB_xy])
        except Exception as e:
            print("error: ", str(e))

            ii=0
            pass
        # if pp.geom_type!='Polygon':
        #     print("Polygon is None")
        #     print("trianglePointO_xy, trianglePointA_xy,trianglePointB_xy:",trianglePointO_xy,trianglePointA_xy,trianglePointB_xy)

        evlop= pp.envelope
        x, y = evlop.exterior.xy
        xlist=x.tolist()
        ylist=y.tolist()
        patches=[]
        patch_values=[]

        x_max=int(max(xlist))
        y_max=int(max(ylist))
        x_min=int(min(xlist))
        y_min=int(min(ylist))

        this_world_shape=this_world.shape
        raster_x_max=this_world_shape[1]-1
        raster_y_max=this_world_shape[0]-1
        if x_max>=raster_x_max:
            x_max=raster_x_max
        if y_max >= raster_y_max:
            y_max = raster_y_max
        if x_min<= 0:
            x_min=0
        if y_min <= 0:
            y_min = 0


        for xx in range(x_min,x_max):
            for yy in range(y_min,y_max):
                # if abs(trianglePointO_xy[0] - xx) <=0.001 and abs(trianglePointO_xy[1] - yy)<=0.001: # 排除自己的位置：
                #     print("trianglePointO_xy:",trianglePointO_xy)
                #     print("xx,yy:",xx,yy)
                #
                #     continue

                gridPoint=Point(xx,yy)
                # print("gridPoint:", gridPoint.geom_type)
                if gridPoint.intersects(pp):
                    gridPoint_xy=[xx,yy]
                    gridPoint_ij=xy_to_ij(gridPoint_xy,rasterized_world_shape[1],rasterized_world_shape[0])
                    if self_location_ij[0]==gridPoint_ij[0] and self_location_ij[1]==gridPoint_ij[1]:
                        continue
                    else:
                        patches.append(gridPoint_ij)
                        # print("gridPoint_ij:",gridPoint_ij)
                        try:
                            patch_values.append(rasterized_world[int(gridPoint_ij[0])][int(gridPoint_ij[1])])
                        except Exception as e:
                            print("error: ", str(e))
                            print("gridPoint_ij:", gridPoint_ij)
        return patches,patch_values

    def patch_on_the_lineOFOriginToPatchMax(patch_max,originPoint,movingSpeed=1):
        originPoint_xy=ij_to_xy(originPoint,rasterized_world_shape[0],rasterized_world_shape[1])
        patch_max_xy=ij_to_xy(patch_max,rasterized_world_shape[0],rasterized_world_shape[1])



        line=LineString([originPoint_xy,patch_max_xy])
        neighbour_patches_xy=[]
        closest_distance=float("inf")
        for x in range(originPoint_xy[0]-1,originPoint_xy[0]+2):
            for y in range(originPoint_xy[1]-1,originPoint_xy[1]+2):
                if x==originPoint_xy[0] and y==originPoint_xy[1]:
                    continue
                this_distance=Point(x,y).distance(line)
                if this_distance<closest_distance:
                    closest_distance=this_distance
                    neighbour_patches_xy=[x,y]

        neighbour_patches=xy_to_ij(neighbour_patches_xy,rasterized_world_shape[1],rasterized_world_shape[0])
        return neighbour_patches

    def angle_of_two_vectors(vector1,vector2):
        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            if np.linalg.norm(vector)==0:
                print(vector)
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::
                    angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
                    angle_between((1, 0, 0), (1, 0, 0))
                    0.0
                    angle_between((1, 0, 0), (-1, 0, 0))
                    3.141592653589793
            """
        v1_u = unit_vector(vector1)
        v2_u = unit_vector(vector2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    def angle_of_two_vectors_ij(startPoint_ij,endPoint_ij_1,endPoint_ij_2):
        startPoint_xy=ij_to_xy(startPoint_ij,rasterized_world_shape[0],rasterized_world_shape[1])
        endPoint_xy_1=ij_to_xy(endPoint_ij_1,rasterized_world_shape[0],rasterized_world_shape[1])
        endPoint_xy_2= ij_to_xy(endPoint_ij_2, rasterized_world_shape[0], rasterized_world_shape[1])

        angle1=angle_of_two_vectors([endPoint_xy_1[0]-startPoint_xy[0],endPoint_xy_1[1]-startPoint_xy[1]],[endPoint_xy_2[0]-startPoint_xy[0],endPoint_xy_2[1]-startPoint_xy[1]])
        return angle1

    # Python 3 program for Bresenham’s Line Generation
    # Assumptions :
    # 1) Line is drawn from left to right.
    # 2) x1 < x2 and y1 < y2
    # 3) Slope of the line is between 0 and 1.
    # We draw a line from lower left to upper
    # right.

    # function for line generation
    def bresenham_meself(x1,y1,x2,y2):
        dx=abs(x2-x1)
        dy=abs(y2-y1)
        cells = [(x1, y1)]
        if dx>dy:
            m=dy/dx

            alpa = m - 1

            for i in range(dx-1):

                if alpa > 0 or abs(alpa)<0.00001:
                    cells.append((cells[-1][0]+1,cells[-1][1]+1))
                    alpa -= 1
                else:
                    cells.append((cells[-1][0]+1,cells[-1][1]))
                alpa += m
        else:
            m=dx/dy
            alpa = m - 1

            for i in range(dy-1):

                if alpa >= 0 or abs(alpa)<0.00001:
                    cells.append((cells[-1][0]+1,cells[-1][1]+1))
                    alpa -= 1
                else:
                    cells.append((cells[-1][0],cells[-1][1]+1))
                alpa += m
        return cells
    def move_from_start_to_aPatch(startPoint_ij, endPoint_ij, shape_i,shape_j):
        startPoint_xy=ij_to_xy(startPoint_ij,shape_i,shape_j)
        endPoint_xy=ij_to_xy(endPoint_ij,shape_i,shape_j)

        cells=bresenham_meself(startPoint_xy[0],startPoint_xy[1],endPoint_xy[0],endPoint_xy[1])
        cells_ij=[]
        for _ in cells:
            cells_ij_=xy_to_ij(_,shape_j,shape_i)
            cells_ij.append(cells_ij_)

        if len(cells_ij)==0:
            m=0
            pass
        return cells_ij[1]

    def one_step_forward_basedon_vector_meself(x1,y1,x2,y2):

        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        unitv=unit_vector(np.array([x2-x1,y2-y1]))
        signx=1 if unitv[0]>=0 else -1
        signy=1 if unitv[1]>=0 else -1

        unitv=(np.absolute(unitv)+np.array([0.5,0.5])).astype(int).tolist()
        new_unitv=[x1+unitv[0]*signx,y1+unitv[1]*signy]
        return new_unitv


    class Agent:
        def __init__(self, id,oi, oj, di, dj,cognition_angle=90, cognition_depth=6,speed=1):
            self.id=id
            self.i = oi
            self.j = oj
            self.origin_i = oi
            self.origin_j = oj
            self.destination_i = di
            self.destination_j = dj
            self.route = [(self.i, self.j)]
            self.change_angles = []
            self.cognition_angle = cognition_angle
            self.cognition_depth = cognition_depth
            if speed>=cognition_depth:
                self.speed=cognition_depth-1
            else:
                self.speed=speed



        def move(self):
            # write a function:
            # This agent moves one step towards its destination and searching for the cells within its view shed with a cognition angle, alpha, and a depth of view, d.
            # If the agent finds a cell with a higher value than the current cell, it moves to that cell. Otherwise, it moves to a random cell within its view shed.
            # If the agent reaches its destination, it stops.

            # If the agent reaches its destination, it stops and clear its route and do something else.
            if abs(self.i - self.destination_i) + abs(self.j - self.destination_j) <= 1:
                # global ANGLE_CHANGE_DIFFERENCE,DISTANCE_DIFFERENCE
                self.i = self.destination_i
                self.j = self.destination_j

                shape_rasterized_world_shape_i = rasterized_world_shape[0]
                shape_rasterized_world_shape_j = rasterized_world_shape[1]
                if self.i >= shape_rasterized_world_shape_i and self.j >= shape_rasterized_world_shape_j:
                    rasterized_world[shape_rasterized_world_shape_i - 1][shape_rasterized_world_shape_j - 1] += 1
                elif self.i >= shape_rasterized_world_shape_i:
                    rasterized_world[shape_rasterized_world_shape_i - 1][self.j] += 1
                elif self.j >= shape_rasterized_world_shape_j:
                    rasterized_world[self.i][shape_rasterized_world_shape_j - 1] += 1
                else:
                    rasterized_world[self.i][self.j] += 1

                self.route.append((self.i, self.j))
                # print("Location i: %3d, j : %2d; di : %3d, dj : %2d" % (
                # self.i, self.j, self.destination_i, self.destination_j))

                ANGLE_CHANGE_DIFFERENCE.append(np.average(np.array(self.change_angles)))
                distance_route = 0

                for ii in range(len(self.route)):
                    if ii == len(self.route) - 1:
                        break
                    if abs(self.route[ii][0] - self.route[ii + 1][0]) + abs(
                            self.route[ii][1] - self.route[ii + 1][1]) == 2:
                        distance_route += 1.414
                    else:
                        distance_route += 1

                distance_all = np.sqrt(
                    (self.destination_i - self.origin_i) ** 2 + (self.destination_j - self.origin_j) ** 2)
                DISTANCE_DIFFERENCE.append(abs(distance_all - distance_route))

                # Reset:
                self.route = []
                self.change_angles = []
                self.origin_i = self.destination_i
                self.origin_j = self.destination_j

                while 1:
                    # np.random.seed(0)
                    # chosenOD = np.random.choice(OD_list,1, p=OD_flowquantity_list)[0]
                    # chosenOD = np.random.choice(OD_location_list,1, p=OD_location_quantity_list)[0]
                    # global current_O_D_list_for_agents_to_pick_pointer, O_D_list_for_agents_to_pick
                    chosenOD_D = OD_destination_geterator()

                    if chosenOD_D[0] == self.origin_i and chosenOD_D[1] == self.origin_j:
                        continue
                    else:
                        self.destination_i = chosenOD_D[0]
                        self.destination_j = chosenOD_D[1]
                        break

                # self.destination_i=chosenOD[1][0]
                # self.destination_j=chosenOD[1][1]
            else:
                self_position_xy=ij_to_xy([self.i,self.j],rasterized_world_shape[0],rasterized_world_shape[1])
                destination_xy=ij_to_xy([self.destination_i,self.destination_j],rasterized_world_shape[0],rasterized_world_shape[1])
                trianglePointA_xy,trianglePointB_xy=visionShed_twoTriPoints(self_position_xy, destination_xy, self.cognition_angle, self.cognition_depth)
                # if trianglePointA == None or trianglePointB== None:
                #     print("Wrong Inof --- Location i: %3d, j : %2d; di : %3d, dj : %2d" % (
                #         self.i, self.j, self.destination_i, self.destination_j))

                patches,patches_values=visionShed_gridPathes([self.i,self.j],self_position_xy,trianglePointA_xy,trianglePointB_xy,rasterized_world)

                # Pick a patch with the highest value:
                if len(patches_values)==0:
                    mm=0
                    pass
                patch_maxValue=max(patches_values)
                max_index=[
                    i for i, j in enumerate(patches_values) if j == patch_maxValue
                ]
                patch_max=[]
                if len(max_index)==1:
                    patch_max=patches[max_index[0]]
                else:
                    for _ in range(10):
                        # np.random.seed(0)
                        # patch_max=patches[np.random.choice([i for i in range(len(max_index))],size=1)[0]]
                        patch_max=patches[max_index[1]]

                        # if len(self.route)>0:
                        #     if patch_max[0]==self.route[-1][0] and patch_max[1]==self.route[-1][1]:
                        #         continue
                        if patch_max[0]==self.i and patch_max[1]==self.j:
                            continue
                        else:
                            break


                    # patch_max=patches[random.choice(max_index)]

                # print("Location i: %3d, j : %2d; di : %3d, dj : %2d \n" % (
                # self.i, self.j, self.destination_i, self.destination_j))
                # print("patch_max:",patch_max)
                if patch_max[0]==self.i and patch_max[1]==self.j:
                    # print("trianglePointA: ",trianglePointA_xy)
                    # print("trianglePointB: ",trianglePointB_xy)
                    # print("patches: ",patches)
                    print("self.i, self.j: ",self.i, self.j)
                    print("patch_max: ",patch_max)
                # next_patch=patch_on_the_lineOFOriginToPatchMax(patch_max,[self.i,self.j],movingSpeed=self.speed)
                # cells_ij=move_from_start_to_aPatch([self.i,self.j],patch_max,rasterized_world_shape[0],rasterized_world_shape[1])
                next_patch=one_step_forward_basedon_vector_meself(self.i,self.j,patch_max[0],patch_max[1])
                startPoint_ij=copy.deepcopy([self.i,self.j])

                shape_rasterized_world_shape_i=rasterized_world_shape[0]
                shape_rasterized_world_shape_j=rasterized_world_shape[1]
                if self.i>=shape_rasterized_world_shape_i and self.j>=shape_rasterized_world_shape_j:
                    rasterized_world[shape_rasterized_world_shape_i-1][shape_rasterized_world_shape_j-1] += 1
                elif self.i>=shape_rasterized_world_shape_i:
                    rasterized_world[shape_rasterized_world_shape_i-1][self.j] += 1
                elif self.j>=shape_rasterized_world_shape_j:
                    rasterized_world[self.i][shape_rasterized_world_shape_j-1] += 1
                else:
                    rasterized_world[self.i][self.j] += 1

                self.i = next_patch[0]
                self.j = next_patch[1]
                self.route.append((self.i,self.j))
                # AFFORDANCE_MAP[self.i][self.j] -= 10



                # print("route: ",self.route)

                #
                # for i in range(self.speed):
                #     if cells_ij[i][0]==self.i and cells_ij[i][1]==self.j:
                #         continue
                #     self.i=cells_ij[i][0]
                #     self.j=cells_ij[i][1]
                #     self.route.append(cells_ij[i])
                #     AFFORDANCE_MAP[self.i][self.j]-=50
                #     # Visualize the moving process:



                # next_moveTo_patch=endpoint_of_line([self.origin_i,self.origin_j],next_patch,self.speed)

                angle_change = angle_of_two_vectors_ij(startPoint_ij, (self.destination_i, self.destination_j),
                                                       (self.i, self.j))
                self.change_angles.append(angle_change)

                # Move to the next patch:
                # self.i=next_moveTo_patch[0]
                # self.j=next_moveTo_patch[1]
                # self.route.append(next_moveTo_patch)


                pass

    # # Build up a r-tree for the rasterized world:
    # idx = index.Index()
    # for i, cell in enumerate(rasterized_world):
    #     idx.insert(i, cell.bounds)


    def world_visualization(rasterized_world,rasterized_world_shape,fig):
        global POSITION_DATA
        position_data_df=pd.DataFrame(POSITION_DATA)
        # Create figure
        # rasterized_world=np.array(rasterized_world,dtype=np.uint8)

        # img = PIL_Image.fromarray(rasterized_world)


        # fig = go.Figure(data=img)
        from PIL import Image
        # AFFORDANCE_MAP = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        fig.add_image(z=np.stack((np.flip(AFFORDANCE_MAP, 0),) * 3, -1), opacity=0.4)
        # fig.add_layout_image(dict(source=Image.fromarray(np.stack((np.flip(AFFORDANCE_MAP,0),) * 3, -1)),opacity=0.8,layer="above"))


        for count, i in enumerate(position_data_df['i']):
            p_i=ij_to_xy([i,position_data_df['j'].iloc[count]],rasterized_world_shape[0],rasterized_world_shape[1])

            fig.add_layout_image(
                dict(
                    source=PERSON_ICON,
                    xref='x',
                    yref='y',
                    x=p_i[0],
                    y=p_i[1],
                    sizex=2,
                    sizey=2,
                    # sizing="stretch",
                    opacity=0.9,
                    layer="above"
                    # "below"
                )

            )

        # fig.update_layout(xaxis_range=[-0.5, rasterized_world_shape[1]-1], yaxis_range=[-0.5,rasterized_world_shape[0]-1],
        #                   overwrite=True)



    agents=[]

    # app = dash.Dash(__name__)
    #
    # app.layout = html.Div([
    #     dcc.Graph(id='real-time-plot'),
    #     dcc.Interval(
    #         id='interval-component',
    #         interval=100,  # Update every 1000 milliseconds (1 second)
    #         n_intervals=ticks
    #     )
    # ])

    # Run the simulation:
    # Initialize the world:
    # Create agents and initilize their origin and destination:
    # position_data = {
    #     'i': [],
    #     'j': [],
    #     # 'image':[]
    # }
    # global current_O_D_list_for_agents_to_pick_pointer, O_D_list_for_agents_to_pick
    for i in range(num_of_agents):
        # np.random.seed(0)
        # chosenOD_O = np.random.choice(OD_list, 1, p=OD_flowquantity_list)[0]
        # chosenOD = random.choices(OD_list, weights=OD_flowquantity_lisst, k=1)[0]
        # print(current_O_D_list_for_agents_to_pick_pointer)
        chosenOD_O = OD_destination_geterator()
        # print(current_O_D_list_for_agents_to_pick_pointer)
        chosenOD_D = OD_destination_geterator()
        # print(current_O_D_list_for_agents_to_pick_pointer)


        # chosenOD = random.choices(OD_list, weights=OD_flowquantity_list, k=1)[0]
        # print("Agent %d: Origin: (%d,%d), Destination: (%d,%d)" % (i, chosenOD[0][0], chosenOD[0][1], chosenOD[1][0], chosenOD[1][1]))
        agents.append(Agent(i, chosenOD_O[0], chosenOD_O[1], chosenOD_D[0], chosenOD_D[1],cognition_angle=cognition_angle, cognition_depth=cognition_depth,speed=speed))
        # POSITION_DATA['i'].append(chosenOD[0][0])
        # POSITION_DATA['j'].append(chosenOD[0][1])

        # position_data['image'].append(PERSON_ICON)
    # fig = go.Figure(data=go.Image(z=np.stack((np.flip(rasterized_world, 0),) * 3, -1)))
    # world_visualization(position_data,rasterized_world,rasterized_world_shape,fig)
    # fig.show()

    # @app.callback(
    #     dash.dependencies.Output('real-time-plot', 'figure'),
    #     [dash.dependencies.Input('interval-component', 'n_intervals')]
    # )
    def update_real_time_plot(n_intervals):
        # Generate some random data (replace this with your own data source)
        # x_data = [1, 2, 3, 4, 5]
        # y_data = [random.randint(0, 100) for _ in x_data]
        global POSITION_DATA
        fig = go.Figure(data=go.Image(z=np.stack((np.flip(rasterized_world, 0),) * 3, -1)))

        world_visualization(rasterized_world, rasterized_world_shape, fig)

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', name='Data'))
        # fig.update_layout()



        POSITION_DATA= {
            'i': [],
            'j': [],
            # 'image':[]
        }
        # One agent's moving:
        for agent in agents:
            agent.move()
            # Update the position data:
            POSITION_DATA['i'].append(agent.i)
            POSITION_DATA['j'].append(agent.j)

        fig.update_layout(title='Real-time Plot', xaxis_title='X-Axis', yaxis_title='Y-Axis',
                          xaxis_range=[-0.5, rasterized_world_shape[1] - 1],
                          yaxis_range=[-0.5, rasterized_world_shape[0] - 1],
                          )
        return fig

    # app.run_server(debug=True)

    for tick in range(ticks):
        # print("tick: ",tick)
        for agent in agents:
            agent.move()

    RASTERIZED_WORLD_emerged_roads=np.where(rasterized_world>=np.mean(rasterized_world),rasterized_world,0)
    RASTERIZED_WORLD_emerged_roads=RASTERIZED_WORLD_emerged_roads[RASTERIZED_WORLD_emerged_roads != 0]
    #Simulate the RASTERIZED_WORLD:
    breaks=jenkspy.jenks_breaks(RASTERIZED_WORLD_emerged_roads.flatten(),n_classes=4)

    RASTERIZED_WORLD_breaks=[]
    for i in range(len(breaks)-1):
        # if i==0:
        #     continue
        # RASTERIZED_WORLD_breaks.append(np.where(np.logical_and(RASTERIZED_WORLD>breaks[i],RASTERIZED_WORLD<=breaks[i+1]),1,0))
        RASTERIZED_WORLD_breaks.append(
        np.where(rasterized_world > breaks[i], 1, 0))



    return RASTERIZED_WORLD_breaks,rasterized_world

def visulization_analysis_results(RASTERIZED_WORLD_breaks,RASTERIZED_ROADS,GEOM_ROADS,display_colors):
    def Analysis_of_roadNetworks(path_list):
        # global RASTERIZED_ROADS
        results=[]
        for i in range(len(path_list)):
            overlaied=np.logical_and(path_list[i], RASTERIZED_ROADS)
            # results.append()
            roadcells_on_path = (overlaied == True).sum()
            paths_all=(path_list[i]==True).sum()
            results.append(round(roadcells_on_path/paths_all,2))
        return results

    def Spatial_analysis_of_roadNetworks(path_list_original,color_num):

        # global GEOM_ROADS,RASTERIZED_ROADS
        from rasterio import Affine

        # Filp the RASTERIZED_ROADS_breaks in path_list:
        path_list=[]
        for i in range(len(path_list_original)):
            path_list.append(np.flip(path_list_original[i],0))


        # gpd.GeoDataFrame(geometry=GEOM_ROADS)
        d={}
        for i in range(len(path_list)):
            d['path_'+str(i+1)]=[]
        d={'geometry':GEOM_ROADS} # 4 paths

        def split_categoried(list,color_num):
            def natural_breaks_and_split(l, colors):
                breaks = jenkspy.jenks_breaks(l, n_classes=colors)

                split_results = []
                for _ in l:
                    for i in range(len(breaks) - 1):

                        if _ >= breaks[i] and _ < breaks[i + 1]:
                            split_results.append(i)
                            break
                        if _ == breaks[-1]:
                            split_results.append(len(breaks) - 2)
                            break
                return split_results
            return natural_breaks_and_split(list, colors=color_num)

        for i in range(len(path_list)):
            values_h=[]
            for geo_road in GEOM_ROADS:


                mask = rasterio.features.geometry_mask([geo_road],
                                                       out_shape=RASTERIZED_ROADS.shape,
                                                       all_touched=True,
                                                       invert=True,
                                                       transform=Affine(1,0,0,0,1,0,0,0,1),
                                                       )

                values_along_line = np.sum(path_list[i][mask])
                values_h.append(values_along_line)
            values_h_none_duplicate = list(dict.fromkeys(values_h))
            if len(values_h_none_duplicate)<color_num:

                values_h_split_categoried=[ 0 for _ in range(len(values_h))]
            else:
                values_h_split_categoried=split_categoried(values_h,color_num)

            d['path_'+str(len(path_list)-i)]=values_h_split_categoried
            pass


        def plot_using_plotly(d_lib,path_list_original):
            fig=make_subplots(rows=2,cols=2,subplot_titles=("Level-4 Paths","Level-3 Paths","Level-2 Paths","Level-1 Paths"))
            # d_h=gpd.GeoDataFrame(d_lib)


            def settingup_trace_list(gdfh,prop_name,color_num):
                colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
                if color_num==5:
                    colors = ['#0000FF', '#00FF00', '00F0E5', '#FFFF00', '#FF0000']
                if color_num==3:
                    colors = ['#0000FF', '#00FF00', '#FF0000']
                tracssss=[]
                for index, row in gdfh.iterrows():
                    _ = row['geometry']
                    prop=row[prop_name]

                    x = _.coords.xy[0].tolist()
                    y = _.coords.xy[1].tolist()
                    trace = go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        line=dict(
                            width=2,
                            # colorscale="darkcyan",
                            # cmin=min(line_widths),
                            # cmax=max(line_widths),
                            color=colors[prop]  # Use line width for color mapping
                            # showscale=True  # Show color scale bar
                        )
                        # name=f'Line {i + 1}'
                    )
                    tracssss.append(trace)
                return tracssss


            for __ in range(len(path_list_original)):
                rowcol=[1,1]
                if __ ==0:
                    rowcol=[1,1]
                elif __ ==1:
                    rowcol=[1,2]
                elif __ ==2:
                    rowcol=[2,1]
                elif __ ==3:
                    rowcol=[2,2]
                flipped_array=np.flip(path_list_original[__],0)
                fig.add_trace(go.Heatmap(z=flipped_array.tolist(),colorscale='Greys'),row=rowcol[0],col=rowcol[1])
                libraryh={'geometry':d_lib['geometry'],'prop':d_lib['path_'+str(len(path_list_original)-__)]}
                tracess = settingup_trace_list(gpd.GeoDataFrame(libraryh), 'prop', color_num=color_num)
                fig.add_traces(tracess, rows=rowcol[0], cols=rowcol[1])

            fig.update_layout(title_text="Paths of different levels",width=800,height=900)
            # fig.show()
            return fig

        fig_local=plot_using_plotly(d,path_list_original)
        return fig_local
        pass

    def plot_using_plotly(RASTERIZED_WORLD_breaks):
        fig=make_subplots(rows=2,cols=2,subplot_titles=("Level-4 Paths","Level-3 Paths","Level-2 Paths","Level-1 Paths"))

        flipped_rasterized1=np.flip(RASTERIZED_WORLD_breaks[0],0)
        flipped_rasterized2=np.flip(RASTERIZED_WORLD_breaks[1],0)
        flipped_rasterized3=np.flip(RASTERIZED_WORLD_breaks[2],0)
        flipped_rasterized4=np.flip(RASTERIZED_WORLD_breaks[3],0)

        fig.add_trace(go.Heatmap(z=flipped_rasterized1.tolist(),colorscale='Greys'),row=1,col=1)
        fig.add_trace(go.Heatmap(z=flipped_rasterized2.tolist(),colorscale='Greys'),row=1,col=2)
        fig.add_trace(go.Heatmap(z=flipped_rasterized3.tolist(),colorscale='Greys'),row=2,col=1)
        fig.add_trace(go.Heatmap(z=flipped_rasterized4.tolist(),colorscale='Greys'),row=2,col=2)
        fig.update_layout(title_text="Emerged Paths of Different Levels",width=700,height=800)
        # fig.show()
        return fig

    fig_path_of_breaks=plot_using_plotly(RASTERIZED_WORLD_breaks)


    roads_on_paths_ratios=Analysis_of_roadNetworks(RASTERIZED_WORLD_breaks)
    # print("roads_on_paths_ratios: ",roads_on_paths_ratios)

    fig_spatial_overlay_with_paths=Spatial_analysis_of_roadNetworks(RASTERIZED_WORLD_breaks,display_colors)

    return fig_path_of_breaks,fig_spatial_overlay_with_paths,roads_on_paths_ratios