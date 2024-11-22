import copy
import pygame
import MatrixLibrary as MLib
import linearalgebra as la
import random
import math
import numpy as np
from abc import ABC, abstractmethod
import sympy as sp
import time
#Constants
#screen size
WIDTH = 1600
HEIGHT = 800
#translation for 2d camera
X_TRANSLATE = 0
Y_TRANSLATE = 0
#enum
COLORS = {
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "WHITE": (255, 255, 255),
    "GAINSBORO": (220, 220, 220),
    "LIGHT_GRAY": (211, 211, 211),
    "SILVER": (192, 192, 192),
    "DARK_GRAY": (169, 169, 169),
    "GRAY": (128, 128, 128),
    "DIM_GRAY": (105, 105, 105),
    "LIGHT_SLATE_GRAY": (119, 136, 153),
    "SLATE_GRAY": (112, 128, 144),
    "DARK_SLATE_GRAY": (47, 79, 79)
}
class Coordinate3DSystem:
    def __init__(self, vector_valued_function = [sp.cos("t"), sp.sin("t"), "t"]):
        self.origin = np.array([[0], [0], [0]]) #O
        self.axises = [np.array([ [1], [0], [0]]), #X-axis
                      np.array([ [0],[1], [0]  ]), #Y-axis
                      np.array([  [0], [0], [1]  ])] #Z-axis
        #call transtaltion matrix to the center of screen
    def draw(self, screen)-> None:
        #Step1: get temporary copy of the axises points and origin to prevent changing the original values
        axises = copy.deepcopy(self.axises)
        origin = copy.deepcopy(self.origin)
        #Scale axises for better visualization
        for i in range(3):
            axises[i] = 2*axises[i]
        #Step2: each point undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system
        #Step2: if projecting on yz plane, x-coordinate is automatically 0, therefore no need to muliply projection matrix
        #which means getting [1] and [2] index of the projection vector
        for i in range(3):
            #parameter for pygame.draw.line is (screen, color, start_pos, end_pos, width)
            #parameter for pygame.draw.circle is (screen, color, center, radius, width)
            pygame.draw.circle(screen, COLORS["RED"], to_pygame( (origin[1][0], origin[2][0]) ,HEIGHT), 10) #draw the origin
            #display character O
            font = pygame.font.Font(None, 36)
            text = font.render("Sun", True, COLORS["RED"])
            screen.blit(text, to_pygame((origin[1][0], origin[2][0]), HEIGHT))
            #draw axis individually x -> red, y -> green, z -> blue
            if i == 0:
                pygame.draw.line(screen, COLORS["RED"], to_pygame((origin[1][0], origin[2][0]), HEIGHT), to_pygame( (axises[i][1][0], axises[i][2][0]), HEIGHT ), 3) #draw x-axis
                #display character x
                text = font.render("X", True, COLORS["RED"])
                screen.blit(text, to_pygame((axises[i][1][0], axises[i][2][0]), HEIGHT))
            elif i == 1:
                pygame.draw.line(screen, COLORS["GREEN"], to_pygame((origin[1][0], origin[2][0]), HEIGHT), to_pygame( (axises[i][1][0], axises[i][2][0]), HEIGHT ), 3) #draw y-axis
                #display character y
                text = font.render("Y", True, COLORS["GREEN"])
                screen.blit(text, to_pygame((axises[i][1][0], axises[i][2][0]), HEIGHT))
            else:
                pygame.draw.line(screen, COLORS["BLUE"], to_pygame((origin[1][0], origin[2][0]), HEIGHT), to_pygame( (axises[i][1][0], axises[i][2][0]), HEIGHT ), 3) #draw z-axis
                #display character z
                text = font.render("Z", True, COLORS["BLUE"])
                screen.blit(text, to_pygame((axises[i][1][0], axises[i][2][0]), HEIGHT))
                
    def rotate(self, alpha: float | int =0, beta: float | int =0, gamma: float | int=0)-> None:
        axises = self.axises
        origin = self.origin
        #rotation matrix around the (self-referenced) axises and not the axises (global) that is used for projection
        Rx = MLib.MatrixLibrary().Rotation_matrix(alpha, art_axis= axises[0])
        Ry = MLib.MatrixLibrary().Rotation_matrix(beta, art_axis= axises[1])
        Rz = MLib.MatrixLibrary().Rotation_matrix(gamma, art_axis= axises[2])
        #apply rotation matrix to the self-referenced axises
        for i in range(3):
            axises[i] = np.dot(Rx, axises[i])
            axises[i] = np.dot(Ry, axises[i])
            axises[i] = np.dot(Rz, axises[i])
    def scale(self, scalar: int|float):
        for i in range(3):
            self.axises[i] = scalar*self.axises[i]
class VectorValuedFunction:
    def __init__(self, symbolic_expr: list[sp.Expr], reference_frame: Coordinate3DSystem):
        self.reference_frame = reference_frame
        self.pos = sp.lambdify("t", [symbolic_expr[0], symbolic_expr[1], symbolic_expr[2]])
        ############################################################################################################
        velocity = VectorValuedFunction._diff(symbolic_expr)
        self.velocity = sp.lambdify("t", [velocity[0], velocity[1], velocity[2]])
        ############################################################################################################
        acceleration = VectorValuedFunction._diff(velocity)
        self.acceleration = sp.lambdify("t", [acceleration[0], acceleration[1], acceleration[2]])
        ############################################################################################################
        TNB_vector = VectorValuedFunction.TNB_vectors(symbolic_expr)
        self.T = sp.lambdify("t", [TNB_vector[0][0], TNB_vector[0][1], TNB_vector[0][2]])
        self.N = sp.lambdify("t", [TNB_vector[1][0], TNB_vector[1][1], TNB_vector[1][2]])
        self.B = sp.lambdify("t", [TNB_vector[2][0], TNB_vector[2][1], TNB_vector[2][2]])
        ############################################################################################################
        tangential_acceleration = VectorValuedFunction._dot(velocity, acceleration) / VectorValuedFunction._norm(velocity) # scalar along T
        self.tangential_acceleration = sp.lambdify("t", [tangential_acceleration*TNB_vector[0][0], tangential_acceleration*TNB_vector[0][1], tangential_acceleration*TNB_vector[0][2]])
        ##############################################
        normal_acceleration = sp.sqrt(VectorValuedFunction._norm(acceleration)**2-tangential_acceleration**2)# scalar along N
        self.normal_acceleration = sp.lambdify("t", [normal_acceleration*TNB_vector[1][0], normal_acceleration*TNB_vector[1][1], normal_acceleration*TNB_vector[1][2]])
        curvature = VectorValuedFunction.curvature(symbolic_expr)
        self.curvature = sp.lambdify("t", curvature)
        ############################################################################################################
        self.xyz_infor = np.array([[[self.pos(0)[0]],[self.pos(0)[1]],[self.pos(0)[2]]]])  # np.array( [ 
                             #             [ [x0],[y0],[z0] ], 
                             #             [ [x1],[y1],[z1] ],
                             #                  ...,
                             #             [ [xn],[yn],[zn] ] 
                             #                                ]) that store information about the position of the object as time goes on
  
    @staticmethod
    def TNB_vectors(vvf: list[sp.Expr])-> list[list[sp.Expr]]:
        T = VectorValuedFunction._unit_tangent_v(vvf)
        N = VectorValuedFunction._unit_normal_v(vvf)
        B = VectorValuedFunction._unit_binormal_v(vvf)
        return [T, N, B]

    @staticmethod
    def curvature(vvf: list[sp.Expr]):
        puv1 = VectorValuedFunction._unit_tangent_v(vvf)
        norm_puv1_1st_diff = VectorValuedFunction._norm(VectorValuedFunction._diff(puv1))
        norm_r_1st_diff = VectorValuedFunction._norm(VectorValuedFunction._diff(vvf))
        return norm_puv1_1st_diff/norm_r_1st_diff

    @staticmethod
    def r_of_c_of_vvf(vvf: list[sp.Expr])-> sp.Expr:
        return VectorValuedFunction._diff(vvf)

    @staticmethod
    def r_of_c_of_norm_of_vvf(vvf: list[sp.Expr])-> sp.Expr:
        return VectorValuedFunction._norm(vvf)

    @staticmethod
    def norm_of_r_of_c_of_vvf(vvf: list[sp.Expr])-> sp.Expr:
        return VectorValuedFunction._norm(VectorValuedFunction._diff(vvf))

    @staticmethod
    def indef_integral_of_vvf(vvf: list[sp.Expr])-> list[sp.Expr]:
        return VectorValuedFunction._integrate(vvf)

    @staticmethod
    def indef_integral_of_norm_of_vvf(vvf: list[sp.Expr])-> sp.Expr:
        return VectorValuedFunction._integrate(VectorValuedFunction._norm(vvf))

    @staticmethod
    def norm_of_indef_integral_of_vvf(vvf: list[sp.Expr])-> sp.Expr:
        return VectorValuedFunction._norm(VectorValuedFunction._integrate(vvf))
    #INSTANCE METHODS
    def draw(self, screen: pygame.Surface, time: time = 0)-> None:
        self._draw_trajectory(screen, self.reference_frame, time)
        self._draw_TNB_vectors(screen, self.reference_frame, time)
        self._draw_velocity(screen, self.reference_frame, time)
        self._draw_acceleration(screen, self.reference_frame, time)
        self._draw_tangential_acceleration(screen, self.reference_frame, time)
        self._draw_normal_acceleration(screen, self.reference_frame, time)

    '''
    def translate(self, dx: int | float, dy: int | float, dz: int | float): # do not use this method for now 
     translation_matrix = MLib.MatrixLibrary().Translation_matrix(dx, dy, dz)
    
     for i in range(len(self.xyz_infor)):
        # Convert to homogeneous coordinates
        print("before:", self.xyz_infor[i])
        homogeneous_coord = np.vstack([self.xyz_infor[i], [[1]]])  # Now shape (4, 1)
        print("after converting to homogeneous:", homogeneous_coord)
        
        # Apply the translation matrix
        transformed = np.dot(translation_matrix, homogeneous_coord)  # Still (4, 1)
        
        # Convert back to 3D coordinates by dropping the last row
        self.xyz_infor[i] = transformed[:3]  # Keep only the (3, 1) part
        print("after translation:", self.xyz_infor[i])
    '''

    #HELPER FUNCTIONS
    def _draw_trajectory(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time = 0) -> None:
        #step1: update new position at time t and add to xyz_infor 
        new_pos = self.pos(time)
        self.xyz_infor= np.append( self.xyz_infor ,[ [[new_pos[0]], [new_pos[1]], [new_pos[2]] ]], axis=0)
        #Step2: each point undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system
        A = np.hstack(coordinate_system.axises)#Linear-Tranformation matrix A
        points = self.xyz_infor.reshape(-1, 3, 1)  # Each point is a 3x1 column vector
        transformed_points = np.matmul(A, points)
        #Step3: no need for projecting to yz plane, just skip x coordinate 
        #Step4: draw the line connecting the points
        for i in range(len(transformed_points)-1):
            pygame.draw.line(screen, COLORS["BLUE"], to_pygame((transformed_points[i][1][0], transformed_points[i][2][0]), HEIGHT), to_pygame((transformed_points[i+1][1][0], transformed_points[i+1][2][0]), HEIGHT), 5)

    def _draw_TNB_vectors(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time) -> None:
        #STEP1: sub at t0, we get 3 TNB vectors
        T = self.T(time)
        new_T = np.array([  [T[0]], [T[1]], [T[2]]  ])
        N = self.N(time)
        new_N = np.array([  [N[0]], [N[1]], [N[2]]  ])
        B = self.B(time)
        new_B = np.array([  [B[0]], [B[1]], [B[2]]  ])
        #STEP2: scale the vectors for better visualization
        #new_T = 2*new_T
        #new_N = 2*new_N
        #new_B = 2*new_B
        #STEP3: each v's and tails of each v's undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system)
        A = np.hstack(coordinate_system.axises) #Linear-Tranformation matrix A        
        new_pos = self.pos(time)
        new_pos = np.array([ [new_pos[0]], [new_pos[1]], [new_pos[2]] ])
        new_T, new_N, new_B, new_pos  = np.dot(A, new_T), np.dot(A, new_N), np.dot(A, new_B), np.dot(A, new_pos)
        #STEPP4:shift toward the current position
        new_T += new_pos
        new_N += new_pos
        new_B += new_pos
        #STEP5: draw the TNB vectors (skip x coord, and draw;no need for projecting to yz plane, just skip x coordinate )
        pygame.draw.line(screen, COLORS["WHITE"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_T[1][0], new_T[2][0]), HEIGHT), 5) #draw T
        #display character T
        font = pygame.font.Font(None, 36)
        text = font.render("T", True, COLORS["WHITE"])
        screen.blit(text, to_pygame((new_T[1][0], new_T[2][0]), HEIGHT))
        ############################################################################################################
        pygame.draw.line(screen, COLORS["GRAY"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_N[1][0], new_N[2][0]), HEIGHT), 5) #draw N
        #display character N
        font = pygame.font.Font(None, 36)
        text = font.render("N", True, COLORS["GRAY"])
        screen.blit(text, to_pygame((new_N[1][0], new_N[2][0]), HEIGHT))
        ############################################################################################################
        pygame.draw.line(screen, COLORS["LIGHT_GRAY"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_B[1][0], new_B[2][0]), HEIGHT), 5) #draw B
        #display character B
        font = pygame.font.Font(None, 36)
        text = font.render("B", False, COLORS["LIGHT_GRAY"])
        screen.blit(text, to_pygame((new_B[1][0], new_B[2][0]), HEIGHT),)
    def _draw_velocity(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time) -> None:
        #STEP1: sub at t0, we get new velocity vector
        velocity = self.velocity(time)
        new_velocity = np.array([  [velocity[0]], [velocity[1]], [velocity[2]]  ])
        #STEP2: scale the vector for better visualization
        #new_velocity = 2*new_velocity
        #STEP3:  v and the tail of each v undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system)
        A = np.hstack(coordinate_system.axises) #Linear-Tranformation matrix A
        new_pos = self.pos(time)
        new_pos = np.array([ [new_pos[0]], [new_pos[1]], [new_pos[2]] ])
        new_velocity, new_pos  = np.dot(A, new_velocity), np.dot(A, new_pos)
        #STEPP4:shift toward the current position
        new_velocity += new_pos
        #STEP5: draw the velocity vector (skip x coord, and draw; no need for projecting to yz plane, just skip x coordinate )
        pygame.draw.line(screen, COLORS["RED"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_velocity[1][0], new_velocity[2][0]), HEIGHT), 5) #draw T
        #display character V
        font = pygame.font.Font(None, 36)
        text = font.render("V", True, COLORS["RED"])
        screen.blit(text, to_pygame((new_velocity[1][0], new_velocity[2][0]), HEIGHT))
    def _draw_acceleration(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time) -> None:
        #STEP1: sub at t0, we get new acceleration vector
        acceleration = self.acceleration(time)
        new_acceleration = np.array([  [acceleration[0]], [acceleration[1]], [acceleration[2]]  ])
        #STEP2: scale the vector for better visualization
        #new_acceleration = 2*new_acceleration
        #STEP3:  v and the tail of each v undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system)
        A = np.hstack(coordinate_system.axises) #Linear-Tranformation matrix A
        new_pos = self.pos(time)
        new_pos = np.array([ [new_pos[0]], [new_pos[1]], [new_pos[2]] ])
        new_acceleration, new_pos  = np.dot(A, new_acceleration), np.dot(A, new_pos)
        #STEPP4:shift toward the current position
        new_acceleration += new_pos
        #STEP5: draw the acceleration vector (skip x coord, and draw; no need for projecting to yz plane, just skip x coordinate )
        pygame.draw.line(screen, COLORS["YELLOW"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_acceleration[1][0], new_acceleration[2][0]), HEIGHT), 5) #draw T
        #display character A
        font = pygame.font.Font(None, 36)
        text = font.render("A", True, COLORS["YELLOW"])
        screen.blit(text, to_pygame((new_acceleration[1][0], new_acceleration[2][0]), HEIGHT))
    def _draw_tangential_acceleration(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time) -> None:
        #STEP1: sub at t0, we get new tangential acceleration vector
        tangential_acceleration = self.tangential_acceleration(time)
        new_tangential_acceleration = np.array([  [tangential_acceleration[0]], [tangential_acceleration[1]], [tangential_acceleration[2]]  ])
        #STEP2: scale the vector for better visualization
        #new_tangential_acceleration = 2*new_tangential_acceleration
        #STEP3:  v and the tail of each v undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system)
        A = np.hstack(coordinate_system.axises)
        new_pos = self.pos(time)
        new_pos = np.array([ [new_pos[0]], [new_pos[1]], [new_pos[2]] ])
        new_tangential_acceleration, new_pos  = np.dot(A, new_tangential_acceleration), np.dot(A, new_pos)
        #STEPP4:shift toward the current position
        new_tangential_acceleration += new_pos
        #STEP5: draw the tangential acceleration vector (skip x coord, and draw; no need for projecting to yz plane, just skip x coordinate )
        pygame.draw.line(screen, COLORS["YELLOW"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_tangential_acceleration[1][0], new_tangential_acceleration[2][0]), HEIGHT), 5) #draw T
        #display character At
        font = pygame.font.Font(None, 36)
        text = font.render("At", True, COLORS["YELLOW"])
        screen.blit(text, to_pygame((new_tangential_acceleration[1][0], new_tangential_acceleration[2][0]), HEIGHT))
    def _draw_normal_acceleration(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time) -> None:
        #STEP1: sub at t0, we get new normal acceleration vector
        normal_acceleration = self.normal_acceleration(time)
        new_normal_acceleration = np.array([  [normal_acceleration[0]], [normal_acceleration[1]], [normal_acceleration[2]]  ])
        #STEP2: scale the vector for better visualization
        #new_normal_acceleration = 2*new_normal_acceleration
        #STEP3:  v and the tail of each v undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system)
        A = np.hstack(coordinate_system.axises)
        new_pos = self.pos(time)
        new_pos = np.array([ [new_pos[0]], [new_pos[1]], [new_pos[2]] ])
        new_normal_acceleration, new_pos  = np.dot(A, new_normal_acceleration), np.dot(A, new_pos)
        #STEPP4:shift toward the current position
        new_normal_acceleration += new_pos
        #STEP5: draw the normal acceleration vector (skip x coord, and draw; no need for projecting to yz plane, just skip x coordinate )
        pygame.draw.line(screen, COLORS["YELLOW"], to_pygame((new_pos[1][0], new_pos[2][0]), HEIGHT), to_pygame((new_normal_acceleration[1][0], new_normal_acceleration[2][0]), HEIGHT), 5)
        #display character An
        font = pygame.font.Font(None, 36)
        text = font.render("An", True, COLORS["YELLOW"])
        screen.blit(text, to_pygame((new_normal_acceleration[1][0], new_normal_acceleration[2][0]), HEIGHT))
        




    def _unit_tangent_v(vvf: list[sp.Expr], verbose=False)-> list[sp.Expr]:
        r_1st_diff = VectorValuedFunction._diff(vvf)
        norm_r_1st_diff = VectorValuedFunction._norm(r_1st_diff)
        T = VectorValuedFunction._scale(1/norm_r_1st_diff, r_1st_diff)
        if verbose:
            print("T:", T)
        return T

    def _unit_normal_v(vvf: list[sp.Expr], verbose=False)-> list[sp.Expr]:
        T = VectorValuedFunction._unit_tangent_v(vvf)
        T_1st_diff = VectorValuedFunction._diff(T)
        norm_T_1st_diff = VectorValuedFunction._norm(T_1st_diff)
        N = VectorValuedFunction._scale(1/norm_T_1st_diff, T_1st_diff)
        if verbose:
            print("N:", N)
        return N

    def _unit_binormal_v(vvf: list[sp.Expr], verbose=False)-> list[sp.Expr]:
        T = VectorValuedFunction._unit_tangent_v(vvf)
        N = VectorValuedFunction._unit_normal_v(vvf)
        B = VectorValuedFunction._cross(T, N)
        if verbose:
            print("B:", B)
        return B

    def _norm(vvf: list[sp.Expr])-> sp.Expr:
        return sp.sqrt(sum([v**2 for v in vvf]))

    def _diff(vvf: list[sp.Expr])-> list[sp.Expr]:
        return [sp.diff(v, "t") for v in vvf]

    def _integrate(vvf: list[sp.Expr])-> list[sp.Expr]:
        return [sp.integrate(v, "t") for v in vvf]

    def _def_integral(vvf: list[sp.Expr], t0: float, t1: float) -> list[sp.Expr]:
        return [sp.integrate(v, ("t", t0, t1)) for v in vvf]

    def _add(vvf1: list[sp.Expr], vvf2: list[sp.Expr]):
        return [v1+v2 for v1, v2 in zip(vvf1, vvf2)]

    def _subtract(vvf1: list[sp.Expr], vvf2: list[sp.Expr]):
        return [v1-v2 for v1, v2 in zip(vvf1, vvf2)]

    def _scale(scalar: sp.Expr, vvf: list[sp.Expr]):
        return [v*scalar for v in vvf]

    def _multiply(vvf1: list[sp.Expr], vvf2: list[sp.Expr]):
        return [v1*v2 for v1, v2 in zip(vvf1, vvf2)]

    def _divide(vvf1: list[sp.Expr], vvf2: list[sp.Expr]):
        return [v1/v2 for v1, v2 in zip(vvf1, vvf2)]

    def _dot(vvf1: list[sp.Expr], vvf2: list[sp.Expr]):
        return sum([v1*v2 for v1, v2 in zip(vvf1, vvf2)])

    def _cross(vvf1: list[sp.Expr], vvf2: list[sp.Expr]):
        return [vvf1[1]*vvf2[2] - vvf1[2]*vvf2[1], vvf1[2]*vvf2[0] - vvf1[0]*vvf2[2], vvf1[0]*vvf2[1] - vvf1[1]*vvf2[0]]
class Plane_3D:
    def __init__(self, reference_frame: Coordinate3DSystem):
        self.reference_frame = reference_frame
        #generate list of points  [   [ [x],[y],[-10] ], ...] whose x ranges from -10 to 10 and y ranges from -10 to 10
        # Define ranges for x, y, and the constant z
        x_range = np.arange(-10, 11)  # x ranges from -10 to 10
        y_range = np.arange(-10, 11)  # y ranges from -10 to 10
        z_constant = -10             # z is a constant
        # Generate the grid of points
        points = []
        for y in y_range:
            for x in x_range:
                points.append([[x], [y], [z_constant]])
        # Convert to a NumPy array
        self.points = np.array(points)

    
    def draw(self, screen)-> None:
        points = copy.deepcopy(self.points)
        A = np.hstack(self.reference_frame.axises)
        transformed_points = np.matmul(A, points)
        #la.print_matrix(transformed_points)
        #draw the points using pygame.draw.polygone(screen, color, points, width)
        #example uf using polygon: pygame.draw.polygon(screen, COLORS["WHITE"], [(100, 100), (200, 200), (300, 100)], 5)            
        #pseudo code: draw.polygone(screen, color, ([[y],[z] ], [[y+1],[z]], [[y+1],[z+1]], [[y],[z+1]]), width)
        for i in range(len(transformed_points)-1):
            if i+1 < len(transformed_points) and (i+1) % 21 != 0:
                pygame.draw.polygon(screen, COLORS["WHITE"], [to_pygame((transformed_points[i][1][0], transformed_points[i][2][0]), HEIGHT), to_pygame((transformed_points[i+1][1][0], transformed_points[i+1][2][0]), HEIGHT), to_pygame((transformed_points[i+22][1][0], transformed_points[i+22][2][0]), HEIGHT), to_pygame((transformed_points[i+21][1][0], transformed_points[i+21][2][0]), HEIGHT)], 5)
            else:
                 
            #display character P
            font = pygame.font.Font(None, 36)
            text = font.render("P", True, COLORS["WHITE"])
            screen.blit(text, to_pygame((transformed_points[i][1][0], transformed_points[i][2][0]), HEIGHT))
                                                            
 
        


    
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 800))#,pygame.FULLSCREEN)
        pygame.display.set_caption("Game Logic and Draw Example")
        self.clock = pygame.time.Clock()

        # Initialize game variables or objects here
        #natural time counter of the physcial world
        self.time = 0
        self.coordinate_system = Coordinate3DSystem()
        self.Plane_3D = Plane_3D(self.coordinate_system)
        #create a vector valued function
        t = sp.symbols("t")
        #earth trajectory + constant position
        self.vvf1 = VectorValuedFunction([10*sp.cos(t), 10*sp.sin(t)-sp.cos(2*t), 0], self.coordinate_system)
        
    def handle_events(self):
        """Handle events: keyboard, mouse, etc."""
        ##notes: z axis points downwards, y axis points to the right. (y,z) plane
        keys = pygame.key.get_pressed()  # Check which keys are held down
        #rotation around the x-axis
        if keys[pygame.K_x]: #rotate clockwise
            alpha = 5
        elif keys[pygame.K_z]: #rotate counter-clockwise
            alpha = -5
        else:
            alpha = 0
        #rotate around the y-axis
        if keys[pygame.K_DOWN]: # counter-clockwise around the x-axis
            beta = 5
        elif keys[pygame.K_UP]:
            beta = -5
        else:
            beta = 0
        #rotate around the z-axis
        if keys[pygame.K_LEFT]: # counter-clockwise around the x-axis
            gamma = 5
        elif keys[pygame.K_RIGHT]:
            gamma = -5
        else:
            gamma = 0
        #Apply rotation
        self.coordinate_system.rotate(alpha, beta, gamma)
        #SCALE
        if keys[pygame.K_i]: #scale in
            scale = 1.1
        elif keys[pygame.K_o]: #scale out
            scale = 0.9
        else:
            scale = 1
        #Apply scale
        self.coordinate_system.scale(scale)
        #TRANSLATION
        #1 means 1 unit in the direction of the x-axis
        #2 means 1 units in the negative direction of the x-axis
        #3 means 1 unit in the direction of the y-axis
        #4 means 1 unit in the negative direction of the y-axis
        dx, dy = 0, 0
        if keys[pygame.K_1]:
            dx = 10
        elif keys[pygame.K_2]:
            dx = -10
        elif keys[pygame.K_3]:
            dy = 10
        elif keys[pygame.K_4]:
            dy = -10
        else:
            dx, dy = 0, 0
        #Apply translation
        camera_translation(dx, dy)
         
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def update_game_logic(self):
        """Run the actual game logic."""
        self.time += 0.1

         

    def draw(self):
        """Render everything on the screen."""
        self.screen.fill((0, 0, 0))  # Clear screen with a black color 
        #draw 
        self.coordinate_system.draw(self.screen)
        self.vvf1.draw(self.screen, self.time)
        self.Plane_3D.draw(self.screen)
        pygame.display.flip()  # Update the display

    def run(self):
        """Main game loop."""
        while True:
            self.handle_events()
            self.update_game_logic()  # Run game logic
            self.draw()  # Draw everything on the screen
            self.clock.tick(60)  # Limit the frame rate to 60 FPS
            #sleep for 1 seconds
            pygame.time.wait(50)

# Helper function to convert coordinates to pygame coordinates
def to_pygame(xy_pair: tuple, height: int) -> tuple:
   #height means the height of the screen
   """Convert coordinates into pygame coordinates (lower-left => top left)."""
   return (xy_pair[0]+WIDTH/2+X_TRANSLATE, height - xy_pair[1]-HEIGHT/2 - Y_TRANSLATE)
def camera_translation(dx: int, dy: int):
    global X_TRANSLATE, Y_TRANSLATE
    X_TRANSLATE += dx
    Y_TRANSLATE += dy
    


# Run the game
if __name__ == "__main__":
    game = Game()
    game.run()


