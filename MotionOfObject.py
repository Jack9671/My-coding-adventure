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
WIDTH = 2000
HEIGHT = 700
#enum
COLORS = { "WHITE": (255, 255, 255), "BLACK": (0, 0, 0), "RED": (255, 0, 0), "GREEN": (0, 255, 0), "BLUE": (0, 0, 255) } 
class Coordinate3DSystem:
    def __init__(self, vector_valued_function = [sp.cos("t"), sp.sin("t"), "t"]):
        self.origin = np.array([[0], [0], [0]]) #O
        self.axises = [np.array([ [10], [0], [0]]), #X-axis
                      np.array([ [0],[10], [0]  ]), #Y-axis
                      np.array([  [0], [0], [10]  ])] #Z-axis
        #call transtaltion matrix to the center of screen
        self.translate(100,100,100)
    def draw(self, screen)-> None:
        #Step1: get temporary copy of the axises points and origin to prevent changing the original values
        axises = copy.deepcopy(self.axises)
        origin = copy.deepcopy(self.origin)
        #Step2: if projecting on yz plane, x-coordinate is automatically 0, therefore no need to muliply projection matrix
        #which means getting [1] and [2] index of the projection vector
        for i in range(3):
            #parameter for pygame.draw.line is (screen, color, start_pos, end_pos, width)
            #parameter for pygame.draw.circle is (screen, color, center, radius, width)
            #draw the origin
            pygame.draw.circle(screen, COLORS["RED"], to_pygame( (origin[1][0], origin[2][0]) ,HEIGHT), 10)
            #draw the axises
            #draw axis individually x -> red, y -> green, z -> blue
            if i == 0:
                pygame.draw.line(screen, COLORS["RED"], to_pygame((origin[1][0], origin[2][0]), HEIGHT), to_pygame( (axises[i][1][0], axises[i][2][0]), HEIGHT ), 10)
            elif i == 1:
                pygame.draw.line(screen, COLORS["GREEN"], to_pygame((origin[1][0], origin[2][0]), HEIGHT), to_pygame( (axises[i][1][0], axises[i][2][0]), HEIGHT ), 10)
            else:
                pygame.draw.line(screen, COLORS["BLUE"], to_pygame((origin[1][0], origin[2][0]), HEIGHT), to_pygame( (axises[i][1][0], axises[i][2][0]), HEIGHT ), 10)
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

    def translate(self, dx: int|float, dy: int|float, dz: int|float):
        axises = self.axises 
        origin = self.origin
        translation_matrix = MLib.MatrixLibrary().Translation_matrix(dx,dy,dz)
        for i in range(3):
            #convert to homogeneous coordinates
            axises[i] = np.vstack(( axises[i] , [[1]]  ))
            axises[i] = np.dot(translation_matrix, axises[i])  
            # Convert back to 3D coordinates by dropping the last row
            axises[i] = axises[i][:3]
            print(f"axis {i}: = {axises[i]}")

        origin = np.vstack(( origin , [[1]]  ))
        origin = np.dot(translation_matrix,origin)
        origin = origin[:3]
        print(f"origin: = {origin}")

              








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
        ##############################################
        tangential_acceleration = VectorValuedFunction._dot(velocity, acceleration) / VectorValuedFunction._norm(velocity) # scalar along T
        self.tangential_acceleration = sp.lambdify("t", tangential_acceleration)
        ##############################################
        normal_acceleration = sp.sqrt(VectorValuedFunction._norm(acceleration)**2-tangential_acceleration**2)# scalar along N
        self.normal_acceleration = sp.lambdify("t", normal_acceleration)
        ############################################################################################################
        TNB_vector = VectorValuedFunction.TNB_vectors(symbolic_expr)
        self.T = sp.lambdify("t", [TNB_vector[0][0], TNB_vector[0][1], TNB_vector[0][2]])
        self.N = sp.lambdify("t", [TNB_vector[1][0], TNB_vector[1][1], TNB_vector[1][2]])
        self.B = sp.lambdify("t", [TNB_vector[2][0], TNB_vector[2][1], TNB_vector[2][2]])
        ############################################################################################################
        curvature = VectorValuedFunction.curvature(symbolic_expr)
        self.curvature = sp.lambdify("t", curvature)
        ############################################################################################################
        self.xyz_infor = np.array([[[0],[0],[0]]])  # np.array( [ 
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
    def draw_trajectory(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, time: time = 0) -> None:
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

    def draw_TNB_vectors(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem, t: time) -> None:
        #STEP1: sub at t0, we get 3 TNB vectors
        T = self.T(time)
        new_T = np.array([  [T[0]], [T[1]], [T[2]]  ])
        N = self.N(time)
        new_N = np.array([  [N[0]], [N[1]], [N[2]]  ])
        B = self.B(time)
        new_B = np.array([  [B[0]], [B[1]], [B[2]]  ])
        #STEP2: each v's undergoes linear transfromation whose columns are orthogonal basis vectors(3 axies of self-referenced coordinate system)
        A = np.hstack(coordinate_system.axises) #Linear-Tranformation matrix A
        new_T,new_N,new_B = np.dot(A, new_T),np.dot(A, new_N),np.dot(A, new_B)

        #STEPP3:shift toward the current position

        #STEP4: no need for projecting to yz plane, just skip x coordinate 
        #STEP5: skip x coord, and draw 
        self.T(time)

                
    '''       
    def rotate(self, alpha: float | int =0, beta: float | int =0, gamma: float | int=0)-> None:
        #rotation matrix around the (self-referenced) axises and not the axises (global) that is used for projection
        Rx = MLib.MatrixLibrary().Rotation_matrix(alpha, art_axis= Coordinate3DSystem().axises[0])
        Ry = MLib.MatrixLibrary().Rotation_matrix(beta, art_axis= Coordinate3DSystem().axises[1])
        Rz = MLib.MatrixLibrary().Rotation_matrix(gamma, art_axis= Coordinate3DSystem().axises[2])
        #apply rotation matrix to the points
        for i in range(len(self.xyz_infor)):
            self.xyz_infor[i] = np.dot(Rx, self.xyz_infor[i])
            self.xyz_infor[i] = np.dot(Ry, self.xyz_infor[i])
            self.xyz_infor[i] = np.dot(Rz, self.xyz_infor[i])
    def scale(self, scale: float | int =1)-> None:
        for i in range(len(self.xyz_infor)):
            self.xyz_infor[i] = scale*self.xyz_infor[i]
    '''
    #HELPER FUNCTIONS
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


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1300, 800))
        pygame.display.set_caption("Game Logic and Draw Example")
        self.clock = pygame.time.Clock()

        # Initialize game variables or objects here
        #natural time counter of the physcial world
        self.time = 0
        self.coordinate_system = Coordinate3DSystem()
        #create a vector valued function
        t = sp.symbols("t")
        #helix
        self.vvf1 = VectorValuedFunction([10*sp.cos(t), 
                                          10*sp.sin(t), t], self.coordinate_system)
        
        #VectorValuedFunction([100*sp.cos(t)+1.5*100*sp.cos(2*t/3),
         #                                100*sp.sin(t)-1.5*100*sp.sin(2*t/3),
          #                                  100*sp.cos(t*20)])

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def update_game_logic(self):
        """Run the actual game logic."""
        #evaluate the position of the object at time t
        if self.time < 100:
           self.time += 0.1
        
           #new_pos = self.vvf1.pos(self.time)
           #print("new_pos:", np.array([[new_pos[0]], [new_pos[1]], [new_pos[2]] ]) )
           #self.vvf1.xyz_infor= np.append( self.vvf1.xyz_infor ,[ [ [new_pos[0]], [new_pos[1]], [new_pos[2]] ] ], axis=0)
           #print the position of the object at time t
           #print(f" new: {self.vvf1.xyz_infor}")  
        else:
            pass            

    def draw(self):
        """Render everything on the screen."""
        self.screen.fill((0, 0, 0))  # Clear screen with a black color 
        #draw 
        self.coordinate_system.draw(self.screen)
        self.vvf1.draw_trajectory(self.screen, self.coordinate_system, self.time)
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
   return (xy_pair[0], height - xy_pair[1])

# Run the game
if __name__ == "__main__":
    game = Game()
    game.run()

