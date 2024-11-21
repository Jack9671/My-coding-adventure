import copy
import pygame
import MatrixOperator as mo
import MatrixLibrary
from MatrixOperator import Col_to_list
import sys
import math
import random

def main():
        pygame.init()
        #Step 1: set up cube
        X=0
        Y=0
        Z= 0
        S = 100.0
        points = [[ [X],[Y],[Z], ], #1
                       [ [S+X],[Y],[Z], ], #2
                       [ [S+X],[S+Y],[Z], ], #3
                       [ [X],[S+Y],[Z], ], #4
                       [ [X],[S+Y],[S+Z], ], #5
                       [ [X],[Y],[S+Z], ], #6
                       [ [S+X],[Y],[S+Z], ], #7
                       [ [S+X],[S+Y],[S+Z]] ]  #8
        # tetrhedron
        points2 = [[ [X],[Y],[Z], ], #1
                 [ [S+X],[Y],[Z], ], #2
                    [ [X],[S+Y],[Z], ], #3
                    [ [X],[Y],[S+Z] ] ] #4        
        #get self.centroid
        i = 0
        centroid = [ [0],[0],[0] ]
        while i < len(points):
            centroid[0][0] += points[i][0][0]
            centroid[1][0] += points[i][1][0]
            centroid[2][0] += points[i][2][0]
            #divide
            i += 1
        centroid[0][0] /= len(points)
        centroid[1][0] /= len(points)
        centroid[2][0] /= len(points)
        # loop drawing
        stop_yet = False
        while stop_yet == False:
            keys = pygame.key.get_pressed()  # Check which keys are held down
            if keys[pygame.K_x]: #rotate clockwise
                alpha = 10 
            elif keys[pygame.K_z]: #rotate counter-clockwise
                alpha = -10
            else:
                alpha = 0
            if keys[pygame.K_DOWN]: # counter-clockwise around the x-axis
                beta = 10
            elif keys[pygame.K_UP]:
                beta = -10
            else:
                beta = 0
            if keys[pygame.K_LEFT]: # counter-clockwise around the x-axis
                gamma = 10
            elif keys[pygame.K_RIGHT]:
                gamma = -10
            else:
                gamma = 0

            Draw_cube(points, centroid, alpha, beta, gamma)
            #Draw_tetrahedron(points2, centroid, alpha, beta, gamma)
        sys.exit()        

def Get_2D_Projection(points, centroid, alpha=0, beta=0, gamma=0):
     ## Initialize the variables
     #random angles
    #alpha = 5#random.randint(0, 360)
    #beta = -5#random.randint(0, 360)
    #gamma = 5#random.randint(0, 360)
    Rx = MatrixLibrary.MatrixLibrary().Rotation_matrix(alpha, axis='x')
    Ry = MatrixLibrary.MatrixLibrary().Rotation_matrix(beta, axis='y')
    Rz = MatrixLibrary.MatrixLibrary().Rotation_matrix(gamma, axis='z')
     #Step1: let the centroid be the origin
    i = 0
    while i < len(points):
         points[i] = mo.Subtract(points[i], centroid)
         i += 1
    #Step2: apply the rotation matrix
    i = 0
    while i < len(points):
            points[i] = mo.Multiply(Rx, points[i])
            points[i] = mo.Multiply(Ry, points[i])
            points[i] = mo.Multiply(Rz, points[i])
            i += 1
    #Step3: Put the points back to the original position
    i = 0
    while i < len(points):
        points[i] = mo.Add(points[i], centroid)
        i += 1
    #Step4: multiply the points by the projection matrix
    fake_points = copy.deepcopy(points) 
    i = 0
    while i < len(points):
       # points[i] = mo.Get_projection_vector([[0,0],
        #                                       [1,0],
         #                                      [0,1]], points[i])
        print(f"point before projection: {fake_points[i]}")
        fake_points[i] = mo.Multiply([[0,0,0],
                                 [0,1,0],
                                 [0,0,1]], fake_points[i])
        print(f"point after projection: {fake_points[i]}")
        i += 1

    return fake_points
    
def Draw_cube(points, centroid, alpha=0, beta=0, gamma=0):
    print("Drawing")
    #step0: get the 2D projection
    points = Get_2D_Projection(points, centroid, alpha, beta, gamma)
    #Step1: make a copy of the points
    temporary_points = copy.deepcopy(points)
    #Step2: Pop the x coordinate out of the points
    i = 0
    while i < len(temporary_points):
        temporary_points[i].pop(0) #remove the x coordinate
        i += 1
    #Step3: Convert the vector_based points to list format for drawing
    i = 0
    while i < len(temporary_points):
        temporary_points[i] = Col_to_list(temporary_points[i])
        i += 1
    #Step4: Shift the points to the center of the screen
    i = 0
    while i < len(temporary_points):
        temporary_points[i][0] += 200
        temporary_points[i][1] += 200
        i += 1
    #Step5: Draw the cube
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("3D Projection")
    done = False
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill((0, 0, 0))
        color = (255,255,255)#random.choice([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[2],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[4],temporary_points[5],temporary_points[6],temporary_points[7]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[7],temporary_points[6]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[2],temporary_points[3],temporary_points[4],temporary_points[7]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[3],temporary_points[4],temporary_points[5]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[0],temporary_points[5],temporary_points[6]], 1)
        pygame.display.flip()
        clock.tick(60)
        #delay for 1 second
        pygame.time.wait(50)
        done = True

def Draw_tetrahedron(points, centroid, alpha=0, beta=0, gamma=0):
    print("Drawing")
    #step0: get the 2D projection
    points = Get_2D_Projection(points, centroid, alpha, beta, gamma)
    #Step1: make a copy of the points
    temporary_points = copy.deepcopy(points)
    #Step2: Pop the x coordinate out of the points
    i = 0
    while i < len(temporary_points):
        temporary_points[i].pop(0) #remove the x coordinate
        i += 1
    #Step3: Convert the vector_based points to list
    i = 0
    while i < len(temporary_points):
        temporary_points[i] = Col_to_list(temporary_points[i])
        #swap the y and z coordinates
        #temporary_points[i][0], temporary_points[i][1] = temporary_points[i][1], temporary_points[i][0]
        i += 1
    #Step4: Shift the points to the center of the screen
    i = 0
    while i < len(temporary_points):
        temporary_points[i][0] += 200
        temporary_points[i][1] += 200
        i += 1
    #Step5: Draw the cube
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("3D Projection")
    done = False
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill((0, 0, 0))
        color = (255,255,255)#random.choice([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[2]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[2],temporary_points[3]], 1)
        pygame.display.flip()
        clock.tick(60)
        #delay for 1 second
        pygame.time.wait(50)
        done = True

    
if __name__ == "__main__":
    main()
        