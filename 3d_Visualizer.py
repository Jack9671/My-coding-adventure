import copy
import pygame
import MatrixLibrary as MLib
import linearalgebra as la
import math
import random
import numpy as np
from abc import ABC, abstractmethod

class Object_3D(ABC):
    def __init__(self, points: list[list[int]], centroid: list[list[int]]):
        self.points = points
        self.centroid = centroid

    #ABSRACT METHOD
    @abstractmethod
    def Draw(self, screen: pygame.Surface):
        """depends on the object"""
        pass

    #INSTANCE METHOD
    def Get_2D_Projection_onto_yz_plane(self):
        num_of_points = len(self.points)
        temporary_points = copy.deepcopy(self.points)
        i = 0
        while i < num_of_points:
            temporary_points[i] = la.Get_projection_vector([[0,0],
                                                       [1,0],
                                                       [0,1]], temporary_points[i])
            i += 1
        return temporary_points 
    
    def Rotate(self, alpha: float | int =0, beta: float | int =0, gamma: float | int=0):
        ## Initialize the rotation matrices
        points = self.points
        centroid = self.centroid
        Rx = MLib.MatrixLibrary().Rotation_matrix(alpha, axis='x')
        Ry = MLib.MatrixLibrary().Rotation_matrix(beta, axis='y')
        Rz = MLib.MatrixLibrary().Rotation_matrix(gamma, axis='z')
        #Step1: let the centroid be the origin
        i = 0
        while i < len(points):
            points[i] = points[i] - centroid
            i += 1
        #Step2: apply the rotation matrix
        i = 0
        while i < len(points):
            points[i] = np.dot(Rx, points[i])
            points[i] = np.dot(Ry, points[i])
            points[i] = np.dot(Rz, points[i])
            i += 1
        #Step3: Put the points back to the original position
        i = 0
        while i < len(points):
            points[i] = points[i] + centroid
            i += 1
        return points
    
    def Scale(self, scale: float | int):
        points = self.points
        centroid = self.centroid
        i = 0
        while i < len(points):
            points[i] = points[i] - centroid
            points[i] = np.dot([[scale,0,0],
                                     [0,scale,0],
                                     [0,0,scale]], points[i])
            points[i] = points[i] + centroid
            i += 1
        return points


class Cube(Object_3D):
    def __init__(self):
        X=0 #initial displacement x
        Y=0 #initial displacement y
        Z= 0 #initial displacement z
        S = 200.0 #size of the shape
        self.points = [[ [X],[Y],[Z], ], #1
                       [ [S+X],[Y],[Z], ], #2
                       [ [S+X],[S+Y],[Z], ], #3
                       [ [X],[S+Y],[Z], ], #4
                       [ [X],[S+Y],[S+Z], ], #5
                       [ [X],[Y],[S+Z], ], #6
                       [ [S+X],[Y],[S+Z], ], #7
                       [ [S+X],[S+Y],[S+Z]] ]  #8
        #get self.centroid
        i = 0
        self.centroid = [ [0],[0],[0] ]
        while i < len(self.points):
            self.centroid[0][0] += self.points[i][0][0]
            self.centroid[1][0] += self.points[i][1][0]
            self.centroid[2][0] += self.points[i][2][0]
            i += 1
        self.centroid[0][0] /= len(self.points)
        self.centroid[1][0] /= len(self.points)
        self.centroid[2][0] /= len(self.points)
        super().__init__(self.points, self.centroid)
        
    def Draw(self, screen):
        #Step 1: Get 2D projection; Make copy of points; Pop the x point; convert to list for drawing, Apply shift the points to the center of the screen
        temporary_points = self.Get_2D_Projection_onto_yz_plane() # because I have to discard the x point, so I use temporary points to preserve the infor of 3d points
        i = 0
        while i < len(temporary_points):
                temporary_points[i].pop(0) # discard x point
                temporary_points[i] = Col_to_list(temporary_points[i]) # convert column vector to list for drawing for compatibility
                temporary_points[i][0] += 150 # shift y
                temporary_points[i][1] += 100 # shift z
                i += 1
        #Step2: Draw 
        color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[2],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[4],temporary_points[5],temporary_points[6],temporary_points[7]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[7],temporary_points[6]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[2],temporary_points[3],temporary_points[4],temporary_points[7]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[3],temporary_points[4],temporary_points[5]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[0],temporary_points[5],temporary_points[6]], 1)

class Tetrahedron(Object_3D):
    def __init__(self):
        X=0
        Y=0
        Z= 0
        S = 200.0
        self.points = [[ [X],[Y],[Z], ], #1
                       [ [S+X],[Y],[Z], ], #2
                       [ [S/2+X],[S+Y],[Z], ], #3
                       [ [S/2+X],[S/2+Y],[S+Z]] ] #4
        #get self.centroid
        i = 0
        self.centroid = [ [0],[0],[0] ]
        while i < len(self.points):
            self.centroid[0][0] += self.points[i][0][0]
            self.centroid[1][0] += self.points[i][1][0]
            self.centroid[2][0] += self.points[i][2][0]
            i += 1
        self.centroid[0][0] /= len(self.points)
        self.centroid[1][0] /= len(self.points)
        self.centroid[2][0] /= len(self.points)
        super().__init__(self.points, self.centroid)
    
    def Draw(self, screen):
        #Step 1: Get 2D projection; Make copy of points; Pop the x point; convert to list for drawing, Apply shift the points to the center of the screen
        temporary_points = self.Get_2D_Projection_onto_yz_plane() # because I have to discard the x point, so I use temporary points to preserve the infor of 3d points
        i = 0
        while i < len(temporary_points):
                temporary_points[i].pop(0) # discard x point
                temporary_points[i] = Col_to_list(temporary_points[i]) # convert column vector to list for drawing for compatibility
                temporary_points[i][0] += 800 # shift y
                temporary_points[i][1] += 100 # shift z
                i += 1
        #Step2: Draw 
        color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[2]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[2],temporary_points[3]], 1)

class Square_Pyramid(Object_3D):
    def __init__(self):
        X=0
        Y=0
        Z= 10.0
        S = 200.0
        self.points = [[ [X],[Y],[Z], ], #1
                       [ [S+X],[Y],[Z], ], #2
                       [ [S+X],[S+Y],[Z], ], #3
                       [ [X],[S+Y],[Z], ], #4
                       [ [S/2+X],[S/2+Y],[S+Z]] ] #5
        #get self.centroid
        i = 0
        self.centroid = [ [0],[0],[0] ]
        while i < len(self.points):
            self.centroid[0][0] += self.points[i][0][0]
            self.centroid[1][0] += self.points[i][1][0]
            self.centroid[2][0] += self.points[i][2][0]
            i += 1
        self.centroid[0][0] /= len(self.points)
        self.centroid[1][0] /= len(self.points)
        self.centroid[2][0] /= len(self.points)
        super().__init__(self.points, self.centroid)
    
    def Draw(self, screen):
        #Step 1: Get 2D projection; Make copy of points; Pop the x point; convert to list for drawing, Apply shift the points to the center of the screen
        temporary_points = self.Get_2D_Projection_onto_yz_plane() # because I have to discard the x point, so I use temporary points to preserve the infor of 3d points
        i = 0
        while i < len(temporary_points):
                temporary_points[i].pop(0) # discard x point
                temporary_points[i] = Col_to_list(temporary_points[i]) # convert column vector to list for drawing for compatibility
                temporary_points[i][0] += 150 # shift y
                temporary_points[i][1] += 450 # shift z
                i += 1
        #Step2: Draw 
        color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[2],temporary_points[3]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[2],temporary_points[3],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[3],temporary_points[0],temporary_points[4]], 1)

class Octahedron(Object_3D):
    def __init__(self):
        X=0
        Y=0
        Z= 0
        S = 200.0
        self.points = [[ [X],[Y],[Z], ], #1
                          [ [S+X],[Y],[Z], ], #2
                          [ [S+X],[S+Y],[Z], ], #3
                          [ [X],[S+Y],[Z], ], #4
                          [ [S/2+X],[S/2+Y],[S+Z] ], #5
                          [ [S/2+X],[S/2+Y],[Z-S] ] ] #6
        
        #get self.centroid
        i = 0
        self.centroid = [ [0],[0],[0] ]
        while i < len(self.points):
            self.centroid[0][0] += self.points[i][0][0]
            self.centroid[1][0] += self.points[i][1][0]
            self.centroid[2][0] += self.points[i][2][0]
            i += 1
        self.centroid[0][0] /= len(self.points)
        self.centroid[1][0] /= len(self.points)
        self.centroid[2][0] /= len(self.points)
        super().__init__(self.points, self.centroid)
    
    def Draw(self, screen):
        #Step 1: Get 2D projection; Make copy of points; Pop the x point; convert to list for drawing, Apply shift the points to the center of the screen
        temporary_points = self.Get_2D_Projection_onto_yz_plane() # because I have to discard the x point, so I use temporary points to preserve the infor of 3d points
        i = 0
        while i < len(temporary_points):
                temporary_points[i].pop(0) # discard x point
                temporary_points[i] = Col_to_list(temporary_points[i]) # convert column vector to list for drawing for compatibility
                temporary_points[i][0] += 800 # shift y
                temporary_points[i][1] += 520 # shift z
                i += 1
        #Step2: Draw 
        color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[2],temporary_points[3],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[3],temporary_points[0],temporary_points[4]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[0],temporary_points[1],temporary_points[5]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[1],temporary_points[2],temporary_points[5]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[2],temporary_points[3],temporary_points[5]], 1)
        pygame.draw.lines(screen, color, True, [temporary_points[3],temporary_points[0],temporary_points[5]], 1)

        


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1300, 800))
        pygame.display.set_caption("Game Logic and Draw Example")
        self.clock = pygame.time.Clock()

        # Initialize game variables or objects here
        self.cube = Cube()
        self.tetrahedron = Tetrahedron()
        self.square_pyramid = Square_Pyramid()
        self.octahedron = Octahedron()
        #self.docdecahedron = Docdecahedron()
        self.Mlib = MLib.MatrixLibrary()

    def handle_events(self):
        """Handle events: keyboard, mouse, etc."""
        ##notes: z axis points downwards, y axis points to the right. (y,z) plane
        keys = pygame.key.get_pressed()  # Check which keys are held down
        #rotation around the x-axis
        if keys[pygame.K_x]: #rotate clockwise
            alpha = 2
        elif keys[pygame.K_z]: #rotate counter-clockwise
            alpha = -2
        else:
            alpha = 0
        #rotate around the y-axis
        if keys[pygame.K_DOWN]: # counter-clockwise around the x-axis
            beta = 2
        elif keys[pygame.K_UP]:
            beta = -2
        else:
            beta = 0
        #rotate around the z-axis
        if keys[pygame.K_LEFT]: # counter-clockwise around the x-axis
            gamma = 2
        elif keys[pygame.K_RIGHT]:
            gamma = -2
        else:
            gamma = 0
        #Apply rotation
        self.cube.Rotate(alpha, beta, gamma)
        self.tetrahedron.Rotate(alpha, beta, gamma)
        self.square_pyramid.Rotate(alpha, beta, gamma)
        self.octahedron.Rotate(alpha, beta, gamma)
        
        #SCALE
        if keys[pygame.K_i]: #scale in
            scale = 1.1
        elif keys[pygame.K_o]: #scale out
            scale = 0.9
        else:
            scale = 1
        #Apply scale
        self.cube.Scale(scale)
        self.tetrahedron.Scale(scale)
        self.square_pyramid.Scale(scale)
        self.octahedron.Scale(scale)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def update_game_logic(self):
        """Run the actual game logic."""
        pass

    def draw(self):
        """Render everything on the screen."""
        self.screen.fill((0, 0, 0))  # Clear screen with a white background
        #draw 
        self.cube.Draw(self.screen)
        self.tetrahedron.Draw(self.screen)
        self.square_pyramid.Draw(self.screen)
        self.octahedron.Draw(self.screen)
        #draw origin of coordinate system at 450, 400
        pygame.draw.circle(self.screen, (255,255,255), (450,400), 5)
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
def to_pygame(coords, height):
   """Convert coordinates into pygame coordinates (lower-left => top left)."""
   return (coords[0], height - coords[1])

# Run the game
if __name__ == "__main__":
    game = Game()
    game.run()

'''
# Run a single iteration of the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw 
    cube.Draw_Cube()
    print("Drawing")
    # Update the display
    pygame.display.flip()

    # Break after a single iteration
    running = False  # Exits the loop after one iteration

pygame.quit()
'''
#weeeeeeeeeeeeeeee
'''
# pygame setup
## 3D world onto 2D screen as yz plane
pygame.init()
screen = pygame.display.set_mode((1000, 600))
clock = pygame.time.Clock()
running = True
# create instance of object AREA
cube = Three_Dimensional_Cube()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw
    cube.Draw_Cube()
    # Update the display
    pygame.display.flip()

pygame.quit()


'''