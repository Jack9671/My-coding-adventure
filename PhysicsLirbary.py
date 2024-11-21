import math
import MatrixOperator as mo
import numpy as np
def main():
    A =np.array([[1,1],
                 [1,2],
                 [1,3]])
    At = np.transpose(A)
    AtA = np.dot(At,A)
    b = np.array([[0],[1]])
    #solve AtA x = b
    x = np.linalg.solve(AtA,b)
    c = np.dot(A,x)
    #print(f"c = {c}")
    d = np.dot(At,c)
    print(d)



def tension_3d_calculator(p: list[list[int]],q:list[list[int]],r: list[list[int]],verbose = False):
    # P = 3D direction vector
    # Q = 3D direction vector
    # R = 3D direction vector
    x= [[1],[0],[0]]
    y= [[0],[1],[0]]
    z= [[0],[0],[1]]
    #g = [[0],[0],[-294]]
    a11 = math.cos(angle(x,p))
    a12 = math.cos(angle(x,q))
    a13 = math.cos(angle(x,r))
    a21 = math.cos(angle(y,p))
    a22 = math.cos(angle(y,q))
    a23 = math.cos(angle(y,r))
    a31 = math.cos(angle(z,p))
    a32 = math.cos(angle(z,q))
    a33 = math.cos(angle(z,r))
    A =[[a11,a12,a13,0],
        [a21,a22,a23,0],
        [a31,a32,a33,294]]
    mo.Print_matrix(A)
    magnitude_of_3_vectors = mo.Solve_system_of_equations(A)
    if verbose:
        for i in range(len(magnitude_of_3_vectors)):
            print (f"||V{i+1}|| = {magnitude_of_3_vectors[i]}")
        print (f"P1 = {componentize(magnitude_of_3_vectors[0][0],angle(x,p), angle(y,p), angle(z,p))}")
        print (f"Q2 = {componentize(magnitude_of_3_vectors[1][0],angle(x,q), angle(y,q), angle(z,q))}")
        print (f"R3 = {componentize(magnitude_of_3_vectors[1][0],angle(x,r), angle(y,r), angle(z,r))}")


            
    return magnitude_of_3_vectors

def magnitude(vector: np.ndarray) -> float:
    # vector = n_dimensional vector
    #Step1: Get the magnitude of the vector
    magnitude = 0
    y_index = 0
    while y_index < len(vector):
        magnitude += vector[y_index][0]**2
        y_index+=1
    return math.sqrt(magnitude)

def angle(v1: list[list[int]], v2: list[list[int]]):
    return math.acos(mo.dot_product(v1,v2)[0][0]/(magnitude(v1)*magnitude(v2)))

def componentize(magnitude: float, xOv: float | int,yOv: float | int , zOv: float | int) -> list[list[int]]:
    return[[round(magnitude*math.cos(xOv),1)],[round(magnitude*math.cos(yOv),1)],[round(magnitude*math.cos(zOv),1)]]



if __name__ == "__main__":
    main()