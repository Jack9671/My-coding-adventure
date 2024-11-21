import random
import math
import copy
import numpy as np
import sympy as sp
import mpmath as mp
import PhysicsLirbary as pl
import MatrixLibrary as ml
import time
def main():
    print("Welcome to the Linear Algebra Library")
    #test time of multiplication
    A = np.random.rand(1000,1000)
    B = np.random.rand(1000,1000)
    start = time.time()
    C = np.dot(A,B)
    end = time.time()
    print(f"Time taken for numpy matrix multiplication: {end-start}")


def det(A: np.ndarray, verbose = False) -> float:
    U = pldu_factor(A)[2]
    permutation_parity = pldu_factor(A)[3]
    determinant = 1 #default
    for i in range(len(U)):
        determinant *= U[i][i]
    determinant *= permutation_parity
    if verbose:
        print(f"The determinant of the matrix is: {determinant}")
    return determinant

def rref(matrix: np.ndarray, verbose: bool = False) -> np.ndarray| None: #Reduced Row Echelon Form
    #information of the matrix [A]
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm: Gauss-Jordan elimination
    #STEP1: Gaussian Elimination
    matrix = gauss_eliminate(matrix, verbose=verbose)
    #STEP2: Jordan Elimination
    matrix = jordan_eliminate(matrix, verbose = verbose)
    #STEP3: Normalize the pivots to 1 to get R
    for y_index in range(num_of_Y): 
        for x_index in range(y_index, num_of_X): #check from left to right to find the non-0 pivot of upper or semi-upper triangular matrix
            if matrix[y_index][x_index] != 0:
                pivot_normalized_factor = matrix[y_index][x_index]
                matrix[y_index] = matrix[y_index] / pivot_normalized_factor #normalize the pivot to 1
                break #After normalizing the current y_index, break the inner loop to move to the next y_index    
    if verbose:
        print("After normalization, R is:")
        print_matrix(matrix)
    #STEP6: Return the R matrix
    return matrix

def pldu_factor(matrix: np.ndarray, get_D_or_not: bool = False, verbose: bool = False) -> list[np.ndarray] | None:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #additional part for finding the determinant
    permutation_parity = 0
    #main algorithm
    #Step1: Initialize the L matrix and a column matrix P containing the index of the row
    L_matrix = np.identity(num_of_Y)
    P_row_index_storage_matrix = [y for y in range(num_of_Y)]
    Arr_of_steps_of_permutation = [0 for y in range(num_of_Y)]
    #Step2: Modified Version of Gaussian Elimination for purpose of finding P and L
    ## The idea is to transform the matrix into an upper triangular matrix
    for x_index in range(num_of_X-1):  # note: customizable or we do not need to find the pivot of the last column of a square matrix
        #main algorithm of getting elimination matrix
        #STEP1: initialize the elimination matrix
        eli_matrix = np.identity(num_of_Y)
        #STEP2: check whether current pivot matrix[x_index][x_index] is 0 or not, if it is 0, exchange and ready for elimination
        y_index = x_index
        if matrix[y_index][x_index] == 0: #if the current y is 0, then exchange the y with the next y that has non-0 pivot
           y_index_of_non0_pivot = _find_non0_pivot(matrix, x_index, y_index)
           permutation_parity += 1
           if y_index_of_non0_pivot != None:
               matrix[[y_index,y_index_of_non0_pivot]] = matrix[[y_index_of_non0_pivot,y_index]] # swap the rows
               L_matrix[y_index], L_matrix[y_index_of_non0_pivot] = L_matrix[y_index_of_non0_pivot], L_matrix[y_index] # swap the rows
               Arr_of_steps_of_permutation.append(y_index)
               Arr_of_steps_of_permutation.append(y_index_of_non0_pivot)
           elif y_index_of_non0_pivot == None:
               print("There exists a zero row during Gaussian Elimination") #if there is no non-0 pivot, then the matrix is singular
               return None
        #STEP3: elimination process for each y down the current y
        for y_index in range(x_index + 1,num_of_Y):#plus 1 because we do not need to eliminate the current row where the current pivot is located
                multipler = matrix[y_index][x_index]/(matrix[x_index][x_index])#
                L_matrix[y_index][x_index] = multipler
                eli_matrix[y_index][x_index] = -(multipler) 
        #then we get eli_matrix
        matrix = np.dot(eli_matrix, matrix)
    #check if the last y_index contains all 0 after elimination, if yes, then the system is singular
    if all(x == 0 for x in matrix[num_of_Y-1]):
        print("There exists a last all-0 row during Gaussian Elimination") if verbose else None
        return None
    #Step3: Set P_row_index_storage_matrix from Arr_of_steps_of_permutation
    for y1 in (len(Arr_of_steps_of_permutation)-1,0,-2):
        y2 = y1 -1
        P_row_index_storage_matrix[Arr_of_steps_of_permutation[y1]], P_row_index_storage_matrix[Arr_of_steps_of_permutation[y2]] = P_row_index_storage_matrix[Arr_of_steps_of_permutation[y2]], P_row_index_storage_matrix[Arr_of_steps_of_permutation[y1]]
    #Step4: get overall P matrix from P_row_index_storage_matrix 
    P_matrix = np.zeros((num_of_Y,num_of_Y))
    for y_index in range(len(P_row_index_storage_matrix)):
        P_matrix[y_index][P_row_index_storage_matrix[y_index]] = 1
    #step6: (optional) get D matrix by normalizing the pivots of U to 1
    if get_D_or_not:
        D_matrix = np.zeros((num_of_Y,num_of_X))
        for y_index in range(num_of_Y):
            pivot_normalized_factor = matrix[y_index][y_index]
            D_matrix[y_index][y_index] = pivot_normalized_factor
            matrix[y_index] = matrix[y_index]/ pivot_normalized_factor #the denominator is because the location of the pivot is y_index = x_index, kind of propagation
    #Step7: check permutation_parity
    if permutation_parity % 2 == 0:
        permutation_parity = 1
    elif permutation_parity % 2 == 1:
        permutation_parity = -1
    if verbose:
        print("The P matrix is:")
        print_matrix(P_matrix)  
        print("The L matrix is:")
        print_matrix(L_matrix)
        if get_D_or_not:
            print("The D matrix is:")
            print_matrix(D_matrix)
        print("The U matrix is:")
        print_matrix(matrix)
        if get_D_or_not:
            print ("Check whether the PLDU factorization is correct or not:")
            print("P*L*D*U is:")
            print_matrix(np.dot(np.dot(np.dot(P_matrix, L_matrix), D_matrix), matrix))
        else:
            print ("Check whether the PLU factorization is correct or not:")
            print("P*L*U is:")
            print_matrix(np.dot(np.dot(P_matrix, L_matrix), matrix))
        print(f"The permutation parity is: {permutation_parity}")
    if get_D_or_not:
        return [P_matrix, L_matrix, D_matrix, matrix, permutation_parity]
    else:
        return [P_matrix, L_matrix, matrix, permutation_parity]

def get_projection_vector(A: np.ndarray, vector_b: np.ndarray, verbose = False) -> np.ndarray: #project b onto A
    #P = A.(A^T.A)-1.A^T
    # A: mxn where m > n
    middle = np.linalg.inv(np.dot(np.transpose(A),A))
    P = np.dot(A,np.dot(middle,np.transpose(A)))
    vector_p = np.dot(P,vector_b)
    if verbose:
        print("The projection vector is:")
        print_matrix(vector_p)
        print("The error vector is:")
        error = vector_b - vector_p #error = b - p
        print_matrix(error)
    return vector_p
def get_projection_matrix(A): # A contains independent columns that span the subspace where a vector is projected onto
   # P = A.(A^T.A)-1.A^T
   ##get (A^T.A)-1
   middle = np.linalg.inv(np.dot(np.transpose(A),A))
   P = np.dot(np.dot(A,middle),np.transpose(A))
   return P     
def get_least_square_solution(A: np.ndarray, vector_b: np.ndarray, verbose = False) -> np.ndarray: #get x* #A: mxn where m > n
    '''
    There are 2 cases, A^T.A is invertible and not invertible
    '''
    ##CASE 1: A^T.A is invertible
    #x = (A^T.A)-1 . A^T.b in C(A^T)
    try:
        left_part = np.linalg.inv(np.dot(np.transpose(A),A))
        right_part = np.dot(np.transpose(A),vector_b)
        x = np.dot(left_part,right_part)
        if verbose:
            print("Case1: The least square solution is:")
            for i in range(len(x)):
                print(f"X{i+1} = {round(x[i][0],3)}")
        return x
    except np.linalg.LinAlgError:
        ##CASE 2: A^T.A is not invertible
        #Find x in A^T.A.x = A^T.b in C(A^T)
        #Step1: get A^T.A
        AtA = np.dot(np.transpose(A),A)
        #Step2: get Anp.^T.b
        Atb = np.dot(np.transpose(A),vector_b)
        #Step3: Append A^tb to A^tA to form an augmented matrix [A^tA A^tb] then RREF it
        RREF_matrix = rref(np.column_stack((AtA, Atb)))
        #Step4: initialize the solution vector x by 0 
        x = np.zeros((len(AtA[0]),1))
        #Step6: Scan for pivot columns.
        '''
        The idea for step 6 is set all entries of 
        col-vector-solution x hat 
        whose y_index matches the pivot row (y_index*) of RREF_matrix 
        to be equal to RREF_matrix[y_index*][last_x_index_of_RREF_matrix] or the "A^tb" part of RREF([A^tA A^tb]) 
        '''
        num_of_Y = len(AtA)
        num_of_X = len(AtA[0]) 
        last_x_index_of_RREF_matrix = len(RREF_matrix[0]) - 1
        all_0_row = False
        for y_index in range(num_of_Y):
            #check whether the current row is all 0 before finding the pivot column  
            all_0_row = all(x == 0 for x in RREF_matrix[y_index]) # not necessary to check the last x ( containing vector A^tb) like RREF_matrix[y_index][:last_x_index_of_RREF_matrix]). The reason is A^tb lies in the C(A^T) so A^tb will have same row 0 as A^T.A when RREF takes place
            if all_0_row == True:
                break 
            #if not, then proceed
            for x_index in range(num_of_X):
                if RREF_matrix[y_index][x_index] != 0:
                    x[y_index][0]= RREF_matrix[y_index][last_x_index_of_RREF_matrix]
                    break #break the inner loop if successfully find one solution entry 
        if verbose:
            print("Case2: The least square solution is:")
            for i in range(len(x)):
                print(f"X{i+1} = {round(x[i][0],3)}")    
        return x

def gauss_eliminate(matrix: np.ndarray, max_x_index_to_stop_eliminate: int = None, stop_if_found_no_pivot_for_a_column: bool = False, verbose: bool = False) -> np.ndarray | None:
    #information of the matrix
    matrix = copy.deepcopy(matrix)
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    last_x_index = num_of_X - 1
    if max_x_index_to_stop_eliminate == None or max_x_index_to_stop_eliminate > last_x_index: #None means users want to eliminate all columns, and the case > last_x_index is to prevent out of range error
       max_x_index_to_stop_eliminate = num_of_X-1 
    ## Clear the entries below the pivot
    for y_index in range(num_of_Y-1): # no need to eliminate the last row
        for x_index in range(y_index, max_x_index_to_stop_eliminate + 1):  
            eli_matrix = _get_elimination_matrix_for_a_column(matrix, x_index, y_index, "down", verbose) # "down" means to clear y_entries on particualr x_index down the current y_index
            if eli_matrix is None: # if none, skip to next x
                if stop_if_found_no_pivot_for_a_column: # recommended for invertibility checking and solving a system of equations
                    print("The matrix is singular") if verbose else None
                    return None
                continue # end current iteration and skip to the next iteration, which is next x
            #check if eli_matrix == identity matrix, if yes, then skip to the next x
            elif np.array_equal(eli_matrix, np.identity(num_of_Y)):
                break # if it is true, it means there is all 0 below the pivot, then increment y to the next y
            elif eli_matrix is not None:
                #after this, we get eli_matrix
                matrix = np.dot(eli_matrix, matrix)
                #ADDITIONAL PART FOR BETTER NUMERICAL STABILITY: ROUND EXTREME SMALL VALUES e^-10 TO 0
                #matrix = round_small_values_to_zero(matrix)
                if verbose: #print the step-by-step process for educational purposes if allowed
                    print(f"the elimination matrix for the {x_index+1}th collumn:")
                    print_matrix(eli_matrix)
                    print(f"the result:")
                    print_matrix(matrix)
                break # if pivot is found in the loop, then stop and increment y to the next y
    return matrix 

def jordan_eliminate(matrix: np.ndarray, max_x_index_to_stop_eliminate: int = None ,verbose: bool = False) -> np.ndarray: #Warning: only used after Gaussian Elimination is performed
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    max_rank = min(num_of_Y, num_of_X)
    if max_x_index_to_stop_eliminate == None or max_x_index_to_stop_eliminate > num_of_X: #None means users want to eliminate all columns, and the case > last_x_index is to prevent out of range error
       max_x_index_to_stop_eliminate = num_of_X-1
    #increment by 1 from second column to the last column specified by max_x_index_to_stop_eliminate
    for y_index in range (1, num_of_Y): 
        for x_index in range (max_x_index_to_stop_eliminate + 1):
            #check from left to right to find the first non-0 pivot
            # if found, then clear the entries above the pivot
            # if not found on the whole row or y_index, then increment to the next y_index
            if matrix[y_index][x_index] != 0:
                eli_matrix = _get_elimination_matrix_for_a_column(matrix, x_index, y_index, "up") # "up" means to clear y_entries on particualr x_index up the current y_index
                #check if eli_matrix == identity matrix, if yes, then skip to the next y
                if np.array_equal(eli_matrix, np.identity(num_of_Y)):
                    break
                matrix = np.dot(eli_matrix, matrix)
                if verbose: 
                     print(f"Jordan Step: the elimination matrix for the {x_index+1}th collumn: ")
                     print_matrix(eli_matrix)
                     print(f"Jordan Step: the result: ")
                     print_matrix(matrix)
                break #break the inner loop if the pivot is found
        if (y_index+1) == max_rank: #if the max rank is reached, then break the outer loop for saving computation
            break
    return matrix

def _find_non0_pivot(matrix: np.ndarray, x_index: int, y_starting_index: int): 
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    for y_index in range(y_starting_index + 1,num_of_Y): #plus 1 because we are looking for the next lower Y after finding the 0 pivot of the current Y
        if matrix[y_index][x_index] != 0:
            return y_index
        y_index+=1
    #otherwise, return None if there is no non-0 pivot
    return None
def _get_elimination_matrix_for_a_column(matrix: np.ndarray, x_index: int, y_index: int, Up_or_Down_the_current_y_index: str, verbose: bool = False) -> np.ndarray | None: #eliminate the y_index on the particular x_index
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    #STEP1: initialize the elimination matrix
    eli_matrix = np.identity(num_of_Y)
    #STEP2: find the largest pivot and detect whether the column is all 0 or not
    y_index_of_largest_pivot = _find_largest_pivot_as_well_as_non_zero_pivot(matrix, x_index, y_index)
    if y_index_of_largest_pivot == y_index:
        pass
    elif y_index_of_largest_pivot != None:
        matrix[[y_index,y_index_of_largest_pivot]] = matrix[[y_index_of_largest_pivot,y_index]] # swap the rows
        if verbose == True:
               print(f"Partial Pivoting: Exchange row {y_index+1} with row {y_index_of_largest_pivot+1}:")
               print_matrix(matrix)
    elif y_index_of_largest_pivot == None:
        return None
    #STEP3: elimination process for each y down the current y or up the current y
    if Up_or_Down_the_current_y_index.lower() == "down":
            multiplier = 1 #default
            y_index_of_current_pivot = y_index
            current_y_index = y_index + 1#plus 1 because we do not need to eliminate the current row where the current pivot is located
            while current_y_index < num_of_Y:
                multipler = -(matrix[current_y_index][x_index])/(matrix[y_index_of_current_pivot][x_index])#
                eli_matrix[current_y_index][y_index_of_current_pivot] = multipler 
                current_y_index+=1
    elif Up_or_Down_the_current_y_index.lower() == "up":
            multiplier = 1 #default
            y_index_of_current_pivot = y_index
            current_y_index = y_index -1 #- 1 because we do not need to eliminate the current row where the current pivot is located
            while current_y_index >= 0:
                multipler = -(matrix[current_y_index][x_index])/(matrix[y_index_of_current_pivot][x_index])#
                eli_matrix[current_y_index][y_index_of_current_pivot] = multipler 
                current_y_index-=1
    return eli_matrix
def _find_largest_pivot_as_well_as_non_zero_pivot(matrix: np.ndarray, x_index: int, y_starting_index: int):
    # Information of the matrix
    num_of_Y = len(matrix)
    
    # Main algorithm
    largest_pivot = matrix[y_starting_index][x_index]
    y_index_of_largest_pivot = y_starting_index
    
    for y_index in range(y_starting_index+1, num_of_Y):
        pivot_value = abs(matrix[y_index][x_index]) 
        # Check if this is the largest pivot found so far
        if pivot_value > largest_pivot:
            largest_pivot = pivot_value
            y_index_of_largest_pivot = y_index    
    # If no non-zero pivot was found, return None
    if largest_pivot == 0:
        return None
    
    return y_index_of_largest_pivot

def print_matrix(matrix: np.ndarray) -> None:
    print("////////////////////////////////////////")
    #round numpy array to 3 decimal places for better readability but not actually cutting off the values
    matrix = np.round(matrix,3)
    for row in matrix:
        print(row)
    return


if __name__ == "__main__":
    main()