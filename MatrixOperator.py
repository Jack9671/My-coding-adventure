import random
import math
import copy
import numpy as np
import mpmath as mp
def main():
    print("This is the main function")
    #test get_least_square_solution
    A = [[1,1],
         [1,1]]
    b = [[2],[2]]
    x = get_least_square_solution(A, b, verbose = True)
    
    p = get_projection_vector(A, b, verbose = True)
    #check if Ax = p
    print(multiply(A,x))
    print(p)

  




     
def add(matrix1: list[list[int]], matrix2: list[list[int]]) -> list[list[int]]:
    #information of the matrix
    num_of_Y_matrix1 = len(matrix1) 
    num_of_X_matrix1 = len(matrix1[0])
    # matrix2 is an arbitrary matrix
    num_of_Y_matrix2 = len(matrix2)
    num_of_X_matrix2 = len(matrix2[0])
    #check whether the matrix1 and matrix2 are compatible for addition
    if num_of_Y_matrix1 != num_of_Y_matrix2 or num_of_X_matrix1 != num_of_X_matrix2:
        print("The matrices are not compatible for addition")
        return None
    #main algorithm
    for y_index in range(num_of_Y_matrix1):
        for x_index in range (num_of_X_matrix1):
            matrix1[y_index][x_index] = matrix1[y_index][x_index] + matrix2[y_index][x_index]
    return matrix1
def subtract(matrix1: list[list[int]], matrix2: list[list[int]]) -> list[list[int]]:
    #information of the matrix
    num_of_Y_matrix1 = len(matrix1) 
    num_of_X_matrix1 = len(matrix1[0])
    # matrix2 is an arbitrary matrix
    num_of_Y_matrix2 = len(matrix2)
    num_of_X_matrix2 = len(matrix2[0])
    #check whether the matrix1 and matrix2 are compatible for subtraction
    if num_of_Y_matrix1 != num_of_Y_matrix2 or num_of_X_matrix1 != num_of_X_matrix2:
        print("The matrices are not compatible for subtraction")
        return None
    #main algorithm
    for y_index in range(num_of_Y_matrix1):
        for x_index in range (num_of_X_matrix1):
            matrix1[y_index][x_index] = matrix1[y_index][x_index] - matrix2[y_index][x_index]
    return matrix1
def multiply(matrix2: list[list[int]], matrix1: list[list[int]]) -> list[list[int]]:
    ## information of the matrix
    # matrix1 is the original matrix
    num_of_Y_matrix1 = len(matrix1) 
    num_of_X_matrix1 = len(matrix1[0])
    # matrix2 is an arbitrary matrix
    num_of_Y_matrix2 = len(matrix2)
    num_of_X_matrix2 = len(matrix2[0])
    ### Side step: Check whether exclusively matrix1 or matrix2 is a scalar
    if num_of_Y_matrix1 == 1 and num_of_X_matrix1 == 1: #matrix1 is a scalar
        return scale(scalar = matrix1[0][0], matrix = matrix2)
    elif num_of_Y_matrix2 == 1 and num_of_X_matrix2 == 1: #matrix2 is a scalar
        return scale(scalar = matrix2[0][0] , matrix = matrix1)
    ## main algorithm
    #check whether the matrix1 and matrix2 are compatible for multiplication
    if num_of_Y_matrix1 != num_of_X_matrix2:
        print("The matrices are not compatible for multiplication")
        return None
    # matrix3 is the result and structure of matrix3 is the same as matrix1
    matrix3 = [[0 for _ in range(num_of_X_matrix1)] for _ in range(num_of_Y_matrix2)] # m2m1 = m3 or Am1 = m3
    #the result of dot product is placed onto the matrix3 at (y_index,x_index) 
    for x_index in range(num_of_X_matrix1):
            for y_index in range(num_of_Y_matrix2):
                vector_YCoefficients = read_column_entries(matrix1, x_index)
                vector_XCoefficients = read_row_entries(matrix2, y_index)
                matrix3[y_index][x_index] = dot_product(vector_XCoefficients, vector_YCoefficients)
    return matrix3
def scale(scalar: int ,matrix: list[list[int]],) -> list[list[int]]:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm
    for y_index in range(num_of_Y):
        for x_index in range(num_of_X):
            matrix[y_index][x_index] = matrix[y_index][x_index] * scalar
    return matrix
def dot_product(col_vector1: list[list[int]], col_vector2: list[list[int]]) -> list[list[int]]: # linear algebra approach A^T * A  
    #information of the column vectors
    num_of_Y_col_vector1 = len(col_vector1)
    num_of_Y_col_vector2 = len(col_vector2)
    #check whether the column vectors are compatible for dot product
    if num_of_Y_col_vector1 != num_of_Y_col_vector2:
        print("The column vectors are not compatible for dot product")
        return None
    #main algorithm
    dot_product = 0
    for y_index in range(num_of_Y_col_vector1):
        dot_product += col_vector1[y_index][0] * col_vector2[y_index][0]
    return dot_product

def rref(matrix: list[list[int]], verbose: bool = False) -> list[list[int]] | None: #Reduced Row Echelon Form
    #information of the matrix [A]
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm: Gauss-Jordan elimination
    #STEP1: Gaussian Elimination
    matrix = gauss_eliminate(matrix, verbose)
    #STEP2: Jordan Elimination
    matrix = jordan_eliminate(matrix, verbose)
    #STEP3: Normalize the pivots to 1 to get R
    for y_index in range(num_of_Y): 
        for x_index in range(y_index, num_of_X): #check from left to right to find the non-0 pivot of upper or semi-upper triangular matrix
            if matrix[y_index][x_index] != 0:
                pivot_normalized_factor = matrix[y_index][x_index]
                for x_index in range(x_index, num_of_X): # execute the normalization process
                    matrix[y_index][x_index] = matrix[y_index][x_index]/ pivot_normalized_factor #the denominator is because the location of the pivot is y_index = x_index, kind of propagation
                break #After normalizing the current y_index, break the inner loop to move to the next y_index    
    if verbose:
        print("After normalization, R is:")
        print_matrix(matrix)
    #STEP6: Return the R matrix
    return matrix

def pldu_factor(matrix: list[list[int]],get_D_or_not: bool = False, verbose: bool = False) -> list[list[list[int]]] | None:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm
    #Step1: Initialize the L matrix and a column matrix P containing the index of the row
    L_matrix = get_identity_matrix(num_of_Y)
    P_row_index_storage_matrix = [y for y in range(num_of_Y)]
    Arr_of_steps_of_permutation = [0 for y in range(num_of_Y)]
    #Step2: Modified Version of Gaussian Elimination for purpose of finding P and L
    ## The idea is to transform the matrix into an upper triangular matrix
    for x_index in range(num_of_X-1):  # note: customizable or we do not need to find the pivot of the last column of a square matrix
        #main algorithm of getting elimination matrix
        eli_matrix = [[0 for _ in range(num_of_Y)] for _ in range(num_of_Y)]
        #STEP1: transform eli_matrix into an identity matrix
        for common_index in range(num_of_Y):
            eli_matrix[common_index][common_index] = 1
        y_index = x_index
        #STEP2: check whether current pivot matrix[x_index][x_index] is 0 or not, if it is 0, exchange and ready for elimination
        if matrix[y_index][x_index] == 0: #if the current y is 0, then exchange the y with the next y that has non-0 pivot
           y_index_of_non0_pivot = find_non0_pivot(matrix, x_index, y_index)
           if y_index_of_non0_pivot != None:
               exchange_rows(matrix, y_index, y_index_of_non0_pivot)
               exchange_rows(L_matrix, y_index, y_index_of_non0_pivot)
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
        matrix = multiply(eli_matrix, matrix)
    #check if the last y_index contains all 0 after elimination, if yes, then the system is singular
    if all(x == 0 for x in matrix[num_of_Y-1]):
        print("There exists a last all-0 row during Gaussian Elimination") if verbose else None
        return None
    #Step3: Set P_row_index_storage_matrix from Arr_of_steps_of_permutation
    for y1 in (len(Arr_of_steps_of_permutation)-1,0,-2):
        y2 = y1 -1
        exchange_rows(P_row_index_storage_matrix, Arr_of_steps_of_permutation[y1], Arr_of_steps_of_permutation[y2])#
    #Step4: get overall P matrix from P_row_index_storage_matrix 
    P_matrix = get_identity_matrix(num_of_Y)
    for y_index in range(len(P_row_index_storage_matrix)):
        P_matrix[y_index][P_row_index_storage_matrix[y_index]] = 1
    #step6: (optional) get D matrix by normalizing the pivots of U to 1
    if get_D_or_not:
        D_matrix = [[0 for _ in range(num_of_X)] for _ in range(num_of_Y)]
        for y_index in range(num_of_Y):
            pivot_normalized_factor = matrix[y_index][y_index]
            D_matrix[y_index][y_index] = pivot_normalized_factor
            for x_index in range(y_index,num_of_X):
                matrix[y_index][x_index] = matrix[y_index][x_index]/ pivot_normalized_factor #the denominator is because the location of the pivot is y_index = x_index, kind of propagation
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
            print_matrix(multiply(multiply(multiply(P_matrix, L_matrix), D_matrix), matrix))
        else:
            print ("Check whether the PLU factorization is correct or not:")
            print("P*L*U is:")
            print_matrix(multiply(multiply(P_matrix, L_matrix), matrix))
    return [P_matrix, L_matrix, matrix]
def power(matrix: list[list[int]], power: int, verbose: bool = False) -> list[list[int]] | None:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm
    result_matrix = copy.deepcopy(matrix)   
    for num_of_multiplication in range(2 , power+1): 
       result_matrix = multiply(matrix, result_matrix)
       if verbose:
         print(f"The matrix to the power of {num_of_multiplication} is:")
         print_matrix(result_matrix)
    return result_matrix 

def transpose(matrix: list[list[int]], verbose: bool = False) -> list[list[int]]:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm
    #Step1: Initialize the transpose matrix
    transposed_matrix = init_matrix(num_of_X,num_of_Y)
    #Step 2: transpose_matrix[x][y] = matrix[y][x]
    for y_index in range(num_of_Y):
        for x_index in range(num_of_X):
            transposed_matrix[x_index][y_index] = matrix[y_index][x_index]
    if verbose:
        print("The transpose matrix is:")
        print_matrix(transposed_matrix)
    return transposed_matrix   
def invert(matrix: list[list[int]], verbose: bool = False) -> list[list[int]] | None:
    #information of the matrix [A]
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    #main algorithm: Gauss-Jordan elimination
    ##note: all operations are based on the matrix A not the augmented matrix [A I]
    ##STEP1: create an augmented matrix [A I]
    I = get_identity_matrix(num_of_Y) #can be num_of_X as well
    #append I to A to form an augmented matrix
    augmented_matrix = [row_A + row_I for row_A, row_I in zip(matrix, I)]

    if verbose == True:
        print("[A I] is:")
        print_matrix(augmented_matrix)
    #STEP2: Gaussian Elimination
    augmented_matrix = gauss_eliminate(augmented_matrix, num_of_X-2, verbose)
    if augmented_matrix == None: return None
    #STEP3: Jordan Elimination
    augmented_matrix = jordan_eliminate(augmented_matrix, verbose)
    '''
    x_index = num_of_X - 1 #-1 because the index starts from 0
    y_index = 0
    while x_index > 0: #not >= 0 because we do not need to upwardly clear the first y_index 
        eli_matrix = get_elimination_matrix_for_a_column(augmented_matrix, x_index, x_index, "up") # "up" means to clear y_entries on particualr x_index up the current y_index
        augmented_matrix = multiply(eli_matrix, augmented_matrix)
        if verbose:
            print(f"Jordan Step: the elimination matrix for the {x_index+1}th collumn: ")
            print_matrix(eli_matrix)
            print(f"Jordan Step: the result: ")
            print_matrix(augmented_matrix)
        x_index-=1 
    '''
    #STEP5: Normalize the pivot to 1 to get [ I A^-1]
    for y_index in range(num_of_Y):
        pivot_normalized_factor = augmented_matrix[y_index][y_index] if augmented_matrix[y_index][y_index] != 0 else 1 
        for x_index in range(y_index, len(augmented_matrix[0])): # num_of_X of the augmented matrix
            augmented_matrix[y_index][x_index] = augmented_matrix[y_index][x_index]/ pivot_normalized_factor #the denominator is because the location of the pivot is y_index = x_index, kind of propagation
    if verbose:
        print("After normalization, [I A^-1] is:")
        print_matrix(augmented_matrix)
    #STEP6: EXTRACT the A^-1 from the augmented matrix [I A^-1] 
    A_inverse = []
    y_index = 0
    first_x_index_of_A_inverse = num_of_X
    while y_index < num_of_Y:
        A_inverse.append(augmented_matrix[y_index][first_x_index_of_A_inverse:])
        y_index+=1
    if verbose:
        print("A^-1 is:")
        print_matrix(A_inverse)
    return A_inverse
def get_projection_vector(A: list[list[int]], vector_b: list[list[int]], verbose = False) -> list[list[int]]: #project b onto A
    #P = A.(A^T.A)-1.A^T
    # A: mxn where m > n
    middle = invert(multiply(transpose(A),A))
    P = multiply(A,multiply(middle,transpose(A)))
    vector_p = multiply(P,vector_b)
    if verbose:
        print("The projection vector is:")
        print_matrix(vector_p)
        print("The error vector is:")
        error = subtract(vector_b, vector_p) #error = b - p
        print_matrix(error)
    return vector_p
def get_projection_matrix(A): # A contains independent columns that span the subspace where a vector is projected onto
   # P = A.(A^T.A)-1.A^T
   ##get (A^T.A)-1
   middle = invert(multiply(transpose(A),A))
   P = multiply(multiply(A,middle),transpose(A))
   return P     
def get_least_square_solution(A: list[list[int]], vector_b: list[list[int]], verbose = False) -> list[list[int]]: #get x* #A: mxn where m > n
    '''
    There are 2 cases, A^T.A is invertible and not invertible
    '''
    ##CASE 1: A^T.A is invertible
    #x = (A^T.A)-1 . A^T.b in C(A^T)
    left_part = invert(multiply(transpose(A),A))
    if left_part != None:
        right_part = multiply(transpose(A),vector_b)
        x = multiply(left_part,right_part)
        if verbose:
            print("Case1: The least square solution is:")
            for i in range(len(x)):
                print(f"X{i+1} = {round(x[i][0],3)}")
        return x
    ##CASE 2: A^T.A is not invertible

    #Find x in A^T.A.x = A^T.b in C(A^T)
    #Step1: get A^T.A
    AtA = multiply(transpose(A),A)
    #Step2: get A^T.b
    Atb = multiply(transpose(A),vector_b)
    #Step3: Append A^tb to A^tA to form an augmented matrix [A^tA A^tb]
    y_index = 0 
    while y_index < len(AtA): # can be len(Atb) as well
        AtA[y_index].append(Atb[y_index][0])
        y_index+=1
    #Step4: RREF the augmented matrix [A^tA A^tb]
    RREF_matrix = rref(AtA)
    #Step5: initialize the solution list with length equal to len(AtA[0])-1 because index starts from 0
    x = [[0] for _ in range(len(AtA[0])-1)]
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
    y_index = 0
    while y_index < num_of_Y:
        #check whether the current row is all 0 before finding the pivot column  
        all_0_row = all(x == 0 for x in RREF_matrix[y_index]) # not necessary to check the last x ( containing vector A^tb) like RREF_matrix[y_index][:last_x_index_of_RREF_matrix]). The reason is A^tb lies in the C(A^T) so A^tb will have same row 0 as A^T.A when RREF takes place
        if all_0_row == True:
            break 

        #if not, then proceed
        x_index = 0
        while x_index < num_of_X:
            if RREF_matrix[y_index][x_index] != 0:
                x[y_index][0]= RREF_matrix[y_index][last_x_index_of_RREF_matrix]
                break
            x_index+=1
        y_index+=1
    if verbose:
        print("Case2: The least square solution is:")
        for i in range(len(x)):
            print(f"X{i+1} = {round(x[i][0],3)}")    
    return x
def get_polynomial_regression(points: list[list[int]], degree: int, verbose = False) -> list[list[int]]: #points = [[x1,y1],[x2,y2],...,[xn,yn]] 
    #Warning: numerical issue is propotional to degree and number of points
    ##Step1: Construct entries of A^T.A as well as A^T.b
    sums_of_x_nth_degree = np.zeros(2 * degree +1, float)  #include 0th degree, so +1
    sums_of_x_nth_degree[0] = len(points) # a11 = m of A: mxn
    sums_of_y_nth_degree = np.zeros(degree + 1, float) #include 0th degree, so +1
    nth_x = 0
    # Update summation of x^n as well as y*x^n
    while nth_x < len(points):
         sums_of_x_nth_degree[1:] += points[nth_x][0] ** np.arange(1, len(sums_of_x_nth_degree) ) 
         sums_of_y_nth_degree += points[nth_x][1] * (points[nth_x][0] ** np.arange(0, len(sums_of_y_nth_degree) ))
         nth_x += 1
    #transform the numpy array to list
    sums_of_x_nth_degree = sums_of_x_nth_degree.tolist()
    sums_of_y_nth_degree = sums_of_y_nth_degree.tolist()
    ##Step2: Construct A^T.A and A^T.b
    AtA = [[0 for _ in range(degree+1)] for _ in range(degree+1)]
    Atb = [[0] for _ in range(degree+1)]
    #scan row by row
    y_index = 0
    increment = 0
    while y_index < len(AtA): #can be len(Atb) as well
        Atb[y_index][0] = sums_of_y_nth_degree[y_index] #get A^T.b by just outer loop
        #then we get A^T.A by another inner loop
        x_index=0
        while x_index < len(AtA[0]):
            AtA[y_index][x_index] = sums_of_x_nth_degree[x_index + increment]
            x_index+=1
        increment +=1
        y_index +=1
    #Step3: Append A^tb to A^tA to form an augmented matrix [A^tA A^tb]
    y_index = 0 
    while y_index < len(AtA): # can be len(Atb) as well
        AtA[y_index].append(Atb[y_index][0])
        y_index+=1
    #Step4: RREF the augmented matrix [A^tA A^tb]
    RREF_matrix = rref(AtA)
    #Step5: initialize the solution list with length equal to len(AtA[0])-1 because index starts from 0
    coefs = [[0] for _ in range(len(AtA[0])-1)]
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
    y_index = 0
    while y_index < num_of_Y:
        #check whether the current row is all 0 before finding the pivot column  
        all_0_row = all(x == 0 for x in RREF_matrix[y_index]) # not necessary to check the last x ( containing vector A^tb) like RREF_matrix[y_index][:last_x_index_of_RREF_matrix]). The reason is A^tb lies in the C(A^T) so A^tb will have same row 0 as A^T.A when RREF takes place
        if all_0_row == True:
            break 
        #if not, then proceed
        x_index = 0
        while x_index < num_of_X:
            if RREF_matrix[y_index][x_index] != 0:
                coefs[y_index][0]= RREF_matrix[y_index][last_x_index_of_RREF_matrix]
                break
            x_index+=1
        y_index+=1
    if verbose:
        print("The least square solution is:")
        for i in range(len(coefs)):
            print(f"X{i+1} = {round(coefs[i][0],3)}")
        #not optimized yet
        absolute_sum_of_error = 0
        length_of_error_vector = 0
        y_index = 0
        while y_index < len(points):
            nth_degree =0
            actual_y = points[y_index][1] #get current b at current y_index
            expected_y = 0
            while nth_degree <= degree: #get current Ax* at current y_index
                expected_y +=  (coefs[nth_degree][0] * (points[y_index][0] ** nth_degree))
                nth_degree += 1
            current_e = actual_y - expected_y
            #sum of error e += |b - Ax*|
            absolute_sum_of_error +=  abs(current_e)
            #length of error vector e += (b - Ax*)^2
            length_of_error_vector += current_e**2
            y_index += 1
        print(f"the absolute sum of error is: {absolute_sum_of_error}")
        print(f"The length of error vector is: {length_of_error_vector}")
    return coefs
def gauss_eliminate(matrix: list[list[int]], max_x_index_to_stop_eliminate: int = None, stop_if_found_no_pivot_for_a_column: bool = False, verbose: bool = False) -> list[list[int]] | None:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    last_y_index = num_of_Y - 1  
    last_x_index = num_of_X - 1  
    if max_x_index_to_stop_eliminate == None or max_x_index_to_stop_eliminate > last_x_index: #None means users want to eliminate all columns, and the case > last_x_index is to prevent out of range error
       max_x_index_to_stop_eliminate = num_of_X-1 
    ## Clear the entries below the pivot
    x_index = 0
    y_index = 0
    while y_index <= last_y_index-1: # no need to eliminate the last row
        x_index = y_index
        while x_index <= max_x_index_to_stop_eliminate:  
            eli_matrix = get_elimination_matrix_for_a_column(matrix, x_index, y_index, "down", verbose) # "down" means to clear y_entries on particualr x_index down the current y_index
            if eli_matrix == None: # if none, skip to next x
                if stop_if_found_no_pivot_for_a_column: # recommended for invertibility checking and solving a system of equations
                    print("The matrix is singular") if verbose else None
                    return None
                x_index+=1
                continue # end current iteration and skip to the next iteration, which is next x
            elif eli_matrix != None:
                #after this, we get eli_matrix
                matrix = multiply(eli_matrix, matrix)
                #ADDITIONAL PART FOR BETTER NUMERICAL STABILITY: ROUND EXTREME SMALL VALUES e^-10 TO 0
                matrix = round_small_values_to_zero(matrix)

                if verbose: #print the step-by-step process for educational purposes if allowed
                    print(f"the elimination matrix for the {x_index+1}th collumn:")
                    print_matrix(eli_matrix)
                    print(f"the result:")
                    print_matrix(matrix)
                break # if pivot is found in the loop, then stop and increment y to the next y
        y_index+=1
    return matrix 
def jordan_eliminate(matrix: list[list[int]], max_x_index_to_stop_eliminate: int ,verbose: bool = False) -> list[list[int]]: #Warning: only used after Gaussian Elimination is performed
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    last_y_index = num_of_Y - 1
    last_x_index = num_of_X - 1
    y_index = 1 # increment by 1 from second column to the last column specified by max_x_index_to_stop_eliminate
    while y_index <= max_x_index_to_stop_eliminate: 
        x_index = 0
        while x_index <= last_x_index:
            #check from left to right to find the first non-0 pivot
            # if found, then clear the entries above the pivot
            # if not found on the whole row or y_index, then increment to the next y_index
            if matrix[y_index][x_index] != 0:
                eli_matrix = get_elimination_matrix_for_a_column(matrix, x_index, y_index, "up") # "up" means to clear y_entries on particualr x_index up the current y_index
                matrix = multiply(eli_matrix, matrix)
                if verbose: 
                     print(f"Jordan Step: the elimination matrix for the {x_index+1}th collumn: ")
                     print_matrix(eli_matrix)
                     print(f"Jordan Step: the result: ")
                     print_matrix(matrix)
                break #break the inner loop if the pivot is found
            elif matrix[y_index][x_index] == 0:
                x_index+=1
        y_index+=1
    return matrix

def gauss_eliminate_for_solve_sys_of_equations(matrix: list[list[int]], max_x_index_to_stop_geting_pivot: int, verbose: bool = False) -> list[list[int]] | None:
    #information of the matrix
    num_of_Y = len(matrix)
    num_of_X = max_x_index_to_stop_geting_pivot
    ## The idea is to transform the matrix into an upper triangular matrix
    x_index = 0
    y_index = 0
    while x_index <= max_x_index_to_stop_geting_pivot:  # note: customizable or we do not need to find the pivot of the last column or the last 2 columns if it is an augmented matrix
        eli_matrix = get_elimination_matrix_for_a_column(matrix, x_index, x_index, "down", verbose) # "down" means to clear y_entries on particualr x_index down the current y_index
        if eli_matrix == None:
                print("There exists a zero row during Gaussian Elimination") if verbose else None
                return None
        else:
                matrix = multiply(eli_matrix, matrix)
                if verbose: #print the step-by-step process for educational purposes if allowed
                    print(f"the elimination matrix for the {x_index+1}th collumn:")
                    print_matrix(eli_matrix)
                    print(f"the result:")
                    print_matrix(matrix)
        x_index+=1
    #check if the last y_index contains all 0 after elimination, if yes, then the system is singular
    if all(x == 0 for x in matrix[num_of_Y-1][:max_x_index_to_stop_geting_pivot+2]):
        print("There exists a last all-0 row during Gaussian Elimination") if verbose else None
        return None
    return matrix 
    
def solve_system_of_equations(A: list[list[int]], b: list[list[int]] , verbose: bool = False) -> list[list[int]] | None:
    Augmented_matrix = [row + col for row, col in zip(A, b)] #combine A and b to form an augmented matrix
    ###information of the augmented matrix
    num_of_Y = len(Augmented_matrix)
    num_of_X = len(Augmented_matrix[0])
    last_y_index= num_of_Y - 1
    last_x_index = num_of_X - 1
    ###main algorithm
    ##STEP1: Gaussian Elimination
    Augmented_matrix = gauss_eliminate(Augmented_matrix,last_x_index-1, stop_if_found_no_pivot_for_a_column = True, verbose=verbose) 
    if Augmented_matrix == None: return None 
    ##STEP2: normalize pivot to 1
    y_index = 0
    while y_index <= last_y_index:
        x_index = y_index #y_index = x_index propagation
        pivot_normalized_factor = Augmented_matrix[y_index][y_index]
        while x_index <= last_x_index:
            Augmented_matrix[y_index][x_index] = Augmented_matrix[y_index][x_index]/ pivot_normalized_factor #the denominator is because the location of the pivot is y_index = x_index, kind of propagation
            x_index+=1
        y_index+=1
    if verbose:
        print("After normalization, [A b] is:")
        print_matrix(Augmented_matrix)
    ## STEP3: solve by back substitution
    #note: the augmented matrix, after undergoing back substitution, has no meaning except for the last x_index(collumn) that contains solutions  
    initial_last_x_index = last_x_index
    initial_last_y_index = last_y_index
    current_last_x_index = initial_last_x_index - 1 # -1 because last x_index is the ouput part b, that is Ax = b, so the last x_index is b
    current_last_y_index = initial_last_y_index 
    while current_last_x_index >= 0 or current_last_y_index >= 0: # either case, does not matter
        y_index = current_last_y_index
        while y_index >= 0: # yIndex decrement by 1
            Augmented_matrix[y_index][current_last_x_index] = Augmented_matrix[y_index][current_last_x_index] * Augmented_matrix[current_last_y_index][initial_last_x_index] # multiply the pivot=1 by the last X in the matrix
            if y_index < current_last_y_index:
                Augmented_matrix[y_index][initial_last_x_index] = Augmented_matrix[y_index][initial_last_x_index] - Augmented_matrix[y_index][current_last_x_index]#
            y_index-=1
        current_last_y_index -= 1
        current_last_x_index -= 1
    #STEP4: return the result of last x_index, which contains solutions of the equation
    arr_of_solutions = [[0] for _ in range(num_of_Y)]
    x_index = last_x_index
    y_index = 0
    while y_index <= last_y_index:
        arr_of_solutions[y_index][0] = Augmented_matrix[y_index][x_index]
        y_index += 1
    if verbose:
        print("By Back Substitution, The solution are:")
        #print each solution X1, X2, X3, ..., Xn
        for i in range(num_of_Y):
            print(f"X{i+1} = {arr_of_solutions[i][0]}")
    return arr_of_solutions

def get_elimination_matrix_for_a_column(matrix: list[list[int]], x_index: int, y_index: int, Up_or_Down_the_current_y_index: str, verbose: bool = False) -> list[list[int]] | None: #eliminate the y_index on the particular x_index
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    eli_matrix = [[0 for _ in range(num_of_Y)] for _ in range(num_of_Y)]
    #STEP1: transform eli_matrix into an identity matrix
    for common_index in range(num_of_Y):
        eli_matrix[common_index][common_index] = 1

    #STEP2: find the largest pivot and detect whether the column is all 0 or not
    '''# not numerically stable part but educational part due to not using partial pivoting
    if matrix[y_index][x_index] == 0: #if the current y is 0, then exchange the y with the next y that has non-0 pivot
       y_index_of_non0_pivot = Find_non0_pivot(matrix, x_index, y_index)
       if y_index_of_non0_pivot != None:
           Exchange_Y(matrix, y_index, y_index_of_non0_pivot)
           if verbose == True:
               print(f"There exists 0 pivot in row {y_index+1}-->Exchange row {y_index+1} with row {y_index_of_non0_pivot+1}:")
               Print_matrix(matrix)
       elif y_index_of_non0_pivot == None:
            return None #if there is no non-0 pivot, then the matrix is singular
    '''
    ##numerically stable part by implementing partial pivoting
    y_index_of_largest_pivot = find_largest_pivot_as_well_as_non_zero_pivot(matrix, x_index, y_index)
    if y_index_of_largest_pivot != None:
        exchange_rows(matrix, y_index, y_index_of_largest_pivot)
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

def exchange_rows(matrix: list[list[int]], y1: int, y2: int) -> None: # exchange the row of the matrix
    matrix[y1], matrix[y2] = matrix[y2], matrix[y1]
    return
def find_non0_pivot(matrix: list[list[int]], x_index: int, y_starting_index: int): 
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    y_index = y_starting_index + 1 #plus 1 because we are looking for the next lower Y after finding the 0 pivot of the current Y
    #print(f"y_index: {y_index}")
    while y_index < num_of_Y:
        if matrix[y_index][x_index] != 0:
            return y_index
        y_index+=1
    #otherwise, return None if there is no non-0 pivot
    return None
def find_largest_pivot_as_well_as_non_zero_pivot(matrix: list[list[int]], x_index: int, y_starting_index: int):
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    y_index = y_starting_index 
    largest_pivot = matrix[y_starting_index][x_index]
    y_index_of_largest_pivot = y_starting_index
    while y_index < num_of_Y:
        if abs(matrix[y_index][x_index]) > abs(largest_pivot):
            largest_pivot = matrix[y_index][x_index]
            y_index_of_largest_pivot = y_index
        y_index+=1
    #check if the largest pivot is 0, then return None
    if largest_pivot == 0:
        return None
    return y_index_of_largest_pivot

def read_column_entries(matrix: list[list[int]], x_index: int) -> list[list[int]]: #used when one needs to have a column vector of a matrix
    #information of the matrix
    num_of_Y = len(matrix)
    #main algorithm
    arr_result = [ [0] for _ in range(num_of_Y)]
    y_index=0
    while y_index < num_of_Y:
        arr_result[y_index][0] = matrix[y_index][x_index]
        y_index+=1    
    return arr_result
def read_row_entries(matrix: list[list[int]], y_index: int) -> list[list[int]]: #used when one needs to have a row vector of a matrix
    #information of the matrix
    num_of_X = len(matrix[0])
    #main algorithm
    arr_result = [ [0] for _ in range(num_of_X)]
    x_index=0
    while x_index < num_of_X:
        arr_result[x_index][0] = matrix[y_index][x_index]
        x_index+=1    
    return arr_result

def check_whether_solutions_satisfy_equations(arr_of_solutions: list, Augmented_matrix: list[list[int]]) -> bool:
    #information of matrix
    num_of_Y = len(Augmented_matrix) #or could be = len(arr_of_solutions)
    num_of_X = len(Augmented_matrix[0])
    #main algorithm
    y_index = 0
    while y_index < num_of_Y:
        right_side_result = round(dot_product(arr_of_solutions, Augmented_matrix[y_index][0:num_of_X])[0][0])
        left_side_result = round(Augmented_matrix[y_index][num_of_X-1])
        if right_side_result != left_side_result:
            return False
        y_index += 1
    return True
def print_matrix(matrix: list[list[int]]) -> None:
    print("////////////////////////////////////////")
    for row in matrix:
        #print([round(num, 3) for num in row])
        #print whole integer part of the number
        #print([int(num) for num in row])
        #print without any rounding
        print(row)
    return

def random_matrix_generator(num_of_Y: int, num_of_X: int) -> list[list[int]]:
    matrix = [[random.randint(0, 10) for _ in range(num_of_X)] for _ in range(num_of_Y)]
    return matrix
def round_small_values_to_zero(matrix: list[list[int]], threshold: float = 1e-1) -> list[list[int]]:
    num_of_Y = len(matrix)
    num_of_X = len(matrix[0])
    y_index = 0
    while y_index < num_of_Y:
        x_index = 0
        while x_index < num_of_X:
            if abs(matrix[y_index][x_index]) < threshold:
                matrix[y_index][x_index] = 0
            x_index+=1
        y_index+=1
    return matrix
def col_to_list(col_vector: list[list[int]]) -> list[int]:
    return [col_vector[i][0] for i in range(len(col_vector))]
def list_to_col(arr: list[int]) -> list[list[int]]:
    return [[arr[i]] for i in range(len(arr))]
def get_identity_matrix(dimension: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(dimension)] for i in range(dimension)]
def init_matrix(num_of_Y: int, num_of_X: int) -> list[list[int]]:
    return [[0 for _ in range(num_of_X)] for _ in range(num_of_Y)]
if __name__ == "__main__":
    main()