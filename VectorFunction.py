import numpy as np
import sympy as sp
import linearalgebra as la
import copy
import pygame
from MotionOfObject import Coordinate3DSystem
# Main function that ties everything together
def main():
    #test the VectorValuedFunction class
    C0 = 2*sp.pi/2360448
    C1 = 3.84*10**8 
    t = sp.symbols("t")
    #check whether (axb)xc = ax(bxc)
    a = [t,sp.sin(t),sp.cos(t)]
    b = [t,t**2,t]
    c = [t**2,t,sp.sin(t)]
    for i in range(1,10):
       for j in range(3):
         print(f"{j+1}th component of (a x b) x c at t{i}: {VectorValuedFunction._cross(VectorValuedFunction._cross(a, b), c)[j].subs(t, i).evalf()}")
         print(f"{j+1}th component of a x (b x c) at t{i}: {VectorValuedFunction._cross(a, VectorValuedFunction._cross(b, c))[j].subs(t, i).evalf()}")
       print("-------------------------------------------------")
    #d =VectorValuedFunction._cross(VectorValuedFunction._cross(a, b), c)
    #print(f" d at t{1}: {d[0].subs(t, 18).evalf()}")

class VectorValuedFunction:
    def __init__(self, symbolic_expr: list[sp.Expr]):
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

        self.xyz_infor = []  # np.array([ [x0,y0,z0], [x1,y1,z1],...,[xn,yn,zn] ]) ]) that store information about the position of the object as time goes on
  
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
    def draw(self, screen: pygame.Surface, coordinate_system: Coordinate3DSystem):
        #Step1: project each point in self.xyz_infor to self-relative coordinate system
        points = copy.deepcopy(self.xyz_infor)
        for i in range(len(points)):
            points[i][0] = la.get_projection_vector(coordinate_system.axises[0], points[i])[0]
            points[i][1] = la.get_projection_vector(coordinate_system.axises[1], points[i])[1]
            points[i][2] = la.get_projection_vector(coordinate_system.axises[2], points[i])[2]
        #Step2: draw the point on the screen

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
        # A x B = (a2b3 - a3b2)i + (a3b1 - a1b3)j + (a1b2 - a2b1)k
        return [vvf1[1]*vvf2[2] - vvf1[2]*vvf2[1], vvf1[2]*vvf2[0] - vvf1[0]*vvf2[2], vvf1[0]*vvf2[1] - vvf1[1]*vvf2[0]]


# Example usage
if __name__ == "__main__":
    main()