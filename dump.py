# import sys
# sys.path.append('.')


# from main import TrajectoryDesign
# import pytest

# # @pytest.fixture(scope='module')
# def build_graph():
#     def distance(point1, point2):
#         return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
#     map_boundary = [[0, 0], [10, 0], [10, 10], [0, 10]]
#     obstacles = [[[2, 2], [2, 3], [3, 3], [3, 2]], [[5, 5], [5, 6], [6, 6], [6, 5]]]
#     end_point = [8, 8]
#     start_point = [0, 0]
#     tau = 1
#     td = TrajectoryDesign(map_boundary, obstacles, end_point, start_point, tau)
#     graph, points = td.build_graph()
#     # print(graph[(2,3)][(3,3)])
#     # print(points)
#     for point1 in points:
#         for point2 in points:
#             if point1 != point2:
#                 assert graph[tuple(point1)][tuple(point2)] == graph[tuple(point2)][tuple(point1)] == distance(point1, point2)




#gurobi# 
# def lines_intersect(p1, p2, q1, q2):
#     """Check if line segments p1-p2 and q1-q2 intersect using Gurobi."""
#     model = gp.Model()
#     model.setParam('OutputFlag', 0)

#     # Define variables for intersection points
#     t = model.addVar(lb=0, ub=1, name="t")
#     u = model.addVar(lb=0, ub=1, name="u")

#     # Line equations
#     model.addConstr((q1[0] - p1[0]) * t == (p2[0] - p1[0]) * u)
#     model.addConstr((q1[1] - p1[1]) * t == (p2[1] - p1[1]) * u)

#     # Solve the model
#     model.optimize()
#     if model.status == GRB.OPTIMAL:
#         return True

#     return False