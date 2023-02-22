import os
from cv2 import norm
import torch
import sys
import argparse
import numpy as np

from utils.gen_utils import *
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import pdist, squareform

def generate_and_solve(num_nodes, num_vehicles):
    """Generates a random VRP and solves it.
    
    References:
        Setting up a TSP instance with OR-tools : https://developers.google.com/optimization/routing/tsp

    Returns:
        distance_matrix of shape (n_nodes, n_nodes) : The randomly generated distance matrix 
                                                      for the VRP.
        routes of shape (n_vehicles, n_nodes) : Route matrix giving the ordering of nodes
                                                on the routes of the vehicles. 
        adj_mat of shape (n_nodes, n_nodes) : Adjacency matrix of the solution graph.
        route_length : Optimal value of the objective function.
    """
    # Instantiate the data problem.
    coords, data = create_data_model(num_nodes, num_vehicles)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
   
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Setting local search metaheuristic
    search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)  
    search_parameters.time_limit.seconds = 1
    #search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Return coordinates, distance matrix, node solutions, adjacency matrix and route length for storage.
    if solution:
        #print_solution(data, manager, routing, solution)
        nodes = get_solution_nodes(solution, routing, manager)
        adj_mat = get_adjacency_from_nodes(nodes)
        route_length = get_total_route_length(data['distance_matrix'], adj_mat)
        
        # return coords, data['distance_matrix'], nodes, adj_mat, route_length
        return coords, route_length, data['distance_matrix']
    else:
        print('No solution found!')

def normalize_dataset(coords, route_length, dist_matrix):

        x_max = np.max(list(zip(*coords))[0])
        y_max = np.max(list(zip(*coords))[1])

        normalizer = np.mean([x_max, y_max])

        x_norm = list(zip(*coords))[0] / normalizer
        y_norm = list(zip(*coords))[1] / normalizer

        coords = list(zip(x_norm, y_norm))
        route_length = route_length / normalizer
        dist_matrix = dist_matrix / normalizer
        
        return coords, route_length, dist_matrix


def main(opts):
    # filename = input("Give name of desired file.\n")
    type = str(input('Which data do you want to generate? (train/test/validation)\n'))

    base_path = os.path.join(os.getcwd(),"vrp_data_large",type,"raw")

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    start = 0

    opts.num_nodes = np.random.randint(20,100)
    opts.num_vehicles = np.random.randint(1,10) # VRP
    # opts.num_vehicles = 1 # TSP

    coords, route_length, dist_matrix = generate_and_solve(opts.num_nodes, opts.num_vehicles)

    DTYPE = torch.float64

    if opts.normalize == 'y': 
        coords, route_length, dist_matrix = normalize_dataset(coords, route_length, dist_matrix)

    # add depot token and number of vehicles
    tokens = np.zeros(len(coords)).reshape(len(coords),1)
    tokens[0] = 1 # depot

    vehicles = torch.zeros(len(coords)).reshape(len(coords),1)
    vehicles[:] = opts.num_vehicles

    # TODO: rename to node features
    coords = np.array(np.concatenate([coords,tokens,vehicles], axis=1)) # VRP
    # coords = np.array(np.concatenate([coords,tokens], axis=1)) # TSP
    
    nodes_tensor = torch.tensor([coords], dtype=DTYPE)
    route_length_tensor = torch.tensor([route_length], dtype=DTYPE)
    dist_matrix = torch.tensor([dist_matrix], dtype=DTYPE)

    dataset = torch.utils.data.TensorDataset(nodes_tensor,
                                            route_length_tensor,
                                            dist_matrix)
    
    path = os.path.join(base_path, f'{start}.pt')
    
    try:
        torch.save(dataset, path)
    except FileExistsError:
        print('File already exists.')
        pass

    start += 1

    for i in range(start, opts.num_instances):

        opts.num_nodes = np.random.randint(20,100)
        opts.num_vehicles = np.random.randint(1,10) # VRP
        # opts.num_vehicles = 1 # TSP

        coords, route_length, dist_matrix = generate_and_solve(opts.num_nodes, opts.num_vehicles)
        if opts.normalize == 'y': 
            coords, route_length, dist_matrix = normalize_dataset(coords,route_length,dist_matrix)
        
        # add depot token and number of vehicles
        tokens = np.zeros(len(coords)).reshape(len(coords),1)
        tokens[0] = 1 # depot

        vehicles = torch.zeros(len(coords)).reshape(len(coords),1)
        vehicles[:] = opts.num_vehicles
    
        coords = np.array(np.concatenate([coords,tokens,vehicles], axis=1)) # VRP
        # coords = np.array(np.concatenate([coords,tokens], axis=1)) # TSP

        # shuffle coords data to change depot location
        idx = torch.randperm(coords.shape[0])
        coords = coords[idx]

        nodes_tensor = torch.tensor([coords], dtype=DTYPE)
        route_length_tensor = torch.tensor([route_length], dtype=DTYPE)
        dist_matrix = torch.tensor([dist_matrix], dtype=DTYPE)

        dataset = torch.utils.data.TensorDataset(nodes_tensor,
                                                route_length_tensor,
                                                dist_matrix)

        path = os.path.join(base_path, f'{i}.pt')

        try:
            torch.save(dataset, path)
        except FileExistsError:
            print('File already exists.')
            pass

        if i % 10 == 0:
            print(f"{i} data points generated.\n")

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_instances", type=int, default=1000,
                        help="Number of data instances to generate")
    parser.add_argument("--normalize", default='y',
                        help="Whether to normalize data points (y/n)")

    opts = parser.parse_args()
    main(opts)

