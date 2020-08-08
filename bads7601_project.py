# Prasit Chulanutrakul 6120412007
import copy
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict

class Graph:

    def __init__(self, graph_type:str, verbose=False):
        '''
        Parameters: graph_type : {'graph', 'weighted_graph'}
                    verbose : bool, defalut False
        Returns:    Graph object
        '''
        self.__graph_type = graph_type
        self.__graph = dict()
        self.__goal = str()
        self.__heuristic = dict()
        self.verbose = verbose

    def add_node(self, node:str, children, heuristic:float=None):
        '''
        Add graph's node and it's neighbor nodes.
        Parameters: node : str, node's name
                    children : list or dict
                                'graph' must be list contains all names of children nodes.
                                'weighted_graph' must be dict which each key contain name of child node and each value is the score between node and child.
                    heuristic : float,  
                                'weighted_graph' must be the huristic value from node to goal, default None.
                                'graph' must be None.
        '''

        if isinstance(children, list):
            all_child_node = dict()
            for each in children:
                all_child_node[each] = None
            self.__graph[node] = all_child_node
        else:
            self.__graph[node] = children

        self.__heuristic[node] = heuristic

    def add_goal(self, goal):
        '''
        Add goal node for 'weighted_graph'.
        Parameters: goal : str, the goal node.
        '''

        if self.__graph_type == 'weighted_graph':
            self.__goal = goal
        else:
            raise Exception('weighted_graph only')

    def remove_node(self, node:str):
        '''
        Remove graph's node.
        Parameters: node : str, node's name
        '''
        if node in self.__graph:
            del self.__graph[node]

        if node in self.__heuristic:
            del self.__heuristic[node]

    def graph_from_file(self, file_dir:str):
        '''
        Create graph from file.
        Parameters: file_dir : str, directory of input file.
        '''
        with open(file_dir, 'r') as f:
            lines = f.readlines()

        if self.__graph_type == 'graph':
            for line in lines:
                line_list = line.strip().split()
                self.add_node(line_list[0], line_list[1:])

        elif self.__graph_type == 'weighted_graph':

            self.add_goal(lines[0].strip())
            # Create graph
            for line in lines[1:]:
                line_list = line.strip().split()
                node = line_list.pop(0)
                heuristic = int(line_list.pop(0))
                child = dict()
                for i, member in enumerate(line_list[::2]):
                    child[member] = float(line_list[2*i+1])
                self.add_node(node, child, heuristic)

    def get_graph(self):
        '''
        Get graph structure
        Return:     str, Graph structure
        '''
        if self.__graph_type == 'graph':
            return {'relations': self.__graph}
        elif self.__graph_type == 'weighted_graph':
            return {'goal': self.__goal, 'relations': self.__graph, 'heuristic': self.__heuristic}

    def get_graph_type(self):
        '''
        Get graph type
        Return:     str, Graph type
        '''
        return self.__graph_type

    def uninform_search(self, algorithm:str, start:str, goal:str, expanded_criteria='alphabetical'):
        '''
        Uninform search
        Parameters: algorithm : {'BFS', 'DFS'}
                    start : str, name of start node
                    goal  : str, name of goal node 
                    expanded_criteria : {'alphabetical', 'reverse_alphabetical', 'parent'}
        Returns:    str, Search result                            
        '''

        # Check arguments
        if algorithm not in ['BFS', 'DFS']:
            raise ValueError('algorithm must be "BFS" or "DFS" only.')

        if expanded_criteria not in ['alphabetical', 'reverse_alphabetical', 'parent']:
            raise ValueError('expanded_criteria must be "alphabetical", "reverse_alphabetical" or "parent" only.')

        # Initial queue
        # Member of queue consists of itself, it's parent, depth of search
        queue = [[start, None, 1]]

        # Initial visited
        visited = OrderedDict()
        expanded = list()

        while queue:

            # Print if verbose is true.
            if self.verbose: 
                print('Queue: ', queue)
                print('Visited: ', visited)

            # Stop searching if the goal was expanded.
            if goal in visited:
                break

            # Getting the node to be expanded from queue
            node_to_expand = queue.pop(0)

            # Check the node whether it was expanded or was not expanded?
            if node_to_expand[0] not in visited:
                # Expanding node

                expanded_nodes = list(self.__graph[node_to_expand[0]].keys())
                # Add the node which was just expanded and it parent into visited.
                visited[node_to_expand[0]] = node_to_expand[1]
                expanded.append(node_to_expand[0])

                # Remove parent node
                if node_to_expand[1] in expanded_nodes:
                    expanded_nodes.remove(node_to_expand[1])

                if algorithm == 'BFS':
                    # --------------------------------------------------------------------------------
                    # QUEUE
                    # --------------------------------------------------------------------------------
                    for expanded_node in expanded_nodes:
                        queue.append([expanded_node, node_to_expand[0], node_to_expand[2] + 1])
                    
                    # Expanded criteria for BFS
                    if expanded_criteria == 'alphabetical':
                        queue = sorted(queue, key = lambda x: (x[2], x[0]), reverse=False)
                    elif expanded_criteria == 'reverse_alphabetical':
                        queue = sorted(queue, key = lambda x: (x[2], x[0]), reverse=True)
                    else: # 'parent'
                        pass
                    # --------------------------------------------------------------------------------
                elif algorithm == 'DFS':
                    # --------------------------------------------------------------------------------
                    # STACK
                    # --------------------------------------------------------------------------------
                    # Expanded criteria for DFS
                    if expanded_criteria == 'alphabetical':
                        sorted_expanded_nodes = sorted(expanded_nodes, reverse=True)
                    elif expanded_criteria == 'reverse_alphabetical':
                        sorted_expanded_nodes = sorted(expanded_nodes, reverse=False)
                    else: # 'parent'
                        sorted_expanded_nodes = expanded_nodes.copy()

                    for expanded_node in sorted_expanded_nodes:
                        queue.insert(0, [expanded_node, node_to_expand[0], node_to_expand[2] + 1])
                    # --------------------------------------------------------------------------------
            else:
                expanded.append(f'({node_to_expand[0]})')

        return {'Expanded':expanded, 'Path':self.__path(visited, goal)}

    def inform_search(self, algorithm:str, start:str):
        '''
        Inform search
        Parameters: algorithm : {'a*', 'greedy'}
                    start : str, name of start node
        Returns:    str, Search result
        '''
            
        if self.__graph_type != 'weighted_graph':
            raise TypeError('"graph_type" must be weighted_graph.')

        # Check arguments
        if algorithm not in ['a*', 'greedy']:
            raise ValueError('algorithm must be "a*" or "greedy" only.')

        # Initial queue
        # Member of queue consists of 
            # 0 itself, 
            # 1 it's parent, 
            # 2 depth of search, 
            # 3 sum_distance, 
            # 4 heuristic, 
            # 5 sum_distance + heuristic
            
        queue = [[start, None, 1, 0, self.__heuristic[start], self.__heuristic[start]]]

        # Initial visited
        visited = OrderedDict()
        expanded = list()

        while queue:

            # Print if verbose is true.
            if self.verbose: 
                print('\n')
                print('Queue: ', queue)
                print('\n')
                print('Visited: ', visited)

            # Stop searching if the goal was expanded.
            if self.__goal in visited:
                break

            # Getting the node to be expanded from queue
            if algorithm == 'a*':
                index_to_pop = min(range(len(queue)), key = lambda i: (queue[i][5], queue[i][0], queue[i][1]))
            elif algorithm == 'greedy':
                index_to_pop = min(range(len(queue)), key = lambda i: (queue[i][4], queue[i][0], queue[i][1]))

            node_to_expand = queue.pop(index_to_pop)

            # Check the node whether it was expanded or was not expanded?
            if node_to_expand[0] not in visited:
                # Expanding node

                expanded_nodes = self.__graph[node_to_expand[0]].copy()
                # Add the node which was just expanded and it parent into visited.
                visited[node_to_expand[0]] = node_to_expand[1]

                if algorithm == 'a*':
                    expanded.append(f'{node_to_expand[0]}.{node_to_expand[5]}')
                elif algorithm == 'greedy':
                    expanded.append(f'{node_to_expand[0]}')

                # Remove parent node
                if node_to_expand[1] in expanded_nodes:
                    del expanded_nodes[node_to_expand[1]]

                # QUEUE
                for expanded_node in expanded_nodes:

                    sum_distance = node_to_expand[3] + expanded_nodes[expanded_node]
                    queue.append([expanded_node, # itself
                                    node_to_expand[0], # it's parent
                                    node_to_expand[2] + 1, # depth of search
                                    sum_distance, # sum_distance
                                    self.__heuristic[expanded_node],
                                    sum_distance + self.__heuristic[expanded_node]
                                ])

            else:
                if algorithm == 'a*':
                    expanded.append(f'({node_to_expand[0]}.{node_to_expand[5]})')
                elif algorithm == 'greedy':
                    expanded.append(f'({node_to_expand[0]})')

        return {'Expanded':expanded, 'Path':self.__path(visited, self.__goal), 'Distance': node_to_expand[3]}

    def print_search_result(self, result):
        print('Expanded: ', ', '.join(result['Expanded']))
        print('Path:     ', ', '.join(result['Path']))
        if 'Distance' in result:
            print('Distance: ', result['Distance'])

    def plot_g(self, g, path:list=None, size:tuple=(20,12), sg_pos:float=-0.01, heuristic_pos:float=0.01):
        '''
        Plot the graph
        Parameters:     g : Graph, the graph object to plot.
                        path : list, list of path node, defalut None
                        size : tupple, size of figure in format of (x, y)
                        sg_pos : float, position of start and goal label
                        heuristic_pos : float, position of heuristic value label

        
        '''
        G = nx.Graph()
        relations = g.get_graph()['relations']
        nodes = list(relations.keys())
        
        G.add_nodes_from(nodes)
        
        edges = list()
        if g.get_graph_type() == 'graph':
            for key in relations:
                for child in relations[key]:
                    edges.append((key, child))
            G.add_edges_from(edges)
            
        elif g.get_graph_type() == 'weighted_graph':
            for key in relations:
                for child in relations[key]:
                    edges.append((key, child, relations[key][child]))

            G.add_weighted_edges_from(edges)
            
        pos = nx.kamada_kawai_layout(G, weight=None)
        #pos = nx.kamada_kawai_layout(G)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        #print(len(edge_labels))
        plt.figure(figsize=size) 
        
        nx.draw_networkx(G, pos=pos, with_labels=True, font_weight='bold', 
                        node_size=500, node_color='#bdc3c7', edge_color='#bdc3c7', width=2)
            
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=8)
        
        if path is not None:
            
            edgelist = list()
            for i, node in list(enumerate(path))[:-1]:
                edgelist.append((path[i], path[i+1]))
            
            nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, edge_color='#ffb700', width=4)
            nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color='#ffc800', node_size=500)
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[path[0]], node_color='#139e2f', node_size=700)
            nx.draw_networkx_nodes(G, pos=pos, nodelist=[path[-1]], node_color='#ff0000', node_size=700)
            

            pos_sg = copy.deepcopy(pos)
            for key in pos_sg:
                pos_sg[key][1] += sg_pos
                
            nx.draw_networkx_labels(G, pos=pos_sg, labels={path[0]:'START'}, font_weight='bold', font_color='#139e2f')
            nx.draw_networkx_labels(G, pos=pos_sg, labels={path[-1]:'GOAL'}, font_weight='bold', font_color='#ff0000')
        
        pos_h = copy.deepcopy(pos)
        for key in pos_h:
            pos_h[key][1] += heuristic_pos
        if g.get_graph_type() == 'weighted_graph':
            nx.draw_networkx_labels(G, pos=pos_h, labels=g.get_graph()['heuristic'], font_size=10)
        
        plt.show()

    def __path(self, visited, goal):
        path = [goal]
        key = visited[goal]
        while visited[key] is not None:
            path.append(key)
            key = visited[key]
            start = key
        path.append(start)
        return list(reversed(path))

    def __call__(self):
        return self.get_graph()

def run_uninform():

    while True:
        
        input_dir = input('>> Please input the input file part or type "end" to exit the program: ')
        if input_dir == 'end':
            break
        graph_type = input('>> Please input type of graph in the input file either "graph" or "weighted_graph": ')
        g = Graph(graph_type, verbose=False)

        start_goal = input('>> Please input the start and goal node in format of "start goal": ')
        start, goal = start_goal.split()
        print(start, goal)
        g.graph_from_file(input_dir) 
        result = g.uninform_search('BFS', start, goal, expanded_criteria='alphabetical')
        print('\nBFS')
        g.print_search_result(result)
        result = g.uninform_search('DFS', start, goal, expanded_criteria='alphabetical')
        print('\nDFS')
        g.print_search_result(result)
        print('\n')

def run_inform():

    while True:
        input_dir = input('>> Please input the input file part or type "end" to exit the program: ')
        if input_dir == 'end':
            break
        graph_type = input('>> Please input type of graph in the input file either "graph" or "weighted_graph": ')
        g = Graph(graph_type, verbose=False)
        
        start = input('>> Please input the start node: ')
        g.graph_from_file(input_dir) 
        result = g.inform_search('greedy', start)
        print('\nGreedy')
        g.print_search_result(result)
        result = g.inform_search('a*', start)
        print('\nA*')
        g.print_search_result(result)
        print('\n')

if __name__ == '__main__':
    
    search_type = input('>> Please type the search type either "uninform" or "inform": ')
    if search_type == 'inform':
        run_inform()
    elif search_type == 'uninform':
        run_uninform()
    else:
        print('Search type error.')