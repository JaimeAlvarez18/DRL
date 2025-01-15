import gymnasium as gym
from gymnasium.spaces import Dict,Box,Discrete
import numpy as np
from grid_field import Grid_field
from scipy.optimize import minimize
import math
from shapely.geometry import Polygon
from matplotlib.path import Path
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy


import time




class Environment(gym.Env):
    def __init__(
            self,
            config
        ) -> None:
        """
        Constructor of the Environment. Inherits from gymnasium.
        Parameters:
        -----------
            config (dict):
                Dictionary with all configuration set in config.yaml.
        """
        super().__init__()
        self.number_of_actions = config['Policy']['Number_of_actions']

        self.same_environment=config['Layout']['Same']
        self.vertices=config['Layout']['figure']
        self.n_vertices=config['Layout']['Number_vertices']
        self.hop = config['Layout']['hop']
        self.cell_size = config['Layout']['cell_size']
        self.rotation_angle = config['Layout']['rotation_angle']
        self.t_width= config['Layout']['width']
        self.t_height= config['Layout']['height']
        self.area= config['Layout']['area']

        self.reward_n_step = config['Penalties']['n_step']
        self.reward_repeated_cell = config['Penalties']['repeated_cell']
        self.reward_out_bounds = config['Penalties']['out_bounds']
        self.sparse_reward = config['Penalties']['sparse_reward']
        self.reward_passed_cell = config['Penalties']['pass_cell']

        self.max_steps = config['Training']['Max_steps']
        self.render_step = config['Training']['N_step_per_render']
        
        # declare variables
        self.n_step=0
        self.total_reward=0
        self.best_score=-math.inf
        self.all_steps=0
        self.n_episode=0
        self.best_episode=0
        self.accumulated=0
        self.total_completed=0
        self.n_times_out=0
        self.time=time.time()


        print('Creating vertices...')
        

        #if using custom vertices
        if self.same_environment:
            # Layout is the same for all episodes, no padding is needed
            add_padding=False 

            # Create Scaler
            scale=MinMaxScaler().fit(self.vertices)

            #Scale vertices
            self.vertices=scale.transform(self.vertices)

            #Scale step
            hop1=np.array([self.hop,self.hop]).reshape(1,-1)
            hop1=scale.transform(hop1)
            self.hop=np.mean(hop1)

            # Scale cell size
            cell1=np.array([self.cell_size,self.cell_size]).reshape(1,-1)
            cell1=scale.transform(cell1)
            self.cell_size=np.mean(cell1)

            # Scale width
            width=np.array([self.t_width,self.t_width]).reshape(1,-1)
            width=scale.transform(width)
            self.t_width=np.mean(width)

            # Scale height
            height=np.array([self.t_height,self.t_height]).reshape(1,-1)
            height=scale.transform(height)
            self.t_height=np.mean(height)

            # Create grid
            self.grid= Grid_field(self.vertices,self.cell_size,add_padding)
            
            #store grid for future episodes
            self.constant_grid=deepcopy(self.grid)
        else:
            #create edge
            self.create_edge(self.cell_size,self.n_vertices,self.area,self.t_width)

        #create truck
        self.rectangle = self.get_random_truck(self.t_width,self.t_height)
        # self.rectangle= self.grid.get_truck(self.t_width,self.t_height)

        #check truck within edge
        out=self.check_out_edge()

        #recreate edge and truck if invalid
        while out:
            #if not using custom vertices
            if self.same_environment == False:
                #create edge
                self.create_edge(self.cell_size,self.n_vertices,self.area,self.t_width)
                
            #create truck
            self.rectangle = self.get_random_truck(self.t_width,self.t_height)
            # self.rectangle= self.grid.get_truck(self.t_width,self.t_height)

            out=self.check_out_edge()

        # Sparse reward
        if self.sparse_reward == 'all_cells':
            self.sparse_reward=self.grid.total_cells
        
        # Min and max values (everything was scaled to interval [0,1])
        min_value=0
        max_value=1

        # Define observation space
        self.observation_space = Dict(
            {"grid":Box(low=min_value, high=max_value, shape=self.grid.middle_points.shape,dtype=float),
             "current_position" : Box(low=min_value, high=max_value, shape=self.rectangle.shape,dtype=float),
             'edges' : Box(low=min_value, high=max_value,shape=self.grid.vertices.shape, dtype=float)
             })
        
        # Define action space
        self.action_space=Discrete(self.number_of_actions)
        

    def create_edge(
            self,
            cell_size : float,
            n_vertices : int,
            area : float,
            width : float
        ) -> None:
        """
        Creates the grid.
        Parameters:
        -----------
            cell_size (float):
                Cell size.
            n_vertices (int):
                Number of vertices of the layout.
            area (float):
                Area the layout must have.
            width (float):
                Width of the tractor.
        Returns:
        --------
            None
        """

        #constants
        n_angl=120
        n_width=width*5
        self.vertices=[]

        # Make sure everything went right
        while len(self.vertices) != n_vertices:
            self.vertices=self.polygon_vertices(n_vertices,area,n_angl,n_width)

        #create grid 
        self.grid= Grid_field(self.vertices,cell_size)
        self.vertices=self.grid.vertices


    def rotate_around_vertex(
            self,
            vertices : list[list], 
            theta : float, 
            vertex_index : int =0
        ) -> list[list]:
        """
        Rotates the given vertices a given angle.
        Parameters:
        -----------
            vertices (list[list]):
                Vertices to rotate (x,y coordinates).
            theta (float):
                Angle to rotate (degrees). If negative rotates clockwise, if positive counterclockwise
            vertex_index (int):
                Vertix to rotate around.
                    if 0 ,rotate around bottom-left vertix
                    if 1, rotate around bottom-right vertix
        Returns:
        --------
            Rotated vertices (list[list]) x,y coordinates.
        """
        angle = np.radians(theta)

        # Create a rotation matrix.
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        # Translate the rectangle so that the vertex to rotate around is at the origin.
        translated_rectangle = vertices - vertices[vertex_index]

        # Rotate the translated rectangle.
        rotated_rectangle = np.dot(translated_rectangle, rotation_matrix)

        # Translate the rotated rectangle back to its original position.
        rotated_rectangle = rotated_rectangle + vertices[vertex_index]

        return rotated_rectangle

 
        
    def move_rectangle(
            self,
            vertices : list[list], 
            direction : list, 
            step_size : float
        ) -> list[list]:
        """
        Moves the vertices in the given direction.
        Parameters:
        -----------
            vertices (list[list]):
                Vertices to move (x,y coordinates).
            direction (list):
                Vector in which the vertices will move.
            step_size (float):
                Step of the movement.
                    If negative move backwards.
                    If positive move forward.
        Returns:
        --------
            Moved vertices (list[list]).
        """
        move_vector = direction * step_size
        
        # Apply the movement vector to all vertices
        moved_vertices = vertices + move_vector
        return moved_vertices

    def get_forward_direction(
            self,
            vertices : list[list]
        ) -> list:
        """
        Calculates the direction (positive or negative) in which the vertices move forward/backwards.
        Parameters:
        -----------
            vertices (list[list]):
                Vertices to move (x,y coordinates).
        Returns:
        --------
            Direction to move (list)
        """
        # Calculate the direction vector from the first vertex to the second vertex
        direction_vector = vertices[3] - vertices[0]
        
        # Normalize the direction vector to get the forward direction
        forward_direction = direction_vector / np.linalg.norm(direction_vector)
        
        return forward_direction
        
    def render(
            self
        ) -> None:
        """
        Plot the figure if has passed self.render steps since last plot.
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        """

        print(f'Render. Total number of steps done: {self.all_steps}. Total reward: {self.total_reward}')
        self.all_steps += self.n_step
        
        
        self.grid.plot_polygon_and_cells(self.rectangle,True,self.total_reward,self.n_step,self.n_episode,self.best_episode,self.best_score)

    def polygon_vertices(
            self,
            num_vertices : int, 
            area : float, 
            min_angle : float, 
            min_edge_length : float
        ) -> list[list]:

        #Generate random vertices
        def random_vertices(num_vertices, radius_range=(0.5, 2)):
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))  
            radii = np.random.uniform(radius_range[0], radius_range[1], num_vertices)  
            vertices = [(r * math.cos(angle), r * math.sin(angle)) for r, angle in zip(radii, angles)]
            return vertices

        # Calculate area of the polygon
        def polygon_area(vertices):
            x = [v[0] for v in vertices]
            y = [v[1] for v in vertices]
            return 0.5 * abs(sum(x[i] * y[(i + 1) % len(vertices)] - y[i] * x[(i + 1) % len(vertices)]
                                for i in range(len(vertices))))

        # Calculate the angles between adjacent edges
        def polygon_angles(vertices):
            angles = []
            for i in range(len(vertices)):
                p1 = np.array(vertices[i - 1])
                p2 = np.array(vertices[i])
                p3 = np.array(vertices[(i + 1) % len(vertices)])
                v1 = p1 - p2
                v2 = p3 - p2
                cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                angles.append(np.degrees(angle))
            return angles

        # Calculate the lengths of the edges
        def edge_lengths(vertices):
            lengths = []
            for i in range(len(vertices)):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % len(vertices)])
                lengths.append(np.linalg.norm(p1 - p2))
            return lengths

        # Calculates the loss of area, min_edge length and min_angle
        def cost_function(vertices_flat):
            vertices = [(vertices_flat[i], vertices_flat[i + 1]) for i in range(0, len(vertices_flat), 2)]
            current_area = polygon_area(vertices)
            angles = polygon_angles(vertices)

            lengths = edge_lengths(vertices)

            # Penalty for deviating from the target area
            area_penalty = (current_area - area) ** 2

            # Penalty for violating minimum angle constraint
            angle_penalty = sum(max(0, min_angle - angle) ** 40 for angle in angles)

            # Penalty for edges below the minimum length
            length_penalty = sum(max(0, min_edge_length - length) ** 40 for length in lengths)

            return area_penalty + angle_penalty + length_penalty

        # Generate random initial vertices
        initial_vertices = random_vertices(num_vertices)
        initial_guess = np.array([coord for vertex in initial_vertices for coord in vertex])  # Flatten vertices

        # Minimize the cost function to adjust the vertices
        result = minimize(cost_function, initial_guess, method='L-BFGS-B', options={'maxiter': 10000})

        if result.success:
        # return optimized vertices
            optimized_vertices_flat = result.x
            optimized_vertices = [[optimized_vertices_flat[i], optimized_vertices_flat[i + 1]] 
                                for i in range(0, len(optimized_vertices_flat), 2)]
            
            for i in range(len(optimized_vertices)-1):
                vertix1 = optimized_vertices[i]
                vertix2=optimized_vertices[i+1]
                if vertix1[1]==vertix2[1] or vertix1[0] == vertix2[0]:
                    return []
            angles = polygon_angles(optimized_vertices)
            lengths = edge_lengths(optimized_vertices)
            for length in lengths:
                if length < 0.25:
                    return []
            for angle in angles:
                if angle < 100:
                    return []
            polygon = Polygon(optimized_vertices)

            # If polygon is not valid, return empty list and retry
            if not polygon.is_valid:
                print('Generated polygon is invalid.')
                return []
            else:
                return np.array(optimized_vertices)
            
        # If function not optimized, return empty list and retry
        else:
            return []
        
    def is_point_in_polygon(
            self,
            point : list, 
            polygon :list[list]
        ) -> bool:
        """
        Check if a point is inside a polygon using the ray-casting method.
        Parameters:
        -----------
            point (list):
                Point to check if is within the polygon (x,y coordinates).
            polygon (list[tuples]):
                Polygon should be an array of vertices [(x1, y1), (x2, y2), ..., (xn, yn)].
        Returns:
        --------
            Whether the point is within the polygon.
        """
        path = Path(polygon)
        return path.contains_point(point)
    
    def get_random_truck(
        self,
        width : float,
        height : float
    ) -> list[list]:
        """
        Creates the tractor in a random position, always looking upwards.
        Parameters:
        -----------
            width (float):
                Width of the tractor.
            height (float):
                Height of the tractor.
        Returns:
        --------
            Corners of the tractor, starting from rear-left and going counter-clockwise.
        """
        starting = np.random.randn(2)
        corner1 = starting 
        corner2 = [starting[0]+width,starting[1]]
        corner3 = [starting[0]+width,starting[1]+height]
        corner4 = [starting[0],starting[1]+height]

        corners=np.array([corner1, corner2, corner3, corner4])

        return corners


    def is_polygon_in_rectangle(
            self,
            polygon : list[list],
        ) -> bool:
        """
        Checks if all vertices of a polygon are completely inside the truck, and if it is the first time that polygon is inside the truck.
        Parameters:
        -----------
            polygon (list[list]):
                Polygon to check if is within the truck.
        Returns:
        --------
            Whether the polygon is within the tractor.
        """
        # Check if all vertices of the polygon are inside the rectangle
        type_polygon = int(polygon[6][0])

        # If is padding
        if type_polygon == -1:
            return False
        
        points=polygon[0:type_polygon,:]
        type_polygon = int(polygon[6][1])

        #edge polygon not passed
        if type_polygon == -1:
            for i in points:
                if self.is_point_in_polygon(i, self.rectangle):
                    return True
            return False
        #non-edge polygon not passed
        if type_polygon == 0:
            for i in points:
                if not self.is_point_in_polygon(i, self.rectangle):
                    return False
            return True
        #edge/non-edge polygon already passed
        if type_polygon == 11 or type_polygon==1:
            return False

    def update_regions_central_point(
            self,
            truck : list[list]
        ) -> float:
        """
        Updates the grid cells in which the truck has already passed. Returns the reward for the agent in the current step.
        Parameters:
        -----------
            truck (list[list]):
                Truck that has moved and passed through new cells.
        Returns:
        --------
            Reward for the results of the action taken in the current step.
        """
        #get non-edge polygons passed
        contained_polygons = [index for index,p in enumerate(self.grid.middle_points) if self.is_point_in_polygon(p[0:2],truck)]
        #get edge-polygons that at least one vertix is within the truck
        passed = [index for index,p in enumerate(self.grid.grid) if self.is_polygon_in_rectangle(p)]
        
        reward=0
        p=0

        for i in contained_polygons:
            if self.grid.middle_points[i][2] == 0:
                self.grid.middle_points[i][2] = 1
                self.grid.grid[i][6][1] = 1
                reward+=1
                self.accumulated+=1
            else:
                p+=1
    
        for i in passed:
            if self.grid.middle_points[i][2] == 0:
                self.grid.middle_points[i][2] = 1
                self.grid.grid[i][6][1] = 1
                # self.grid.grid[i][6][1]=11 If using edge polygons differently
                reward+=1
                self.accumulated+=1
            else:
                # cells that the truck has already passed
                p+=1
            

        return reward*self.reward_passed_cell-p*self.reward_repeated_cell
    
    def check_out_edge(
            self,
            rectangle=None
        ) -> bool:
        """
        Checks if the truck is out of the figure.
        Parameters:
        -----------
            None
        Returns:
        --------
            Whether the truck is out of the figure.
        """
        # Check if all vertices of the truck are within the figure

        if rectangle == None:
            for vertix in self.rectangle:
                if not self.is_point_in_polygon(vertix,self.grid.vertices):
                    return True
            return False
        else:
            for vertix in rectangle:
                if not self.is_point_in_polygon(vertix,self.grid.vertices):
                    return True
            return False

    def check_finished_episode(
            self
        ) -> bool:
        """
        Checks if the episode has finished. An episode finishes if the truck has passed through all grid cells.
        Parameters:
        -----------
            None.
        Returns:
        --------
            None.
        """
        for i in self.grid.grid:
            if i[6][0] != -1:
                if i[6][1] == 0 or i[6][1] == -1:
                    return False
        return True
    
    def step(
            self,
            action : int
        ):
        """
        Computes the action taken by the PPO algorithm. Changes the tractor position if the action taken was legal, computes the reward for that action and calculates the
        new state of the environment.
        Parameters:
        -----------
            action (int):
                Action taken by the PPO algorithm.
        Returns:
        --------
            next_state (dict):
                Next state of the environment.
            reward (float):
                Reward gathered by the agent having taken the action in the previous state of the environment.
            done (bool):
                Whether the episode is finished.
            done1 (bool):
                Whether the episode is truncated.
            Dictionary with further information.

        """
        # declare constants
        reward=0
        done=False
        done1=False

        # save current position of the truck
        self.previous_position=deepcopy(self.rectangle)
        if action == 0: #rotate counterclockwise around bottom-left vertix (small rotation)

            rotate=self.rotation_angle
            vertix=0
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds
            

        elif action == 1: #rotate clockwise around bottom-left vertix (small rotation)

            rotate=-self.rotation_angle
            vertix=0
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds


        elif action == 2:#rotate counterclockwise around bottom-right vertix (small rotation)

            rotate=self.rotation_angle
            vertix=1
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds


        elif action == 3:#rotate clockwise around bottom-right vertix (small rotation)

            rotate=-self.rotation_angle
            vertix=1
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds
            
        
        elif action == 4: #move forward (big step)
            move = self.hop
            direction=self.get_forward_direction(self.rectangle)
            self.rectangle=self.move_rectangle(self.rectangle,direction,move)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds
            else:
                reward0 = self.update_regions_central_point(self.previous_position) #before moving
                reward1 = self.update_regions_central_point(self.rectangle) #after moving
                reward=reward0+reward1
                done=self.check_finished_episode()
                if done:
                    reward+=self.sparse_reward
                    print('Finished episode successfully')

        elif action == 5:#move backwards (big step)
            move =-self.hop
            direction=self.get_forward_direction(self.rectangle)
            self.rectangle=self.move_rectangle(self.rectangle,direction,move)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds

        if action == 6: #rotate counterclockwise around bottom-left vertix (big rotation)
            rotate=self.rotation_angle
            vertix=2
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds

        elif action == 7: #rotate clockwise around bottom-left vertix (big rotation)
            rotate=-self.rotation_angle
            vertix=2
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds

        elif action == 8:#rotate counterclockwise around bottom-right vertix (big rotation)
            rotate=self.rotation_angle
            vertix=3
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds
 
        elif action == 9:#rotate clockwise around bottom-right vertix (big rotation)
            rotate=-self.rotation_angle
            vertix=3
            self.rectangle=self.rotate_around_vertex(self.rectangle,rotate,vertix)
            out=self.check_out_edge()
            if out:
                #load saved position of the truck
                self.rectangle=deepcopy(self.previous_position)
                self.n_times_out+=1
                reward-=self.reward_out_bounds

        next_state={
            "grid":self.grid.middle_points,
            "current_position":self.rectangle,
            "edges":self.grid.vertices
        }

        self.n_step += 1
        reward -= self.reward_n_step

        #If too much steps
        if self.n_step >self.max_steps:
            done1=True

        # plot
        if not out:
            if self.render_step != 0:
                if self.n_step %self.render_step == 0 or self.render_step == 1: 
                    self.render()
        self.total_reward += reward
        return next_state, reward, done,done1, {}


    def reset(self,
        seed=5
    ) -> tuple[dict,dict]:
        """
        Resets the environment and creates a new episode. If training on same environment, resets the environment to the initial state. If not, creates a new environment.
        Parameters:
        -----------
            seed (int):
                Random seed needed by gymnasium library
        Returns:
        --------
            next_state (dict):
                Dictionary with the new environment.
            Further information.

        """
        self.grid.clean()
        print('-'*200)
        print('New reset')
        print(f'Total reward previous episode: {self.total_reward}. Number of episode:{self.n_episode}. Number of squares painted: {self.accumulated}. Number of times out: {self.n_times_out}. Total squares: {self.grid.total_cells}. Number of steps: {self.n_step}. Time: {round(time.time()-self.time,2)} seconds')
        self.time=time.time()
        if self.total_reward >= self.best_score:
            self.best_score=self.total_reward
            self.best_episode=self.n_episode
        self.total_reward=0
        self.n_episode += 1
        self.accumulated=0
        self.n_step=0
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.n_times_out=0

        if self.same_environment:
            self.grid=deepcopy(self.constant_grid)
        else:
            self.create_edge(self.cell_size,self.n_vertices,self.area,self.t_width)

        # # self.rectangle= self.grid.get_truck(self.t_width,self.t_height)
        self.rectangle = self.get_random_truck(self.t_width,self.t_height)
        out=self.check_out_edge()
        while out:
            #if using custom vertices
            if self.same_environment == False:
                #create edge
                self.create_edge(self.cell_size,self.n_vertices,self.area,self.t_width)
            
            # self.rectangle= self.grid.get_truck(self.t_width,self.t_height)
            self.rectangle = self.get_random_truck(self.t_width,self.t_height)

            out=self.check_out_edge()

        self.action_space=Discrete(self.number_of_actions)
        
        next_state={
            "grid":self.grid.middle_points,
            "current_position":self.rectangle,
            "edges":self.grid.vertices
        }

        return next_state, {}


        
    
