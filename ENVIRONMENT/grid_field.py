from matplotlib.patches import Polygon as MPolygon
from shapely.geometry import Polygon, box, Point, LineString, GeometryCollection,MultiPolygon
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path
import math
from sklearn.preprocessing import MinMaxScaler
from shapely import area





class Grid_field():
    def __init__(
            self,
            vertices,
            cell_size,
            add_padding
        )-> None:
        """
        Constructor of the grid in which the agent will be training on.
        Parameters:
        -----------
            vertices (list):
                Vertices of the figure (x,y coordinates).
            cell_size (float):
                Cell size of the grid.
            add_padding (bool):
                Whether padding is needed.
        Returns:
        --------
            None.
        """
        self.vertices=vertices
        self.cell_size=cell_size

        #Scale the vertices
        self.scale = MinMaxScaler().fit(self.vertices)
        self.vertices=self.scale.transform(vertices)
        self.edges=Polygon(self.vertices)

        #Create the grid
        self.create_square_grid_with_clipping(0.001)

        #Order the grid cells, from top to bottom and from left to right
        self.grid = sorted(self.grid, key=lambda square: (-self.get_middle_point(square)[1], self.get_middle_point(square)[0]))

        #Obtain middel point of each cell
        self.get_middle_points()


        #Add padding
        self.normalize_grid(add_padding)

        #Params for representation
        plt.rcParams['figure.figsize'] = [20, 15]
        
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
    
    def get_middle_points(
            self
        ) -> None:
        """
        Calculates the middle point of each grid cell and store them in self.middle_points.
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        """
        self.middle_points=[]
        for cell in self.grid:
            x=np.mean(cell,axis=0).tolist()
            x.append(0)
            self.middle_points.append(x)

    def get_middle_point(
            self,
            cell : list[list]
        ) -> np.ndarray:
        """
        Computes the middle point of a single grid.
        Parameters:
        -----------
            cell (list[list]):
                Vertices of the cell (x,y coordinates)
        Returns:
        --------
            Middle point of the cell.
        """
        x=np.mean(cell,axis=0).tolist()
        return x
    
    def is_square(
            self,
            cell : list[list]
        ) -> bool:
        """
        Check if a cell is in the edge of the polygon
        Parameters:
        -----------
            cell (list[list]):
                Contains (x,y) coordinates of the vertices of the cell
        Returns:
            Whether the cell is in the edge
        """
        if len(cell.exterior.coords) - 1 != 4:  
            return False
        
        # Extract the coordinates of the vertices
        coords = list(cell.exterior.coords)
        
        # Calculate the lengths of the four sides
        side_lengths = [
            math.dist(coords[i], coords[i + 1]) for i in range(4)
        ]
        
        # Check if all sides are equal
        if not all(math.isclose(side, side_lengths[0], rel_tol=1e-9) for side in side_lengths):
            return False
        return True

    
    def reset_borders(
            self
        ) -> None:
        """
        Asign different values for edge cells.
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        """
        for index,cell in enumerate(self.grid):
            type_cell=int(cell[6][0])
            if type_cell != -1:
                points=cell[0:type_cell,:]
                points=Polygon(points)
                if not self.is_square(points):
                    self.grid[index][6][1]=-1


    def normalize_grid(
            self,
            add_padding
        ):
        """
        Normalize the number of cells, as Gymnasium framework needs same observation spaces, and the number of cells in each episode can differ.
        Structure of cell grid:
        Can be formed of 3-6 vertices. First 6 lists will be those vertices (x,y). If number of vertices is <6, the rest will be padded with (0,0)
        First element of 7th position will be the number of real vertices of the cell (3-6). If -1, the cell is padding.
        Last element of 7th position will be 0 if it is a non-edge cell not passed, 1 if non-edge cell passed, -1 if a edge cell not passed and 11 
        if edge cell passed. E.g:
            [[0,0.5],[0.7,0.7],[0,1],[0,0],[0,0],[0,0],[3,0]] -> Triangle not passed
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-1,0]] -> Padding
            [[0,0.5],[0.7,0.7],[1,1],[1,0],[0,0][0,0],[4,-1]] -> Square in the edge not passed
        Parameters:
        -----------
            n_total_cells (int):
                Number of total cells that will have the grid. This number must be greater than the number of cells of the initial grid.
        Returns:
        --------
            None
        
        """

        #Padding
        padding=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-1,0]] 

        self.type_cell=[]
        for cell in self.grid:
            type_cell=len(cell)
            while len(cell)!=6:
                cell.append([0,0])
            cell.append([type_cell,0])

        self.total_cells=len(self.grid)
        
        #Add padding
        if add_padding:
            n_total_cells=1.2*self.total_cells
            self.total_cells_padded=n_total_cells
            while len(self.grid) != n_total_cells:

                #Add padding to cell grids
                self.grid.append(padding)

                #Add padding to middle_points
                self.middle_points.append([-1,-1,-1])
        else:
            self.total_cells_padded=self.total_cells
        
        self.middle_points=np.array(self.middle_points)
        self.grid=np.array(self.grid)

        #Compute cell in edge
        self.reset_borders()

    def get_truck(
            self,
            width : float,
            height : float
        ) -> list[list]:
        """
        Initialize the truck in one of the edges.
        Parameters:
        -----------
            width (int):
                Width of the truck.
            height (int):
                Height of the truck
        Returns:
            List of shape (4,2) for x,y coordinates for each vertix
        """
        control=False
        while control == False:
            #Get two consecutive random edges
            edge_index = random.randrange(0, len(self.vertices))
            start_point = self.vertices[edge_index]
            end_point = self.vertices[(edge_index + 1) % len(self.vertices)] 

            # Calculate the direction vector of the edge
            direction = start_point - end_point

            # Calculate the length of the edge
            edge_length = np.linalg.norm(direction)

            if edge_length >= width:
                control=True

        # Normalize the direction vector
        direction_normalized = direction / edge_length


        # Calculate the two points on the edge separated by the given distance
        point1 = start_point 
        point2 = start_point - direction_normalized * width
        rectangle=self.create_rectangle(point1,point2,height)

        while len(rectangle) != 4:
            rectangle=self.create_rectangle(point1,point2,height)
        return rectangle

    def create_rectangle(
            self,
            point1 : list, 
            point2 : list, 
            height : float
        ):
        """
        Create a rectangle given the bottom vertices of the rectangle and its height.
        Parameters:
        -----------
            point1 (list):
                First point (x1, y1).
            point2 (list):
                First point (x2, y2).
            height (float):
                Height of the rectangle.
        Returns:
        --------
            List with the corners (x,y) of the rectangle
        """

        p1 = np.array(point1)
        p2 = np.array(point2)

        # Calculate the direction vector of the line segment
        direction = p2 - p1
        length = np.linalg.norm(direction)

        # Normalize the direction vector
        if length == 0:
            raise ValueError("The two points must be different to form a line segment.")
        direction_normalized = direction / length

        # Calculate the perpendicular vector
        perpendicular = np.array([-direction_normalized[1], direction_normalized[0]])

        # Calculate the rectangle corners
        corner1 = p1 
        corner2 = p2 
        corner3 = p2 + perpendicular * height
        corner4 = p1 + perpendicular * height

        corners=[corner1,corner2,corner3,corner4]
        rectangle=Polygon(corners)
        intersection = rectangle.intersection(self.edges)

        #if it is incorrect
        if isinstance(intersection, LineString):
            #corners in a clockwise order
            corner1 = p1 
            corner2 = p2 
            corner3 = p2 - perpendicular * height
            corner4 = p1- perpendicular * height

        corners=np.array([corner1, corner2, corner3, corner4])

        return corners

    
    def is_polygon_in_rectangle(
            self,
            rectangle : list[list]
        ) -> bool:
        """
        Check if a polygon is fully inside the figure.
        Parameters:
        -----------
            rectangle (list[list]):
                Polygon.
        Returns:
            Whether the polygon is fully within the figure.
        """
        for i in rectangle:
            if not self.is_point_in_polygon(i, self.edges):
                return False
        return True

    def create_square_grid_with_clipping(
            self,
            min_area : float
        ) -> None:
        """
        Creates the grid cells.
        Parameters:
        -----------
            min_area (float):
                Minimum area of a cell.
        Returns:
        --------
            None
        """
        # Create a polygon from the vertices
        polygon = Polygon(self.vertices)

        # Calculate the bounding box of the polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        self.grid = []

        # Loop over the bounding box to create square cells
        for x in np.arange(min_x, max_x, self.cell_size):
            for y in np.arange(min_y, max_y, self.cell_size):
                # Define the square cell as a shapely box
                square = box(x, y, x + self.cell_size, y + self.cell_size)

                # Get the intersection of the square with the polygon
                intersection = square.intersection(polygon)

                # Check if the intersection is a valid figure
                if intersection.is_empty:
                    continue  

                if isinstance(intersection, Point):
                    continue

                if isinstance(intersection, LineString):
                    continue
                if isinstance(intersection, GeometryCollection):
                    for geom in intersection.geoms:
                        if isinstance(geom, LineString):
                            continue
                        if isinstance(geom, Point):
                            continue
                        self.grid.append(list(geom.exterior.coords[:-1]))
                    continue
                if isinstance(intersection, MultiPolygon):
                    for geom in intersection.geoms:
                        if isinstance(geom, LineString):
                            continue
                        if isinstance(geom, Point):
                            continue
                        self.grid.append(list(geom.exterior.coords[:-1]))
                    continue

                #Eliminate cells with small area (difficult to converge)
                if area(intersection) < min_area:
                    continue
                self.grid.append(list(intersection.exterior.coords[:-1]))

    def clean(
            self
        ) -> None:
        """
        Cleans the matplotlib plot to render a new plot.
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        """
        plt.clf()
        

    def plot_polygon_and_cells(
            self,
            truck : list[list],
            boolean : bool,
            total_reward : float,
            n_step : int,
            n_episode : int,
            best_episode : int,
            score : float,
            ) -> None:

        """
        Plot the polygon, passed and non-passed cells, middle points and truck 

        Parameters:
        -----------
            truck (list[list]):
                Truck. Each vertix will have a color, to represent its direction.
            boolean (bool):
                Whether it is the first representation.
            total_reward (float):
                Total reward of the current_episode.
            n_step (int):
                Number of steps done in the episode.
            n_episode (int):
                Number of current episode.
            best_episode (int):
                Episode in which the reward was the highest.
            score (float):
                Highest score gathered.
        Returns:
        --------
            None.
        """
        
        if boolean:
            plt.clf()
        else:
            plt.figure()
            plt.ion()

        # Plot the figure
        polygon = MPolygon(self.vertices, closed=True, fill=None, edgecolor='blue', linewidth=2)
        plt.gca().add_patch(polygon)

        # Plot each square cell
        for cell in self.grid:
            passed=int(cell[6][1])
            type_polygon = int(cell[6][0])
            if type_polygon == -1:
                continue
            cell_array=[]
            cell_array=cell[0:type_polygon,:]
            cell_array = np.array(cell_array)

            if passed==1:
                plt.fill(cell_array[:, 0], cell_array[:, 1], edgecolor='green', fill='green', linewidth=0,facecolor='green')

            # if using the borders cells differently from non-edge cells
            # elif passed == 11:
            #     plt.fill(cell_array[:, 0], cell_array[:, 1], edgecolor='yellow', fill='yellow', linewidth=0,facecolor='yellow')
            elif passed==0 or passed ==-1:
                plt.fill(cell_array[:, 0], cell_array[:, 1], edgecolor='red', fill=None, linewidth=0.1)

        # Plot middle points
        for i in self.middle_points:
            if i[2] == 1:
                plt.scatter(i[0],i[1],color='green',s=1)
            else:
                plt.scatter(i[0],i[1],color='red',s=1)

        # Plot truck
        polygon = MPolygon(truck, closed=True, fill=None, edgecolor='black', linewidth=0.5)
        plt.gca().add_patch(polygon)
        plt.plot(truck[0][0], truck[0][1], 'o', color='purple', markersize=10, label='Truck 0')  # 'o' for circle
        plt.plot(truck[1][0], truck[1][1], 'o', color='blue', markersize=10, label='Truck 1')
        plt.plot(truck[2][0], truck[2][1], 'o', color='orange', markersize=10, label='Truck 2')
        plt.plot(truck[3][0], truck[3][1], 'o', color='yellow', markersize=10, label='Truck 3')
        # plt.xlim((0,1))
        # plt.ylim((0,1))
        # plt.axis('off') # Turn off the axis
        # plt.savefig('salvado.png',bbox_inches='tight')
        

        # Set limits and aspect
        plt.figtext(0.5, 0.95, f'Episode number: {n_episode}; Number of steps: {n_step}; Total reward: {total_reward}.', ha='center', fontsize=14, color='black')
        plt.figtext(0.5, 0.93, f'Best episode: {best_episode}; Best reward: {score}.', ha='center', fontsize=14, color='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("DRL Plot")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('equal')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.grid()
        plt.draw()
        plt.pause(0.00001)


