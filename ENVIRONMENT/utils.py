import numpy as np
from matplotlib.path import Path
def move_rectangle(

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

def check_out_edge(
    rectangle,
    vertices
    ) -> bool:
    """
    Checks if the truck is out of the figure.
    Parameters:
    -----------
        None
    Returns:
    --------
        Whether the truck is out of the figure
    """
    # Check if all vertices of the truck are within the figure

    for vertix in rectangle:
        if not is_point_in_polygon(vertix,vertices):
            return True
    return False

def rotate_around_vertex(

    vertices : list[list], 
    theta : float, 
    vertex_index : int =0
    ) -> list[list]:
    
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

def get_forward_direction(
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

def is_point_in_polygon(

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