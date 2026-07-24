import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import open3d as o3d
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import glob
import os
import random
import pandas as pd
import os.path as osp
import pandas as pd
import time
import shapefile
from shapely.geometry import Polygon
import rasterio
from pyproj import Transformer
import laspy


def fast_hist(pred, label, n):
  k = (label >= 0) & (label < n)
  return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)

def per_class_iu(hist):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def write_ply_xyz(save_path,points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    points = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([points], text=text).write(save_path)

def vectors_angles(vectors,vec2):
    try:
        vectors.shape[1]
    except:
        num_vector = False
    else:
        num_vector = True
    if num_vector is True:
        angles = np.arccos(np.dot(vectors, vec2) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(vec2)))
    else:
        angles = np.arccos(np.dot(vectors, vec2) / (np.linalg.norm(vectors) * np.linalg.norm(vec2)))
    angles = np.rad2deg(angles)
    return angles

def is_point_inside_polygon(point, polygon):
    """
    Determine whether a point is inside a polygon.

    Args:
        point: The point to check, in the form [x, y], where x and y are floats or integers.
        polygon: The polygon, a list of points, each in the form [x, y], where x and y are floats or integers.

    Returns:
        True if the point is inside the polygon; otherwise, False.
    """
    num_intersections = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1) % len(polygon)]
        if point[1] < min(p1[1], p2[1]) or point[1] > max(p1[1], p2[1]):
            continue
        if point[0] > max(p1[0], p2[0]):
            continue
        if p1[1] == p2[1]:
            if point[0] < min(p1[0], p2[0]):
                continue
            else:
                num_intersections += 1
                continue
        x_intersect = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
        if point[0] < x_intersect:
            num_intersections += 1
    return num_intersections % 2 == 1


def point_to_line_distance(point, line_start, line_end):
    line_direction = line_end - line_start
    point_to_start = point - line_start
    t = np.dot(point_to_start, line_direction) / np.dot(line_direction, line_direction)
    projection_point = line_start + t * line_direction
    distance = np.linalg.norm(point - projection_point)
    return distance

def calculate_normal(points):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return normal

def calculate_heading(normal):
    angle = np.arctan2(normal[1], normal[0])
    angle = np.degrees(angle)
    angle = -angle % 360
    angle = (angle + 90) % 360
    return angle

def convert_polyline_to_lines(polyline):
    num_points = len(polyline)
    lines = np.empty((num_points-1, polyline.shape[1] * 2))
    for i in range(num_points-1):
        lines[i] = np.concatenate((polyline[i], polyline[i+1]))
    
    return lines

def calculate_line_normal(point1, point2):
    """
    Calculate the perpendicular direction of a line.

    Args:
        point1: The first point on the line, in the form [x1, y1], where x1 and y1 are floats or integers.
        point2: The second point on the line, in the form [x2, y2], where x2 and y2 are floats or integers.

    Returns:
        A perpendicular direction vector in the form [nx, ny], where nx and ny are floats.

    """
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    normal = np.array([-1/slope, 1])
    normal = normal / np.linalg.norm(normal)
    return normal

def calculate_normal_vector(point_a, point_b):
    slope =(point_b[1]-point_a[1])/(point_b[0]- point_a[0])
    normal_vector=np.array([-1,1/slope])
    return normal_vector

def calculate_angle(normal1, normal2):
    """
    Calculate the angle between two vectors.

    Args:
        normal1: The first vector, in the form [nx1, ny1], where nx1 and ny1 are floats or integers.
        normal2: The second vector, in the form [nx2, ny2], where nx2 and ny2 are floats or integers.

    Returns:
        The angle between the two vectors (in radians).
    """
    print('normal1',normal1)
    print('normal2',normal2)
    dot_product = np.dot(normal1, normal2)
    norm1 = np.linalg.norm(normal1)
    norm2 = np.linalg.norm(normal2)
    cos_theta = dot_product / (norm1 * norm2)
    theta = np.arccos(cos_theta)
    return theta

def mkdirs(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def folder_files(path,type='txt'):
    return glob.glob(path+'/*.'+type, recursive=True)

def random_color_255():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return [r,g,b]  

def read_ply_xyz_label(path,label='label'):
    plydata = PlyData.read(path)
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    if label != None:
        labels = plydata.elements[0].data[label]
        points = np.vstack((x,y,z,labels)).T
    else:
        points = np.vstack((x,y,z)).T
    return points


def read_ply(path,label='label'):
    plydata = PlyData.read(path)
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    r = plydata.elements[0].data['red']
    g = plydata.elements[0].data['green']
    b = plydata.elements[0].data['blue']
    if label != None:
        labels = plydata.elements[0].data[label]
        points = np.vstack((x,y,z,r,g,b,labels)).T
    else:
        points = np.vstack((x,y,z,r,g,b)).T
    return points

def label_filter(array,label_position,type):
    cond = array[:,label_position] == type
    return array[cond]

def npxyz_to_pcd(np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np)
    return pcd

def write_shp(shapname, data):
    with shapefile.Writer(shapname) as w:
        w.field('name', 'C')

        results = data.tolist()

        for result in results:
            w.point(*result)
            w.record('point')
    w.close()

def dict_add(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]
    return d

def random_color_255_list(color_list=None):
    r,g,b = random_color_255()

    if np.array([r,g,b]) in np.array(color_list):
        return random_color_255_list(color_list)
    else:
        return r,g,b

def polygon_inside_intersect(p1,p2):
    p1 = Polygon(p1)
    p2 = Polygon(p2)
    
    if p1.within(p2) or p1.intersects(p2):
        return True
    else:
        return False

def pts_distance(pt1,pt2):
    return np.linalg.norm(pt1 - pt2)

def kdtree(points1,points2, k=10, radius=None):
    """
    kdtree: return index of point1
    """
    tree = KDTree(points1)
    if radius:
        ind, dis  = tree.query_radius(points2,r=radius,return_distance=True)
    else:
        dis, ind = tree.query(points2,k=k)
    return dis, ind


def read_shp(path):
    sf = shapefile.Reader(path)
    shapes = sf.shapes()
    records = sf.records()
    field = sf.fields
    return shapes, records, field

def write_raster(data, save_path, height, width, transform, crs):
    with rasterio.open(save_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype, transform=transform, crs=crs) as dst:
            dst.write(data, 1)

def transform_coords(points, transformer=Transformer.from_crs(2326, 4326)):
    """
    return lat, lon
    """
    lat, lon = transformer.transform(points[1],points[0])
    return np.array([lat, lon]).T

def transform_coords_back(points, transformer=Transformer.from_crs(4326, 2326)):
    """
    return x, y
    """
    y, x = transformer.transform(points[:,1], points[:,0])
    return np.array([x, y]).T

def generate_box(center, size, axis):
    """
    return: box corners
    """
    x_axis = axis[0]  
    y_axis = axis[1]
    z_axis = axis[2]

    length = size[0]
    width = size[1] 
    height = size[2] 

    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    corners = [
        center + half_length * x_axis + half_width * y_axis + half_height * z_axis,
        center + half_length * x_axis + half_width * y_axis - half_height * z_axis,
        center + half_length * x_axis - half_width * y_axis + half_height * z_axis,
        center + half_length * x_axis - half_width * y_axis - half_height * z_axis,
        center - half_length * x_axis + half_width * y_axis + half_height * z_axis,
        center - half_length * x_axis + half_width * y_axis - half_height * z_axis,
        center - half_length * x_axis - half_width * y_axis + half_height * z_axis,
        center - half_length * x_axis - half_width * y_axis - half_height * z_axis,
    ]
    return np.array(corners)

def rotate_vector_around_point(v, p,angles=[60, 120, 180, 240, 300]):
    rotated_vectors = []
    for angle in angles:
        theta = np.deg2rad(angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        v_rotated = np.dot(R, v - p) + p
        v_rotated = normalize_vector(v_rotated.reshape(1,-1)).reshape(-1)
        rotated_vectors.append(v_rotated)
    return rotated_vectors

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calculate_heading(normal):
    angle = np.arctan2(normal[1], normal[0])
    angle = np.degrees(angle)
    angle = -angle % 360
    angle = (angle + 90) % 360
    return angle

def normalize_raster(array):
    array_normal = (array - array.min()) / (array.max() - array.min())
    return array_normal