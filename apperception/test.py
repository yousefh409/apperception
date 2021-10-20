### IMPORTS
import cv2

from world import *
from world_executor import *
from video_util import *
from metadata_util import *
import lens
import point

#import tasm

### Let's define some attribute for constructing the world first
name = 'traffic_scene' # world name
units = 'metrics'      # world units
video_file = './amber_videos/traffic-scene-shorter.mp4' #example video file
lens_attrs = {'fov': 120,
              'cam_origin': (0, 0, 0),
              'skew_factor': 0}
point_attrs = {'p_id': 'p1',
               'cam_id': 'cam1',
               'x': 0,
               'y': 0,
               'z': 0,
               'time': None,
               'type':'pos'}
camera_attrs = {'ratio': 0.5}
fps = 30

### First we define a world
w1 = World(name=name, units=units)

### Secondly we construct the camera
fov, res, cam_origin, skew_factor = lens_attrs['fov'], [1280, 720], lens_attrs['cam_origin'], lens_attrs['skew_factor']
cam_lens = lens.PinholeLens(res, cam_origin, fov, skew_factor)

pt_id, cam_id, x, y, z, time, pt_type = point_attrs['p_id'], point_attrs['cam_id'], point_attrs['x'], point_attrs['y'], point_attrs['z'], point_attrs['time'], point_attrs['type']
location = point.Point(pt_id, cam_id, x, y, z, time, pt_type)

ratio = camera_attrs['ratio']


w1.execute()
w2 = w1.camera(cam_id=cam_id,
               location=location,
               ratio=ratio,
               video_file=video_file,
               metadata_identifier=name+"_"+cam_id,
               lens=cam_lens)
w2.execute()

print(w1.get_camera()) # should be empty
print(w2.get_camera()) # should contain one camera