"""
This config file holds all config values necessary to run this program.
"""
"""
Book and Tag parameters
"""
BOOK_WIDTH = 12
BOOK_HEIGHT = 18
TAG_SIZE = 2.4
BOOK_TAG_RATIO_HEIGHT = BOOK_HEIGHT/TAG_SIZE
BOOK_TAG_RATIO_WIDTH = BOOK_WIDTH/TAG_SIZE

"""
Calibration
"""
CALIBRATION_PATH = 'Setup/Camera/calibration.yml' #changed
CAM_K = 'cam_K'
CAM_KC = 'cam_kc'
PROJ_K = 'proj_K'
PROJ_KC = 'proj_kc'
R = 'R'
T = 'T'
CAM_DEPTH_K = 'cam_depth_K' #Emma
CAM_DEPTH_KC = 'cam_depth_kc' #Emma
DATA = 'data'
DEPTH_SCALE = 1
DEPTH_TRUNC = 2100#1500
VOXEL_SIZE = 5#1#5
CALIBRATION_SCALING = 1#2

"""
CMA-ES
"""
START_PARAMS = tuple([6, 0, 6, 0, 0])
MESH_JSON_PATH = '/home/vigitia/Desktop/repositories/DynamicProjectionMapping/meshDeformations_13x19.json'
CMA_ES_POPULATION_SIZE = 8
CMA_ES_ITERATIONS = 10
CMA_ES_TOP_N = int(CMA_ES_POPULATION_SIZE / 4)
CMA_ES_SIGMA = 1
CMA_ES_STOP = 15
CMA_ES_SOFT_STOP = 50
LOCAL_MINIMA_THRESHOLD = 0.9
CMA_ES_MIN_PROGRESS = 4
MIN_SIGMA = 1

"""
ImageDeformer
"""
IMAGE_PATH = '../../02_assets/TestImages/marked.png' #'../../02_assets/TestImages/chess_board.png' #
IMAGE_PADDING = 5

"""
ImageDenoiser
"""
FILTER_D = 50#5#15
FILTER_COLOR = 100#5#75
FILTER_SPACE = 100#5#75

"""
Mesh Parameters
"""
MESH_WIDTH = 13
MESH_HEIGHT = 19
MESH_MAX_BENDING_VALUE = 5

"""
ProjectionService
"""
PROJECTION_WINDOW_NAME = 'Projection'
PROJECTION_IMAGE_SHAPE = (1080, 1920, 3)
SKIPPED_VERTICES = 3
VERTEX_RADIUS = 1
VERTEX_COLOR = (0, 0, 255)
VERTEX_THICKNESS = -1
BUTTON_PROJ_NEXT = ' '

"""
RunManager
"""
NUMBER_OF_ITERATIONS = 1000
LATENCY_JSON_PATH = '/media/vigitia/CCCOMA_X64FRE_DE-DE_DV9/results/runs.json'
EVALUATION_VIDEO_RGB_DIR = "/media/vigitia/CCCOMA_X64FRE_DE-DE_DV9/location_movement/rgb/"
EVALUATION_VIDEO_DEPTH_DIR = "/media/vigitia/CCCOMA_X64FRE_DE-DE_DV9/location_movement/depth/"
MAX_STEPS_WITHOUT_PROGRESS = 10
OPTIMIZATION_JSON_PATH = '/media/vigitia/CCCOMA_X64FRE_DE-DE_DV9/results/runs.json'
ALIGNMENT_PRE_OPTIMIZATION = 20
ALIGNMENT_DIR = '/media/vigitia/CCCOMA_X64FRE_DE-DE_DV9/results/'
ALIGNMENT_RUNS = 10
ALIGNMENT_JSON = '_jitter.json'
ALIGNMENT_RGB = "_rgb_jitter.png"
ALIGNMENT_DEPTH = "_depth_jitter.png"
ALIGNMENT_FITNESS = ALIGNMENT_DIR + "fitness.json"
ALIGNMENT_FITNESS_RMSE = ALIGNMENT_DIR + "fitness_rmse.json"

"""
TrackingService
"""
TAG_ID = 0
TAG_ID_MARKER_PATTERN = 17
Y_CUTOFF_MIN = 800
Y_CUTOFF_MAX = 2150
X_CUTOFF_MIN = 700
X_CUTOFF_MAX = 3200
