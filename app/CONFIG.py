#projector ViewSonic LS700-4K
#PROJECTOR_RESOLUTION = [3840, 2160]
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

#my mac
#WINDOW_WIDTH = 2880
#WINDOW_HEIGHT = 1694

#projector
#WINDOW_WIDTH = 1920
#WINDOW_HEIGHT = 1080

#WINDOW_WIDTH = 960
#WINDOW_HEIGHT = 540

#CAMERA
# WINDOW_WIDTH = 4096
# WINDOW_HEIGHT = 3072

#CAMERA
SHAPE = (1280, 720)

# head_nr = 2
# img_nr = 1
# DEPTH_IMG_PATH = f"CameraImages_Person{head_nr}/depth_np_{img_nr:02d}.npy"
# RGB_IMG_PATH = f"CameraImages_Person{head_nr}/rgb_{img_nr:02d}.png"

# MODEL_PATH = f"models/head_person{head_nr}/head_p{head_nr}.obj"
# MATERIAL_PATH = f"models/head_person{head_nr}/head_p{head_nr}.jpg"

# # MODEL_PATH = f"models/study_4/study_4_head.obj"
# # MATERIAL_PATH = f"models/study_4/study_4_black.png"

# OBJECT_SCALE = 1000.0

# # Person 1
# if head_nr == 1:
#     ROTATIONS = {
#         "Nose up": (0, -90, 0),
#         "Left ear up": (180, 0, 0),
#         "Right ear up": (0, 0, 0),
#         "Default": (0, 0, 0)
#     }

# elif head_nr == 5 or head_nr == 8:
#     ROTATIONS = {
#         "Nose up": (0, -30, 0),
#         "Left ear up": (0, -120, 0),
#         "Right ear up": (0, 60, 0),
#         "Default": (0, 0, 0)
#     }

# else:
#     ROTATIONS = {
#         "Nose up": (0, 0, 0),
#         "Left ear up": (0, -90, 0),
#         "Right ear up": (0, 90, 0),
#         "Default": (0, 0, 0)
#     }

# printed head
DEPTH_IMG_PATH = "Images_Head_New/depth_np_00.npy"
RGB_IMG_PATH = "Images_Head_New/rgb_00.png"

MODEL_PATH = "models/head/head_small.obj"
MATERIAL_PATH = "models/head/head_material.png"

OBJECT_SCALE = 1.0#0.99 #3D Print Shrinking

ROTATIONS = {
    "Left ear up": (0, -90, 0),
    "Right ear up": (0, 90, 0),
    "Nose up": (0, 0, 0),
    "Default": (0, 0, 0)
}

# printed bunny
#DEPTH_IMG_PATH = "Images_Bunny_New/depth_np_00.npy"
#RGB_IMG_PATH = "Images_Bunny_New/rgb_00.png"

VOXEL_SIZE = 14.0#12.0 #  12#16

MARGIN_TOP = 0.9
MARGIN_BOTTOM = 0.9
MARGIN_SIDES = 1.2