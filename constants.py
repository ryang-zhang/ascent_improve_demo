coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
]

category_to_id_gibson = [
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv"
]

mp3d_category_id = {
    'void': 1,
    'chair': 2,
    'sofa': 3,
    'plant': 4,
    'bed': 5,
    'toilet': 6,
    'tv_monitor': 7,
    'table': 8,
    'refrigerator': 9,
    'sink': 10,
    'stairs': 16,
    'fireplace': 12
}

# mp_categories_mapping = [4, 11, 15, 12, 19, 23, 6, 7, 15, 38, 40, 28, 29, 8, 17]

mp_categories_mapping = [4, 11, 15, 12, 19, 23, 26, 24, 28, 38, 21, 16, 14, 6, 16]

hm3d_category = [
        "chair",
        "sofa",
        "plant",
        "bed",
        "toilet",
        "tv_monitor",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
]

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]

# NYUv2 数据集的标准 colormap (40 类)
import numpy as np
# NYU40 Color Map: 每个类别对应的 RGB 颜色值
NYU40_COLORMAP = np.array([
    (0, 0, 0),          # 0: Background
    (174, 199, 232),    # 1: Wall
    (152, 223, 138),    # 2: Floor
    (31, 119, 180),     # 3: Cabinet
    (255, 187, 120),    # 4: Bed
    (188, 189, 34),     # 5: Chair
    (140, 86, 75),      # 6: Sofa
    (255, 152, 150),    # 7: Table
    (214, 39, 40),      # 8: Door
    (197, 176, 213),    # 9: Window
    (148, 103, 189),    # 10: Bookshelf
    (196, 156, 148),    # 11: Picture
    (23, 190, 207),     # 12: Counter
    (247, 182, 210),    # 13: Blinds
    (219, 219, 141),    # 14: Desk
    (255, 127, 14),     # 15: Shelves
    (158, 218, 229),    # 16: Curtain
    (44, 160, 44),      # 17: Dresser
    (112, 128, 144),    # 18: Pillow
    (227, 119, 194),    # 19: Mirror
    (82, 84, 163),      # 20: Floor mat
    (123, 102, 210),    # 21: Clothes
    (197, 176, 213),    # 22: Ceiling
    (148, 103, 189),    # 23: Books
    (196, 156, 148),    # 24: Refrigerator
    (23, 190, 207),     # 25: Television
    (247, 182, 210),    # 26: Paper
    (219, 219, 141),    # 27: Towel
    (255, 127, 14),     # 28: Shower curtain
    (158, 218, 229),    # 29: Box
    (44, 160, 44),      # 30: Whiteboard
    (112, 128, 144),    # 31: Person
    (227, 119, 194),    # 32: Nightstand
    (82, 84, 163),      # 33: Toilet
    (123, 102, 210),    # 34: Sink
    (197, 176, 213),    # 35: Lamp
    (148, 103, 189),    # 36: Bathtub
    (196, 156, 148),    # 37: Bag
    (23, 190, 207),     # 38: Other structure
    (247, 182, 210),    # 39: Other furniture
    (219, 219, 141),    # 40: Other prop
], dtype=np.uint8)

import matplotlib.colors as mcolors

# 提取 mpcat40 的类别和对应颜色
MPCAT40_NAMES = [ "", # may need to add this placeholder to align with
                "unknown", "wall", "floor", "chair", "door", "table",  # 0-5
                "picture", "cabinet", "cushion", "window", "sofa", # 6-10
                "bed", "curtain", "night stand", "plant", "sink", # 11-15 
                "stairs", "ceiling", "toilet", "stool", "towel",  # 16-20
                "mirror", "tv_monitor", "shower", "column", "bathtub",   # 21-25
                "counter", "fireplace", "lighting", "beam", "railing",  # 26-30
                "shelving", "blinds", "gym_equipment", "seating", "board_panel",  # 31-35 
                "furniture", "appliances", "clothes", "objects", "misc", "unlabeled",  # 36-41
            ]
MPCAT40_COLORS = [
    "#111111", # may need to add this placeholder to align with
    "#ffffff", "#aec7e8", "#708090", "#98df8a", "#c5b0d5",
    "#ff7f0e", "#d62728", "#1f77b4", "#bcbd22", "#ff9896",
    "#2ca02c", "#e377c2", "#de9ed6", "#9467bd", "#8ca252",
    "#843c39", "#9edae5", "#9c9ede", "#e7969c", "#637939",
    "#8c564b", "#dbdb8d", "#d6616b", "#cedb9c", "#e7ba52",
    "#393b79", "#a55194", "#ad494a", "#b5cf6b", "#5254a3",
    "#bd9e39", "#c49c94", "#f7b6d2", "#6b6ecf", "#ffbb78",
    "#c7c7c7", "#8c6d31", "#e7cb94", "#ce6dbd", "#17becf",
    "#7f7f7f", "#000000"
]

# 将 HEX 转换为 RGB
MPCAT40_RGB_COLORS = np.array([
    mcolors.hex2color(color) for color in MPCAT40_COLORS
]) * 255  # 转换为整数 RGB 值
MPCAT40_RGB_COLORS = MPCAT40_RGB_COLORS.astype(np.uint8)

# Place365 数据集有 365 个场景类别
NUM_CLASSES = 365

def generate_distinct_colors(num_colors):
    """
    生成一组视觉上区分度较高的 RGB 颜色。
    """
    # 使用 HSV 颜色空间生成均匀分布的颜色
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    # 将 HSV 转换为 RGB
    rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsv_colors]
    # 转换为 0-255 范围的整数 RGB 值
    rgb_colors = np.array(rgb_colors) * 255
    rgb_colors = rgb_colors.astype(np.uint8)
    return rgb_colors

# 生成 365 种颜色，加上白色作为第 0 个颜色
PLACE365_RGB_COLORS = np.vstack([[255, 255, 255], generate_distinct_colors(NUM_CLASSES)]).astype(np.uint8)

# 类别名
PROMPT_SEPARATOR = "|"
STAIR_CLASS_ID = 17  # MPCAT40中 楼梯的类别编号是 16 + 1
CHAIR_CLASS_ID = 4  # MPCAT40中 椅子的类别编号是 3 + 1
SOFA_CLASS_ID = 11  # MPCAT40中 沙发的类别编号是 10 + 1
PLANT_CLASS_ID = 15  # MPCAT40中 植物的类别编号是 14 + 1
BED_CLASS_ID = 12  # MPCAT40中 床的类别编号是 11 + 1
TOILET_CLASS_ID = 19  # MPCAT40中 马桶的类别编号是 18 + 1
TV_CLASS_ID = 23  # MPCAT40中 电视的类别编号是 22 + 1

# 关键修改：调整列表项生成方式，确保缩进正确
INDENT_L1 = " " * 4
INDENT_L2 = " " * 8

# 关键修改：动作
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
LOOK_UP = 4
LOOK_DOWN = 5

STICKY_FRONTIER_DISTANCE_THRESHOLD = 0.3
STICKY_FRONTIER_STEP_THRESHOLD = 20
REPEATED_SELECTION_THRESHOLD = 20
MULTI_FLOOR_ASK_STEP_THRESHOLD = 60
FLOOR_EXP_STEP_THRESHOLD = 100

# reference_captions = [[obj] for obj in reference_object_list]
REFERENCE_ROOMS = [
            "bathroom", "bedroom", "dining_room", "garage", "hall",
            "kitchen", "laundry_room", "living_room", "office", "rec_room"
        ]
# 直接映射表
DIRECT_MAPPING = {
    # Bathroom 相关
    "bathroom": "bathroom",
    "shower": "bathroom",
    "jacuzzi/indoor": "bathroom",

    # Bedroom 相关
    "bedroom": "bedroom",
    "bedchamber": "bedroom",
    "dorm_room": "bedroom",
    "hotel_room": "bedroom",
    "childs_room": "bedroom",

    # Dining Room 相关
    "dining_room": "dining_room",
    "dining_hall": "dining_room",
    "banquet_hall": "dining_room",
    "restaurant": "dining_room",
    "cafeteria": "dining_room",

    # Garage 相关
    "garage/indoor": "garage",
    "garage/outdoor": "garage",
    "parking_garage/indoor": "garage",
    "parking_garage/outdoor": "garage",

    # Hall 相关
    "entrance_hall": "hall",
    "lobby": "hall",
    "corridor": "hall",
    "mezzanine": "hall",

    # Kitchen 相关
    "kitchen": "kitchen",
    "restaurant_kitchen": "kitchen",

    # Laundry Room 相关
    "laundry_room": "laundry_room",
    "laundromat": "laundry_room",

    # Living Room 相关
    "living_room": "living_room",
    "home_theater": "living_room",
    "television_room": "living_room",

    # Office 相关
    "office": "office",
    "office_cubicles": "office",
    "conference_room": "office",
    "home_office": "office",
    "computer_room": "office",

    # Rec Room 相关
    "recreation_room": "rec_room",
    "playroom": "rec_room",
    "amusement_arcade": "rec_room",
    "gymnasium/indoor": "rec_room",
    "arcade": "rec_room",
}
