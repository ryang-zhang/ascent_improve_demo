from typing import Dict, Tuple, Any, Union, List, Optional
from habitat_baselines.common.tensor_dict import TensorDict
import numpy as np
import cv2
import os
from ascent.mapping.object_point_cloud_map import ObjectPointCloudMap
from ascent.mapping.obstacle_map import ObstacleMap
from ascent.mapping.value_map import ValueMap
from constants import (
    PROMPT_SEPARATOR,
)
from torch.autograd import Variable as V
from torch.nn import functional as F
from model_api.blip2itm_out import BLIP2ITMClient
from vlfm.utils.geometry_utils import get_fov, rho_theta
from PIL import Image
from copy import deepcopy
from vlfm.vlm.coco_classes import COCO_CLASSES
from model_api.grounding_dino_out import GroundingDINOClient, ObjectDetections
from model_api.ram_out import RAMClient
from model_api.sam_out import MobileSAMClient
from model_api.dfine_out import DFineClient
from ascent.utils import (
    xyz_yaw_pitch_roll_to_tf_matrix,
    check_stairs_in_upper_50_percent,
    load_place365_categories,
    load_floor_probabilities_by_dataset,
    extract_room_categories,
    get_action_tensor,
    load_place365_model,
)
from torchvision import transforms as trn

class Map_Controller:
    MAP_SIZE = 1600
    IMAGE_RESIZE_DIM = 256
    IMAGE_CROP_DIM = 224
    RGB_NORM_MEAN = [0.485, 0.456, 0.406]
    RGB_NORM_STD = [0.229, 0.224, 0.225]
    """
    MapController 接口，负责管理地图。
    StairNavigationHandler 将通过此接口与地图交互。
    """
    def __init__(self, device, num_envs: int, map_size: int = 1600, text_prompt: str = "", object_map_erosion_size: int = 1600,
                 min_obstacle_height: float = 0.2, max_obstacle_height: float = 1.2, obstacle_map_area_threshold: float = 0.2, 
                 agent_radius: float = 0.2, hole_area_thresh: float = 0.2, use_max_confidence: bool = True,
                 coco_threshold: float = 0.2, non_coco_threshold: float = 0.2):
        self.device = device
        # 包含每个环境的地图实例列表
        self._num_envs = num_envs
        self._map_size = map_size
        self._text_prompt = text_prompt
        self._object_map_erosion_size = object_map_erosion_size
        self.min_obstacle_height = min_obstacle_height
        self.max_obstacle_height = max_obstacle_height
        self.obstacle_map_area_threshold = obstacle_map_area_threshold
        self.agent_radius = agent_radius
        self.hole_area_thresh = hole_area_thresh
        self.use_max_confidence = use_max_confidence
        self.target_detection_list: List[Any] = [None] * self._num_envs
        self.coco_detection_list: List[Any] = [None] * self._num_envs
        self.non_coco_detection_list: List[Any] = [None] * self._num_envs
        text_prompt_channels = len(self._text_prompt.split(PROMPT_SEPARATOR)) if hasattr(self, '_text_prompt') else 1 # 确保 _text_prompt 已初始化
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold
        # 初始化所有楼层的地图列表 (统一处理，避免重复逻辑)
        self._object_map_list: List[List[ObjectPointCloudMap]] = [[] for _ in range(self._num_envs)]
        self._obstacle_map_list: List[List[ObstacleMap]] = [[] for _ in range(self._num_envs)]
        self._value_map_list: List[List[ValueMap]] = [[] for _ in range(self._num_envs)]
        for env_idx in range(self._num_envs):
            self._object_map_list[env_idx].append(ObjectPointCloudMap(
                erosion_size=self._object_map_erosion_size, size=self.MAP_SIZE
            ))
            self._obstacle_map_list[env_idx].append(ObstacleMap(
                min_height=self.min_obstacle_height, max_height=self.max_obstacle_height,
                area_thresh=self.obstacle_map_area_threshold, agent_radius=self.agent_radius,
                hole_area_thresh=self.hole_area_thresh, size=self.MAP_SIZE,
            ))
            self._value_map_list[env_idx].append(ValueMap(
                value_channels=text_prompt_channels,
                use_max_confidence=self.use_max_confidence,
                obstacle_map=None, size=self.MAP_SIZE,
            ))
        
        self.floor_num: List[int] = [len(self._obstacle_map_list[env]) for env in range(self._num_envs)]

        # 当前活跃地图的引用
        self._cur_floor_index: List[int] = [0] * self._num_envs
        self._object_map: List[ObjectPointCloudMap] = [self._object_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]
        self._obstacle_map: List[ObstacleMap] = [self._obstacle_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]
        self._value_map: List[ValueMap] = [self._value_map_list[env][self._cur_floor_index[env]] for env in range(self._num_envs)]

        # 爬楼梯状态
        self._done_initializing: List[bool] = [False] * self._num_envs
        self._initialize_step: List[int] = [0] * self._num_envs
        self._reach_stair: List[bool] = [False] * self._num_envs
        self._reach_stair_centroid: List[bool] = [False] * self._num_envs
        self._carrot_goal_xy: List[List] = [[] for _ in range(self._num_envs)]
        self._last_carrot_xy: List[List] = [[] for _ in range(self._num_envs)]
        self._last_carrot_px: List[List] = [[] for _ in range(self._num_envs)]
        self._climb_stair_flag: List[int] = [0] * self._num_envs
        self._stair_dilate_flag: List[bool] = [False] * self._num_envs
        self._stair_frontier: List[Any] = [None] * self._num_envs
        self._climb_stair_over: List[bool] = [True] * self._num_envs
        self._temp_stair_map: List[List] = [[] for _ in range(self._num_envs)]
        self._get_close_to_stair_step: List[int] = [0] * self._num_envs
        self._frontier_stick_step: List[int] = [0] * self._num_envs

        # itm, vlm
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "13182")))
        self._blip_cosine: List[float] = [0] * self._num_envs
        self._target_object: List[str] = ["" for _ in range(self._num_envs)]
        self.place365_classes = load_place365_categories("third_party/places365/categories_places365.txt")
        self.place365_centre_crop = trn.Compose([
            trn.Resize((self.IMAGE_RESIZE_DIM, self.IMAGE_RESIZE_DIM)),
            trn.CenterCrop(self.IMAGE_CROP_DIM),
            trn.ToTensor(),
            trn.Normalize(self.RGB_NORM_MEAN, self.RGB_NORM_STD)
        ])
        self.scene_classify_model = load_place365_model(arch='resnet50', device=self.device)
        # 2. 统一端口获取和外部服务客户端初始化
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "13184")))
        self._coco_object_detector = DFineClient(port=int(os.environ.get("DFINE_PORT", "13186")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "13183")))
        self._ram = RAMClient(port=int(os.environ.get("RAM_PORT", "13185")))   

        ## 目标相关
        self._double_check_goal: List[bool] = [False] * self._num_envs
        self.cur_dis_to_goal: List[float] = [np.inf] * self._num_envs
        self._passive_up_stair_steps: List[int] = [0] * self._num_envs
        self._passive_down_stair_steps: List[int] = [0] * self._num_envs
        self.PASSIVE_STAIR_DETECTION_THRESHOLD = 3  # 连续3步在楼梯区域内即触发
    def reset(self, env: int) -> None:
        self._cur_floor_index[env] = 0
        # 确保只保留第一层的地图实例并重置它们
        # 如果需要删除多余楼层，则执行以下操作：
        del self._object_map_list[env][1:]
        del self._value_map_list[env][1:]
        del self._obstacle_map_list[env][1:]
            
        # 重新建立对当前（第一层）地图的引用
        self._object_map[env] = self._object_map_list[env][0]
        self._value_map[env] = self._value_map_list[env][0]
        self._obstacle_map[env] = self._obstacle_map_list[env][0]

        # 重置地图内部状态
        self._object_map[env].reset()
        self._value_map[env].reset()
        self._obstacle_map[env].reset()

        ## 楼层管理
        self.floor_num[env] = len(self._obstacle_map_list[env]) # 更新楼层数 (此时应为1)

        ## 爬楼梯状态
        self._initialize_step[env] = 0
        self._done_initializing[env] = False
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._carrot_goal_xy[env] = []
        self._last_carrot_xy[env] = []
        self._last_carrot_px[env] = []
        self._stair_dilate_flag[env] = False
        self._climb_stair_flag[env] = 0 
        self._climb_stair_over[env] = True
        self._temp_stair_map[env] = []
        self._get_close_to_stair_step[env] = 0 # VLM/爬楼梯相关
        self._frontier_stick_step[env] = 0
        self._blip_cosine[env] = 0 # VLM相关

        ## 目标检测
        self._target_object[env] = ""
        self.target_detection_list[env] = None
        self.coco_detection_list[env] = None
        self.non_coco_detection_list[env] = None

        # 找目标状态
        self._double_check_goal[env] = False
        self.cur_dis_to_goal[env] = np.inf
        self._passive_up_stair_steps[env] = 0
        self._passive_down_stair_steps[env] = 0
    def is_robot_in_stair_map_fast(self, env: int, robot_px:np.ndarray, stair_map: np.ndarray):
        """
        高效判断以机器人质心为圆心、指定半径的圆是否覆盖 stair_map 中值为 1 的点。

        Args:
            env: 当前环境标识。
            stair_map (np.ndarray): 地图的 _stair_map。
            robot_xy_2d (np.ndarray): 机器人质心在相机坐标系下的 (x, y) 坐标。
            agent_radius (float): 机器人在相机坐标系中的半径。
            obstacle_map: 包含坐标转换功能和地图信息的对象。

        Returns:
            bool: 如果范围内有值为 1,则返回 True,否则返回 False。
        """
        x, y = robot_px[0, 0], robot_px[0, 1]

        # 转换半径到地图坐标系
        radius_px = self.agent_radius * self._obstacle_map[env].pixels_per_meter

        # 获取地图边界
        rows, cols = stair_map.shape
        x_min = max(0, int(x - radius_px))
        x_max = min(cols - 1, int(x + radius_px))
        y_min = max(0, int(y - radius_px))
        y_max = min(rows - 1, int(y + radius_px))

        # 提取感兴趣的子矩阵
        sub_matrix = stair_map[y_min:y_max + 1, x_min:x_max + 1]

        # 创建圆形掩码
        y_indices, x_indices = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius_px ** 2

        # 获取sub_matrix中为 True 的坐标
        if np.any(sub_matrix[mask]):  # 在圆形区域内有值为True的元素
            # 找出sub_matrix中值为 True 的位置
            true_coords_in_sub_matrix = np.column_stack(np.where(sub_matrix))  # 获取相对于sub_matrix的坐标

            # 通过mask过滤,只留下圆形区域内为 True 的坐标
            true_coords_filtered = true_coords_in_sub_matrix[mask[true_coords_in_sub_matrix[:, 0], true_coords_in_sub_matrix[:, 1]]]

            # 将相对坐标转换为 stair_map 中的坐标
            true_coords_in_stair_map = true_coords_filtered + [y_min, x_min]
            
            return True, true_coords_in_stair_map
        else:
            return False, None

    def _add_floor_map(self, env: int, index: int):
        """添加新的楼层地图"""
        new_object_map = ObjectPointCloudMap(erosion_size=self._object_map_erosion_size, size=1600)
        new_obstacle_map = ObstacleMap(
            min_height=self.min_obstacle_height, max_height=self.max_obstacle_height,
            area_thresh=self.obstacle_map_area_threshold, agent_radius=self.agent_radius,
            hole_area_thresh=self.hole_area_thresh, size=1600,
        )
        new_value_map = ValueMap(
            value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=self.use_max_confidence, obstacle_map=None, size=1600,
        )
        self._object_map_list[env].insert(index, new_object_map)
        self._obstacle_map_list[env].insert(index, new_obstacle_map)
        self._value_map_list[env].insert(index, new_value_map)
        self.floor_num[env] = len(self._obstacle_map_list[env])

    def _remove_floor_map(self, env: int, index: int):
        """移除指定的楼层地图"""
        del self._object_map_list[env][index]
        del self._value_map_list[env][index]
        del self._obstacle_map_list[env][index]
        self.floor_num[env] -= 1

    def _update_current_maps(self, env: int):
        """根据当前楼层索引更新当前环境的地图引用"""
        self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
        self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
        self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]

    def _process_stair_climb_state(self, env: int, robot_xy: np.ndarray, robot_px: np.ndarray, stair_map: np.ndarray, climb_direction: int):
        """
        处理机器人达到或离开楼梯的状态逻辑。
        climb_direction: 1 表示上楼，2 表示下楼
        """
        already_reach_stair, reach_yx = self.is_robot_in_stair_map_fast(env, robot_px, stair_map)

        if not self._reach_stair[env]:
            if already_reach_stair:
                self._reach_stair[env] = True
                self._get_close_to_stair_step[env] = 0
                if climb_direction == 1:
                    self._obstacle_map[env]._up_stair_start = robot_px[0].copy()
                else: # climb_direction == 2
                    self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
        elif not self._reach_stair_centroid[env]:
            if self._stair_frontier[env] is not None and \
               np.linalg.norm(self._stair_frontier[env] - np.atleast_2d(robot_xy)) <= 0.3: # 注意这里原来是robot_xy_2d，改为robot_px[0]以匹配px_to_xy的转换
                self._reach_stair_centroid[env] = True
        else: # _reach_stair_centroid[env] == True
            if not self.is_robot_in_stair_map_fast(env, robot_px, stair_map)[0] and \
               self._obstacle_map[env]._climb_stair_paused_step >= 30:
                self._reset_stair_climb_state(env)
                self._climb_stair_over[env] = True
                self._obstacle_map[env]._disabled_frontiers.add(tuple(self._stair_frontier[env][0]))
                print(f"Frontier {self._stair_frontier[env]} is disabled due to no movement.")

                if climb_direction == 1:
                    self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
                    self._obstacle_map[env]._up_stair_map.fill(0)
                    self._obstacle_map[env]._has_up_stair = False
                    self._remove_floor_map(env, self._cur_floor_index[env] + 1)
                else: # climb_direction == 2
                    self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
                    self._obstacle_map[env]._down_stair_frontiers.fill(0)
                    self._obstacle_map[env]._has_down_stair = False
                    self._obstacle_map[env]._look_for_downstair_flag = False
                    self._remove_floor_map(env, self._cur_floor_index[env] - 1)
                    self._cur_floor_index[env] -= 1 # 如果下楼的楼梯是误判，那么当前层需要往下减一层了

            elif not self.is_robot_in_stair_map_fast(env, robot_px, stair_map)[0]:
                self._reset_stair_climb_state(env)
                self._climb_stair_over[env] = True
                if climb_direction == 1:
                    self._obstacle_map[env]._up_stair_end = robot_px[0].copy()
                    if not self._obstacle_map_list[env][self._cur_floor_index[env]+1]._done_initializing:
                        self._handle_new_floor_initialization(env, climb_direction)
                    else:
                        self._cur_floor_index[env] += 1
                        self._update_current_maps(env)
                else: # climb_direction == 2
                    self._obstacle_map[env]._down_stair_end = robot_px[0].copy()
                    if not self._obstacle_map_list[env][self._cur_floor_index[env]-1]._done_initializing:
                        self._handle_new_floor_initialization(env, climb_direction)
                    else:
                        self._cur_floor_index[env] -= 1
                        self._update_current_maps(env)
                print("climb stair success!!!!")

    def _reset_stair_climb_state(self, env: int):
        """重置楼梯攀爬相关的状态变量"""
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._stair_dilate_flag[env] = False
        self._climb_stair_flag[env] = 0
        self._obstacle_map[env]._climb_stair_paused_step = 0
        self._last_carrot_xy[env] = []
        self._last_carrot_px[env] = []

    def _disable_stair_and_reset_state(self, env: int, disabled_frontier: np.ndarray, is_reverse: bool = False):
        """
        辅助函数：禁用楼梯并重置相关状态。
        将重复的楼梯禁用和状态重置逻辑封装起来。
        """
        # 确保禁用前沿是可哈希的
        if disabled_frontier.size > 0:
            self._obstacle_map[env]._disabled_frontiers.add(tuple(disabled_frontier))
            print(f"Frontier {disabled_frontier} is disabled due to no movement or reaching start.")
        
        # 重置卡顿相关计数
        self._get_close_to_stair_step[env] = 0
        self._frontier_stick_step[env] = 0
        self._obstacle_map[env]._climb_stair_paused_step = 0
        self._last_carrot_xy[env] = np.array([]) # 清空胡萝卜点
        self._last_carrot_px[env] = np.array([])
        
        # 重置楼梯状态标志
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._stair_dilate_flag[env] = False
        self._climb_stair_over[env] = True # 标记楼梯操作已结束
        self._climb_stair_flag[env] = 0 # 重置爬楼梯标志
        self._obstacle_map[env]._disable_end = False # 重置楼梯终点禁用标记

        # 根据上楼/下楼情况更新地图和楼层信息
        if self._climb_stair_flag[env] == 1: # 上楼被禁用
            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._up_stair_map == 1] = 1
            self._obstacle_map[env]._up_stair_map.fill(0)
            self._obstacle_map[env]._up_stair_frontiers = np.array([]) # 清空前沿
            self._obstacle_map[env]._has_up_stair = False
            self._obstacle_map[env]._look_for_downstair_flag = False
            # 误判上楼或无法上楼，删除多余的地图层
            if not is_reverse: # 如果不是反向返回，即是正常上楼失败
                if self._cur_floor_index[env] + 1 < len(self._object_map_list[env]): # 避免索引越界
                    del self._object_map_list[env][self._cur_floor_index[env] + 1]
                    del self._value_map_list[env][self._cur_floor_index[env] + 1]
                    del self._obstacle_map_list[env][self._cur_floor_index[env] + 1]
                    self.floor_num[env] -= 1 # 楼层数减1
        elif self._climb_stair_flag[env] == 2: # 下楼被禁用
            self._obstacle_map[env]._disabled_stair_map[self._obstacle_map[env]._down_stair_map == 1] = 1
            self._obstacle_map[env]._down_stair_map.fill(0)
            self._obstacle_map[env]._down_stair_frontiers = np.array([]) # 清空前沿
            self._obstacle_map[env]._has_down_stair = False
            self._obstacle_map[env]._look_for_downstair_flag = False
            # 误判下楼或无法下楼，删除多余的地图层并调整当前楼层索引
            if not is_reverse: # 如果不是反向返回，即是正常下楼失败
                if self._cur_floor_index[env] - 1 >= 0: # 避免索引越界
                    del self._object_map_list[env][self._cur_floor_index[env] - 1]
                    del self._value_map_list[env][self._cur_floor_index[env] - 1]
                    del self._obstacle_map_list[env][self._cur_floor_index[env] - 1]
                    self.floor_num[env] -= 1 # 楼层数减1
                    self._cur_floor_index[env] -= 1 # 如果下楼是误判，当前层需要往下减一层

    def _update_stair_state(self, env: int):
        """更新楼梯相关的内部状态。"""
        self._obstacle_map[env]._climb_stair_paused_step = 0
        self._climb_stair_over[env] = True
        self._climb_stair_flag[env] = 0
        self._reach_stair[env] = False
        self._reach_stair_centroid[env] = False
        self._stair_dilate_flag[env] = False

    def _handle_stair_floor_change(self, env: int, direction: int, stair_map_prev_floor, stair_start_prev_floor, stair_end_prev_floor, stair_frontiers_prev_floor, stair_map_cur_floor_attr, stair_start_cur_floor_attr, stair_end_cur_floor_attr, stair_frontiers_cur_floor_attr):
        """处理楼层切换时楼梯地图的更新逻辑。
        direction: 1 for up, -1 for down
        """
        self._done_initializing[env] = False
        self._initialize_step[env] = 0
        
        if direction == 1:
            self._obstacle_map[env]._explored_up_stair = True
            self._cur_floor_index[env] += 1
        elif direction == -1:
            self._obstacle_map[env]._explored_down_stair = True
            self._cur_floor_index[env] -= 1

        # 更新当前楼层的地图
        self._object_map[env] = self._object_map_list[env][self._cur_floor_index[env]]
        self._obstacle_map[env] = self._obstacle_map_list[env][self._cur_floor_index[env]]
        self._value_map[env] = self._value_map_list[env][self._cur_floor_index[env]]

        # 处理楼梯地图的复制和过滤
        ori_stair_map = stair_map_prev_floor.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_stair_map.astype(np.uint8), connectivity=8)
        
        closest_label = -1
        min_distance = float('inf')
        for i in range(1, num_labels):
            centroid_px = centroids[i]
            centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
            distance = np.linalg.norm(self._stair_frontier[env] - centroid)
            if distance < min_distance:
                min_distance = distance
                closest_label = i
        
        if closest_label != -1:
            ori_stair_map[labels != closest_label] = 0

        # 根据方向更新对应的楼梯属性
        setattr(self._obstacle_map_list[env][self._cur_floor_index[env]], stair_map_cur_floor_attr, ori_stair_map)
        setattr(self._obstacle_map_list[env][self._cur_floor_index[env]], stair_start_cur_floor_attr, stair_end_prev_floor.copy())
        setattr(self._obstacle_map_list[env][self._cur_floor_index[env]], stair_end_cur_floor_attr, stair_start_prev_floor.copy())
        setattr(self._obstacle_map_list[env][self._cur_floor_index[env]], stair_frontiers_cur_floor_attr, stair_frontiers_prev_floor.copy())

    def _update_linked_stair_map(self, env: int, original_stair_map: np.ndarray, stair_frontiers: np.ndarray, 
                                target_map_attr: str, start_attr: str, end_attr: str, frontier_attr: str,
                                prev_start_attr: str, prev_end_attr: str, prev_frontier_attr: str):
        """
        更新新楼层中与已爬楼梯对应的楼梯地图信息。
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            original_stair_map.astype(np.uint8), connectivity=8
        )
        
        closest_label = -1
        min_distance = float('inf')
        
        for i in range(1, num_labels):
            centroid_px = centroids[i]
            centroid = self._obstacle_map[env]._px_to_xy(np.atleast_2d(centroid_px))
            distance = np.abs(stair_frontiers[0][0] - centroid[0][0]) + \
                    np.abs(stair_frontiers[0][1] - centroid[0][1])
            if distance < min_distance:
                min_distance = distance
                closest_label = i
        
        if closest_label != -1:
            filtered_stair_map = original_stair_map.copy()
            filtered_stair_map[labels != closest_label] = 0
            
            # 设置新楼层的楼梯地图
            setattr(self._obstacle_map[env], target_map_attr, filtered_stair_map)
            
            # ✅ 修复：直接从传入的楼梯信息获取，而不是从其他楼层
            # 因为 _handle_new_floor_initialization 已经在切换楼层后调用此方法
            # 所以这里需要保存上一个楼层（调用前的当前楼层）的起止点
            
            # 获取上一个楼层的索引
            prev_floor_idx = self._cur_floor_index[env] + 1 if "down" in target_map_attr else self._cur_floor_index[env] - 1
            
            if 0 <= prev_floor_idx < len(self._obstacle_map_list[env]):
                prev_obstacle_map = self._obstacle_map_list[env][prev_floor_idx]
                
                # 起止点互换（上楼时的终点是下楼时的起点）
                setattr(self._obstacle_map[env], start_attr, getattr(prev_obstacle_map, prev_end_attr).copy())
                setattr(self._obstacle_map[env], end_attr, getattr(prev_obstacle_map, prev_start_attr).copy())
                setattr(self._obstacle_map[env], frontier_attr, getattr(prev_obstacle_map, prev_frontier_attr).copy())

    def _update_obstacle_map(self, observations_cache: List[Dict], red_semantic_pred_list: List[np.array], pitch_angle: List[int]) -> None:
        for env in range(self._num_envs):
            robot_xy = observations_cache[env]["robot_xy"]
            robot_px = self._obstacle_map[env]._xy_to_px(np.atleast_2d(robot_xy))
            
            # ✅ 新增：被动检测楼梯（仅在非爬楼梯状态时）
            if self._climb_stair_over[env] and self._climb_stair_flag[env] == 0:
                self._detect_passive_stair_entry(env, robot_px)
            
            # 原有的爬楼梯状态处理
            if not self._climb_stair_over[env]:
                stair_map_to_use = None
                if self._climb_stair_flag[env] == 1:
                    stair_map_to_use = self._obstacle_map[env]._up_stair_map
                elif self._climb_stair_flag[env] == 2:
                    stair_map_to_use = self._obstacle_map[env]._down_stair_map

                if stair_map_to_use is not None:
                    if not self._stair_dilate_flag[env]:
                        self._temp_stair_map[env] = cv2.dilate(
                            stair_map_to_use.astype(np.uint8),
                            (7, 7),
                            iterations=1,
                        )
                        self._stair_dilate_flag[env] = True
                    else:
                        self._temp_stair_map[env] = stair_map_to_use
                    
                    # 处理楼梯状态
                    self._process_stair_climb_state(env, robot_xy, robot_px, self._temp_stair_map[env], self._climb_stair_flag[env])


            self._obstacle_map[env].update_map(
                observations_cache[env]["depth"],
                observations_cache[env]["tf_camera_to_episodic"],
                observations_cache[env]["min_depth"],
                observations_cache[env]["max_depth"],
                observations_cache[env]["fx"],
                observations_cache[env]["fy"],
                observations_cache[env]["camera_fov"],
                self._object_map[env].movable_clouds,
                self._person_masks[env],
                self._stair_masks[env],
                red_semantic_pred_list[env],
                pitch_angle[env],
                self._climb_stair_over[env],
                self._reach_stair[env],
                self._climb_stair_flag[env],
            )
            frontiers = self._obstacle_map[env].frontiers
            self._obstacle_map[env].update_agent_traj(observations_cache[env]["robot_xy"], observations_cache[env]["robot_heading"])
            observations_cache[env]["frontier_sensor"] = frontiers

            # 附加：处理新楼层地图的创建
            if self._obstacle_map[env]._has_up_stair and self._cur_floor_index[env] + 1 >= len(self._object_map_list[env]):
                self._add_floor_map(env, len(self._object_map_list[env]))
            if self._obstacle_map[env]._has_down_stair and self._cur_floor_index[env] == 0:
                self._add_floor_map(env, 0)
                self._cur_floor_index[env] += 1 # 当前不是最底层了
            
            self.floor_num[env] = len(self._obstacle_map_list[env])
            self._obstacle_map[env].project_frontiers_to_rgb_hush(observations_cache[env]["rgb"])
        
    def _update_value_map(self, observations_cache: List[Dict]) -> None:
        for env in range(self._num_envs):         
            cosines = [
                [
                    self._itm.cosine(
                        observations_cache[env]["rgb"],
                        p.replace("target_object", self._target_object[env].replace("|", "/")),
                    )
                    for p in self._text_prompt.split(PROMPT_SEPARATOR)
                ]
            ]
            self._value_map[env].update_map(np.array(cosines[0]), 
                                            observations_cache[env]["depth"], 
                                            observations_cache[env]["tf_camera_to_episodic"],
                                            observations_cache[env]["min_depth"],
                                            observations_cache[env]["max_depth"],
                                            observations_cache[env]["camera_fov"],)

            self._value_map[env].update_agent_traj(
                observations_cache[env]["robot_xy"],
                observations_cache[env]["robot_heading"],
            )
            self._blip_cosine[env] = cosines[0][0]
    def _handle_new_floor_initialization(self, env: int, climb_direction: int):
        """处理新楼层的初始化和地图更新"""
        self._done_initializing[env] = False
        self._initialize_step[env] = 0
        
        if climb_direction == 1: # 上楼
            # 标记当前楼层的上楼梯已探索
            self._obstacle_map[env]._explored_up_stair = True
            # 标记新楼层（上一层）的下楼梯已探索
            self._obstacle_map_list[env][self._cur_floor_index[env]+1]._explored_down_stair = True
            
            # ✅ 修复：使用当前楼层的索引
            ori_up_stair_map = self._obstacle_map[env]._up_stair_map.copy()
            stair_frontiers = self._obstacle_map[env]._up_stair_frontiers
            
            # 切换到新楼层
            self._cur_floor_index[env] += 1
            self._update_current_maps(env)
            
            # 将当前楼层的上楼梯信息保存到新楼层的下楼梯属性
            self._update_linked_stair_map(
                env, ori_up_stair_map, stair_frontiers, 
                target_map_attr="_down_stair_map", 
                start_attr="_down_stair_start", 
                end_attr="_down_stair_end", 
                frontier_attr="_down_stair_frontiers",
                prev_start_attr="_up_stair_start",
                prev_end_attr="_up_stair_end",
                prev_frontier_attr="_up_stair_frontiers"
            )
            
            # 标记新楼层已有下楼梯
            self._obstacle_map[env]._has_down_stair = True

        else: # climb_direction == 2 (下楼)
            # 标记当前楼层的下楼梯已探索
            self._obstacle_map[env]._explored_down_stair = True
            # 标记新楼层（下一层）的上楼梯已探索
            self._obstacle_map_list[env][self._cur_floor_index[env]-1]._explored_up_stair = True

            # ✅ 修复：使用当前楼层的索引
            ori_down_stair_map = self._obstacle_map[env]._down_stair_map.copy()
            stair_frontiers = self._obstacle_map[env]._down_stair_frontiers

            # 切换到新楼层
            self._cur_floor_index[env] -= 1
            self._update_current_maps(env)
            
            # 将当前楼层的下楼梯信息保存到新楼层的上楼梯属性
            self._update_linked_stair_map(
                env, ori_down_stair_map, stair_frontiers, 
                target_map_attr="_up_stair_map", 
                start_attr="_up_stair_start", 
                end_attr="_up_stair_end", 
                frontier_attr="_up_stair_frontiers",
                prev_start_attr="_down_stair_start",
                prev_end_attr="_down_stair_end",
                prev_frontier_attr="_down_stair_frontiers"
            )
            
            # 标记新楼层已有上楼梯
            self._obstacle_map[env]._has_up_stair = True

    def _detect_passive_stair_entry(self, env: int, robot_px: np.ndarray):
        """
        检测机器人是否被动进入楼梯区域。
        如果机器人在楼梯区域内停留超过阈值步数，自动触发爬楼梯模式。
        """
        # 检查上楼梯
        if self._obstacle_map[env]._has_up_stair and len(self._obstacle_map[env]._up_stair_frontiers) > 0:
            in_up_stair, _ = self.is_robot_in_stair_map_fast(
                env, robot_px, self._obstacle_map[env]._up_stair_map
            )
            
            if in_up_stair:
                self._passive_up_stair_steps[env] += 1
                self._passive_down_stair_steps[env] = 0  # 重置下楼梯计数
                
                if self._passive_up_stair_steps[env] >= self.PASSIVE_STAIR_DETECTION_THRESHOLD:
                    # 检查是否已经探索过上层
                    next_floor_idx = self._cur_floor_index[env] + 1
                    if next_floor_idx >= len(self._obstacle_map_list[env]) or \
                       not self._obstacle_map_list[env][next_floor_idx]._this_floor_explored:
                        print(f"Environment {env}: Passive upstairs detection triggered!")
                        self._trigger_stair_climbing(env, climb_direction=1, robot_px=robot_px)
                    self._passive_up_stair_steps[env] = 0
            else:
                self._passive_up_stair_steps[env] = 0
        
        # 检查下楼梯
        if self._obstacle_map[env]._has_down_stair and len(self._obstacle_map[env]._down_stair_frontiers) > 0:
            in_down_stair, _ = self.is_robot_in_stair_map_fast(
                env, robot_px, self._obstacle_map[env]._down_stair_map
            )
            
            if in_down_stair:
                self._passive_down_stair_steps[env] += 1
                self._passive_up_stair_steps[env] = 0  # 重置上楼梯计数
                
                if self._passive_down_stair_steps[env] >= self.PASSIVE_STAIR_DETECTION_THRESHOLD:
                    # 检查是否已经探索过下层
                    prev_floor_idx = self._cur_floor_index[env] - 1
                    if prev_floor_idx < 0 or \
                       not self._obstacle_map_list[env][prev_floor_idx]._this_floor_explored:
                        print(f"Environment {env}: Passive downstairs detection triggered!")
                        self._trigger_stair_climbing(env, climb_direction=2, robot_px=robot_px)
                    self._passive_down_stair_steps[env] = 0
            else:
                self._passive_down_stair_steps[env] = 0
    
    def _trigger_stair_climbing(self, env: int, climb_direction: int, robot_px: np.ndarray):
        """
        触发爬楼梯模式（从被动检测）。
        """
        self._climb_stair_over[env] = False
        self._climb_stair_flag[env] = climb_direction
        self._reach_stair[env] = True
        self._reach_stair_centroid[env] = False  # 被动进入时，可能还未到达质心
        self._get_close_to_stair_step[env] = 0
        
        if climb_direction == 1:
            self._obstacle_map[env]._up_stair_start = robot_px[0].copy()
            self._stair_frontier[env] = self._obstacle_map[env]._up_stair_frontiers
            print(f"Env {env}: Auto-triggered upstairs climbing (passive detection)")
        else:  # climb_direction == 2
            self._obstacle_map[env]._down_stair_start = robot_px[0].copy()
            self._stair_frontier[env] = self._obstacle_map[env]._down_stair_frontiers
            print(f"Env {env}: Auto-triggered downstairs climbing (passive detection)")


    def _get_object_detections_with_stair_and_person(self, img: np.ndarray, non_coco_caption: str, env: int) -> ObjectDetections:
    
        target_classes = self._target_object[env].split("|")
        target_in_coco = any(c in COCO_CLASSES for c in target_classes)
        target_in_non_coco = any(c not in COCO_CLASSES for c in target_classes)
        if len(non_coco_caption) < 1:
            non_coco_caption_with_stair = "stair ."
        else:
            non_coco_caption_with_stair = non_coco_caption +  " stair ."
        # 进行目标检测
        coco_detections = self._coco_object_detector.predict(img)
        non_coco_detections = self._object_detector.predict(img, caption=non_coco_caption_with_stair)
        
        # 过滤目标类别和置信度
        if target_in_coco:
            target_detections = deepcopy(coco_detections)
        elif target_in_non_coco:
            target_detections = deepcopy(non_coco_detections)
        target_detections.filter_by_class(target_classes)  
        det_conf_threshold = self._coco_threshold if target_in_coco else self._non_coco_threshold
        target_detections.filter_by_conf(det_conf_threshold)

        # 如果yolo目标检测结果为空，，尝试Gdino重新检测
        if target_in_coco and target_in_non_coco and target_detections.num_detections == 0:
            target_detections = self._object_detector.predict(img, caption=non_coco_caption)
            target_detections.filter_by_class(target_classes)  
            target_detections.filter_by_conf(self._non_coco_threshold)

        return target_detections, coco_detections, non_coco_detections
    
    def _update_object_map_with_stair_and_person(self, height: int, width: int, observations_cache: List[Dict], non_coco_caption: str, num_steps: List[int], try_to_navigate: List[bool]):
        """
        根据给定的 RGB 和深度图像以及相机到情景坐标系的变换矩阵更新对象地图。
        """
        # 初始化掩码数组
        self._object_masks = np.zeros((self._num_envs, height, width), dtype=np.uint8)
        self._person_masks = np.zeros((self._num_envs, height, width), dtype=bool)
        self._stair_masks = np.zeros((self._num_envs, height, width), dtype=bool)

        for env in range(self._num_envs):
            # 步骤检查，如果不需要处理此环境则跳过，并且如果到了新的楼层，因为一上来就初始化，所以也不需要存
            if self._obstacle_map[env]._floor_num_steps == 0: #  and num_steps[env] == 0
                continue

            # 获取当前环境的观测数据
            rgb = observations_cache[env]["rgb"]
            depth = observations_cache[env]["depth"]
            tf_camera_to_episodic = observations_cache[env]["tf_camera_to_episodic"]
            min_depth = observations_cache[env]["min_depth"]
            max_depth = observations_cache[env]["max_depth"]
            fx = observations_cache[env]["fx"]
            fy = observations_cache[env]["fy"]

            # 获取目标、COCO 和非COCO检测结果
            target_detections, coco_detections, non_coco_detections = \
                self._get_object_detections_with_stair_and_person(rgb, non_coco_caption, env)

            # --- 封装：更新当前步骤的对象和房间信息 ---
            self._update_current_step_scene_info(env, rgb)

            # --- 处理目标对象 ---
            for idx in range(len(target_detections.logits)):
                target_bbox_denorm = target_detections.boxes[idx] * np.array([width, height, width, height])
                target_object_mask = self._mobile_sam.segment_bbox(rgb, target_bbox_denorm.tolist())
                self._object_masks[env][target_object_mask > 0] = 1 # 用于绘制

                self._object_map[env].update_map(
                    self._target_object[env],
                    depth,
                    target_object_mask,
                    tf_camera_to_episodic,
                    min_depth,
                    max_depth,
                    fx,
                    fy,
                )

                # 目标导航双重检查逻辑
                if try_to_navigate[env] and not self._double_check_goal[env]:
                    match_score = self._blip_cosine[env]
                    print(f"Blip2 match score: {match_score}") # 可以改为日志
                    if match_score >= 0.15:
                        self._double_check_goal[env] = True
                        print("Double check success!!!") # 可以改为日志

            ## 当前步很可能有目标，tight search
            # if self._blip_cosine[env] >= 0.15: # 只有没找到楼梯的才tight
            #     self._obstacle_map[env]._tight_search_thresh = True

            # 处理楼梯检测（保持原来的固定阈值）
            non_coco_detections.filter_by_class(["stair"])
            non_coco_detections.filter_by_conf(0.60)  # 专门为楼梯优化的阈值
            
            for idx in range(len(non_coco_detections.logits)):
                stair_bbox_denorm = non_coco_detections.boxes[idx] * np.array([width, height, width, height])
                stair_mask = self._mobile_sam.segment_bbox(rgb, stair_bbox_denorm.tolist())
                self._stair_masks[env][stair_mask > 0] = 1

            # 更新探索地图
            cone_fov = get_fov(fx, depth.shape[1])
            self._object_map[env].update_explored(tf_camera_to_episodic, max_depth, cone_fov)

            # 存储检测结果列表
            self.target_detection_list[env] = target_detections
            self.coco_detection_list[env] = coco_detections
            self.non_coco_detection_list[env] = non_coco_detections

    def _update_current_step_scene_info(self, env: int, cur_rgb: np.ndarray):
        """
        辅助函数：更新当前步骤的对象和房间信息（RAM 和 Place365 模型推断）。
        """
        # 初始化当前步骤的对象和房间列表
        self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps] = []
        self._object_map[env].each_step_rooms[self._obstacle_map[env]._floor_num_steps] = []

        # 对象识别 (RAM)
        cur_objs = self._ram.predict(cur_rgb)
        cur_objs_list = [item.strip() for item in cur_objs.split('|')]
        self._object_map[env].each_step_objects[self._obstacle_map[env]._floor_num_steps] = cur_objs_list
        for obj in cur_objs_list:
            self._object_map[env].this_floor_objects.add(obj)

        # 场景分类/房间识别 (Place365)
        pil_image = Image.fromarray(cur_rgb).convert("RGB")
        if pil_image.mode != "RGB": # 确保图像为 RGB 模式
            pil_image = pil_image.convert("RGB")
        
        place365_input_img = V(self.place365_centre_crop(pil_image).unsqueeze(0)).to(self.device)
        logit = self.scene_classify_model.forward(place365_input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        
        # 获取最可能的房间类别
        probs, idx = h_x.sort(0, True)
        top_5_indices = idx[:5]
        top_5_classes = [self.place365_classes[i] for i in top_5_indices]
        room_candi_type = extract_room_categories(top_5_classes) 

        self._object_map[env].each_step_rooms[self._obstacle_map[env]._floor_num_steps] = room_candi_type
        self._object_map[env].this_floor_rooms.add(room_candi_type)

    # def _update_distance_on_object_map(self, observations_cache) -> None:
    #     self.cur_dis_to_goal = [np.inf] * self._num_envs
    #     for env in range(self._num_envs):
    #         self._object_map[env].update_agent_traj(
    #             observations_cache[env]["robot_xy"],
    #             observations_cache[env]["robot_heading"],
    #         )
    #         if np.argwhere(self._object_map[env]._map).size > 0 and self._target_object[env] in self._object_map[env].clouds:
    #             curr_position = observations_cache[env]["tf_camera_to_episodic"][:3, 3]
    #             closest_point = self._object_map[env]._get_closest_point(self._object_map[env].clouds[self._target_object[env]], curr_position)
    #             self.cur_dis_to_goal[env] = np.linalg.norm(closest_point[:2] - curr_position[:2])

    def _update_distance_on_object_map(self, observations_cache) -> None:
        self.cur_dis_to_goal = [np.inf] * self._num_envs
        for env in range(self._num_envs):
            self._object_map[env].update_agent_traj(
                observations_cache[env]["robot_xy"],
                observations_cache[env]["robot_heading"],
            )
            
            # 修复：先检查目标是否存在
            if self._target_object[env] not in self._object_map[env].clouds:
                continue
                
            # 修复：使用 get_target_cloud 而不是直接用 clouds
            target_cloud = self._object_map[env].get_target_cloud(self._target_object[env])
            
            # 修复：检查过滤后的点云是否为空
            if len(target_cloud) == 0:
                continue
                
            curr_position = observations_cache[env]["tf_camera_to_episodic"][:3, 3]
            closest_point = self._object_map[env]._get_closest_point(target_cloud, curr_position)
            self.cur_dis_to_goal[env] = np.linalg.norm(closest_point[:2] - curr_position[:2])