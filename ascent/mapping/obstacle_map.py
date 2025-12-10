from typing import Any, Union, Dict, Optional, List

import cv2
import numpy as np
# import imagehash
from PIL import Image

from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import get_two_farthest_points, vectorize_get_line_points

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes

from vlfm.vlm.detections import ObjectDetections
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap

from collections import deque

import os
from frontier_exploration.frontier_detection import contour_to_frontiers, interpolate_contour, get_frontier_midpoint, get_closest_frontier_point
import open3d as o3d
import matplotlib.pyplot as plt
from frontier_exploration.utils.general_utils import wrap_heading

STAIR_CLASS_ID = 17  # MPCAT40中 楼梯的类别编号是 16 + 1

def clear_connected_region(map_array, start_y, start_x):
    rows, cols = map_array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右四个方向

    # 初始化队列进行 BFS
    queue = deque([(start_y, start_x)])
    map_array[start_y, start_x] = False  # 将起始点标记为 False

    while queue:
        y, x = queue.popleft()
        
        # 遍历四个方向
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and map_array[ny, nx]:  # 在地图范围内且为 True
                map_array[ny, nx] = False  # 设置为 False
                queue.append((ny, nx))  # 将该点加入队列继续搜索

class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    # _map_dtype: np.dtype = np.dtype(bool)
    # _frontiers_px: np.ndarray = np.array([])
    # frontiers: np.ndarray = np.array([])
    # radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
    ):
        super().__init__(size, pixels_per_meter)

        # initialize class variable
        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)

        self._map_size = size
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._strict_navigable_map = np.zeros((size, size), dtype=bool)
        self._movable_obstacle_map = np.zeros((size, size), dtype=bool)  # mainly for humans
        self._up_stair_map = np.zeros((size, size), dtype=bool)  # for upstairs
        self._down_stair_map = np.zeros((size, size), dtype=bool)  # for downstairs
        self._disabled_stair_map = np.zeros((size, size), dtype=bool)  # for disabled stairs
        # self.newly_assigned_points = 0
        # self._temp_down_stair_map = np.zeros((size, size), dtype=bool)
        self.agent_radius = agent_radius # * 1.3
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * self.agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)

        # try to shrink
        # kernel_size = np.ceil(kernel_size).astype(int) # 
        # kernel_size = 5 # 直接改成5看看是不是非得奇数

        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

        kernel_size += 2
        self._strict_navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        self._has_up_stair = False
        self._has_down_stair = False
        self._done_initializing = False
        self._this_floor_explored = False

        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])

        self._up_stair_start = np.array([])
        # self._up_stair_centroid = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        # self._down_stair_centroid = np.array([])
        self._down_stair_end = np.array([])

        self._carrot_goal_px = np.array([])
        self._explored_up_stair = False
        self._explored_down_stair = False

        self.stair_boundary = np.zeros((size, size), dtype=bool)
        self.stair_boundary_goal = np.zeros((size, size), dtype=bool)
        self._floor_num_steps = 0
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) # np.array([])
        # self._temp_down_stair_map_frontiers_px = np.array([])
        # self._temp_stair_traj = np.array([])
        # self._search_down_stair = False
        self._climb_stair_paused_step = 0
        self._disable_end = False
        # self._look_for_downstair = True
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])
        # 防止楼梯间
        self._reinitialize_flag = False
        self._tight_search_thresh = False
        self._best_frontier_selection_count = {}

        self.previous_frontiers = []  # 存储之前已经可视化过的 frontiers 的索引
        self.frontier_visualization_info = {}  # 存储每个 frontier 对应的 步数
        self._each_step_rgb = {} # 存储每一步对应的rgb, 仅供debug
        # self._each_step_rgb_hash = {} # 存储每一步对应的rgb hash
        self._each_step_rgb_phash = {} # 存储每一步对应的rgb phash
        self._finish_first_explore = False
        self._neighbor_search = False
        self._collision_check_points = self._generate_collision_check_points()
    def _generate_collision_check_points(self):
        """优化碰撞检测点的生成"""
        num_points = 12  # 减少采样点数量
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius_px = int(self.agent_radius * self.pixels_per_meter * 0.8)  # 稍微减小检测半径
        points = []
        
        # 添加中心点周围的几个点
        points.append((0, 0))  # 中心点
        
        # 添加圆周上的点
        for angle in angles:
            x = int(radius_px * np.cos(angle))
            y = int(radius_px * np.sin(angle))
            points.append((x, y))
        
        return points

    def check_path_collision(self, start_px, end_px):
        """检查从起点到终点的路径是否会发生碰撞
        
        Args:
            start_px: 起点像素坐标 (x, y)
            end_px: 终点像素坐标 (x, y)
            
        Returns:
            bool: True表示有碰撞，False表示安全
        """
        # 获取路径上的所有点
        path_length = int(np.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1]))
        if path_length == 0:
            return False
            
        x = np.linspace(start_px[0], end_px[0], path_length)
        y = np.linspace(start_px[1], end_px[1], path_length)
        path_points = np.column_stack((x, y)).astype(int)
        
        # 检查路径上每个点的碰撞
        for point in path_points:
            # 检查机器人圆周上的所有点
            for dx, dy in self._collision_check_points:
                check_x = point[0] + dx
                check_y = point[1] + dy
                
                # 确保检查点在地图范围内
                if (0 <= check_x < self._map.shape[1] and 
                    0 <= check_y < self._map.shape[0]):
                    # 检查是否碰到障碍物
                    if self._map[check_y, check_x]:
                        return True
                else:
                    # 超出地图范围视为碰撞
                    return True
                    
        return False

    def is_safe_path(self, start_px, end_px):
        """检查路径是否安全（考虑碰撞和安全距离）"""
        if self.check_path_collision(start_px, end_px):
            return False
            
        path_length = int(np.hypot(end_px[0] - start_px[0], end_px[1] - start_px[1]))
        if path_length == 0:
            return True
            
        x = np.linspace(start_px[0], end_px[0], path_length)
        y = np.linspace(start_px[1], end_px[1], path_length)
        path_points = np.column_stack((x, y)).astype(int)
        
        for point in path_points:
            if not self.is_safe_navigable(np.array([point])):
                return False
                
        return True

    def reset(self) -> None:
        super().reset()

        # initialize class variable
        self._map_dtype = np.dtype(bool)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self.radius_padding_color = (100, 100, 100)

        
        self._navigable_map.fill(0)
        self._strict_navigable_map.fill(0)
        self._movable_obstacle_map.fill(0) # for movable_obstacle_map

        self._up_stair_map.fill(0) # for upstairs_map
        self._down_stair_map.fill(0) # for downstairs_map
        self._disabled_stair_map.fill(0) # True for not possible for stair
        self.explored_area.fill(0)
        self.stair_boundary.fill(0)
        self.stair_boundary_goal.fill(0)

        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        # self.newly_assigned_points = 0

        self._has_up_stair = False
        self._has_down_stair = False
        self._explored_up_stair = False
        self._explored_down_stair = False
        self._done_initializing = False
        self._up_stair_frontiers_px = np.array([])
        self._up_stair_frontiers = np.array([])
        self._down_stair_frontiers_px = np.array([])
        self._down_stair_frontiers = np.array([])

        self._up_stair_start = np.array([])
        # self._up_stair_centroid = np.array([])
        self._up_stair_end = np.array([])
        self._down_stair_start = np.array([])
        # self._down_stair_centroid = np.array([])
        self._down_stair_end = np.array([])

        self._carrot_goal_px = np.array([])

        self._floor_num_steps = 0      
        self._disabled_frontiers = set()
        self._disabled_frontiers_px =  np.array([], dtype=np.float64).reshape(0, 2) # np.array([])
        # self._search_down_stair = False
        # self._temp_stair_traj = np.array([])
        self._climb_stair_paused_step = 0
        self._disable_end = False
        self._look_for_downstair_flag = False
        self._potential_stair_centroid_px = np.array([])
        self._potential_stair_centroid = np.array([])

        self._reinitialize_flag = False
        self._tight_search_thresh = False
        self._best_frontier_selection_count = {}

        self.previous_frontiers = []  # 存储之前已经可视化过的 frontiers 的索引
        self.frontier_visualization_info = {}  # 存储每个 frontier 对应的 RGB 图以及箭头标记
        self._each_step_rgb = {}
        # self._each_step_rgb_hash = {}
        self._each_step_rgb_phash = {}
        self._finish_first_explore = False
        self._neighbor_search = False
        
    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # Update the navigable area, which is an inverse of the obstacle map after a
            # dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)
            
    def upstair_to_downstair(self, start_y, start_x):
        rows, cols = self._up_stair_map.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右四个方向

        # 初始化队列进行 BFS
        queue = deque([(start_y, start_x)])
        self._up_stair_map[start_y, start_x] = False  # 将起始点标记为 False
        self._down_stair_map[start_y, start_x] = True

        while queue:
            y, x = queue.popleft()
            
            # 遍历四个方向
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and self._up_stair_map[ny, nx]:  # 在地图范围内且为 True
                    self._up_stair_map[ny, nx] = False  # 设置为 False
                    self._down_stair_map[start_y, start_x] = True
                    queue.append((ny, nx))  # 将该点加入队列继续搜索

    def project_frontiers_to_rgb_hush(self, rgb: np.ndarray) -> dict: 
        # , robot_xy: np.ndarray, min_arrow_length: float = 4.0, max_arrow_length: float = 10.0
        """
        Projects the frontiers from the map to the corresponding positions in the RGB image,
        and visualizes them on the RGB image.

        Args:
            rgb (np.ndarray): The RGB image (H x W x 3).
            robot_xy (np.ndarray): The robot's position in the map coordinates (x, y).
            min_arrow_length (float): The minimum length of the arrow in meters. Default is 4.0 meter.
            max_arrow_length (float): The maximum length of the arrow in meters. Default is 10.0 meter.

        Returns:
            dict: A dictionary containing the visualized RGB images with frontiers marked for each new frontier.
        """
        # Step 1: Convert frontiers from pixel coordinates to world coordinates
        if len(self.frontiers) == 0 or self._floor_num_steps == 0:
            return {}  # No frontiers to project

        # Step 2: Identify new frontiers
        new_frontiers = [f for f in self.frontiers if f.tolist() not in self.previous_frontiers]
        if len(new_frontiers) == 0:
            return {}
        
        self.previous_frontiers.extend([f.tolist() for f in new_frontiers])

        # Step 3: Visualize frontiers on the RGB image
        visualized_rgb_ori = rgb.copy()
        self._each_step_rgb[self._floor_num_steps] = visualized_rgb_ori

        for frontier in new_frontiers:

            visualization_info = {
                'floor_num_steps': self._floor_num_steps,
                # 'arrow_end_pixel': arrow_end_pixel
            }
            self.frontier_visualization_info[tuple(frontier)] = visualization_info

    def extract_frontiers_with_image(self, frontier):
        """
        Visualizes frontiers on the RGB images using the stored information in self._each_step_rgb
        and self.frontier_visualization_info. Draws a blue circle with index at the end of a line.
        """
        
        # 获取相关信息
        floor_num_steps = self.frontier_visualization_info[tuple(frontier)]['floor_num_steps']
        visualized_rgb = self._each_step_rgb[floor_num_steps].copy()
        return floor_num_steps, visualized_rgb
    def extract_frontiers_with_image_phash(self, frontier):
        """
        Visualizes frontiers on the RGB images using the stored information in self._each_step_rgb
        and self.frontier_visualization_info. Draws a blue circle with index at the end of a line.
        """
        
        # 获取相关信息
        floor_num_steps = self.frontier_visualization_info[tuple(frontier)]['floor_num_steps']
        visualized_rgb_phash = self._each_step_rgb_phash[floor_num_steps]
        return floor_num_steps, visualized_rgb_phash
                
    def update_map(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        movable_clouds: Dict[str, np.ndarray], 
        # stair_clouds: Dict[str, np.ndarray], 
        person_mask: np.ndarray, # only a (480,640) mask, also for multiple persons
        stair_mask: np.ndarray, # only a (480,640) mask, also for multiple stairs
        seg_mask: np.ndarray,
        agent_pitch_angle: int,
        search_stair_over: bool,
        reach_stair: bool,
        climb_stair_flag: int,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """

        if update_obstacles:
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                         
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth

            # for person
            if "person" in movable_clouds and np.any(person_mask) > 0: # 检测出人
                # 在可移动障碍物地图中标记行人位置
                human_xy_points = movable_clouds["person"][:, :2]  # movable_obstacle_cloud[:, :2]  # 获取行人位置的像素坐标
                human_pixel_points = self._xy_to_px(human_xy_points)
                # 遍历每个 human_pixel_points 点，进行标记和清除
                for x, y in human_pixel_points:
                    # 在 _movable_obstacle_map 上标记为可移动障碍物
                    if 0 <= x < self._movable_obstacle_map.shape[1] and 0 <= y < self._movable_obstacle_map.shape[0]:
                        self._movable_obstacle_map[y, x] = True  # 标记为可移动障碍物

                    # 对每个点调用 BFS，清除相邻的连通障碍区域
                    if 0 <= x < self._map.shape[1] and 0 <= y < self._map.shape[0]:
                        clear_connected_region(self._map, y, x)

            # for stair
            ## upstair or look down to find downstair
            if np.any(stair_mask) > 0 and np.sum(seg_mask == STAIR_CLASS_ID) > 20: # STAIR_CLASS_ID in seg_mask
                stair_map = (seg_mask == STAIR_CLASS_ID)
                fusion_stair_mask = stair_mask & stair_map
                if np.any(fusion_stair_mask) > 0: # 检测出楼梯
                # fusion_stair_mask = stair_mask
                    stair_depth = np.full_like(depth, max_depth)
                    scaled_depth_stair = scaled_depth.copy()
                    stair_depth[fusion_stair_mask] = scaled_depth_stair[fusion_stair_mask]
                    # scaled_depth_stair[fusion_stair_mask] = max_depth
                    # 在楼梯地图中标记楼梯位置
                    # stair_xy_points = stair_clouds["stair"][:, :2]  # easy to false positive
                    stair_cloud_camera_frame = get_point_cloud(stair_depth, fusion_stair_mask, fx, fy)
                    stair_cloud_episodic_frame = transform_points(tf_camera_to_episodic, stair_cloud_camera_frame)
                    stair_xy_points = stair_cloud_episodic_frame[:, :2]
                    stair_pixel_points = self._xy_to_px(stair_xy_points)
                    if agent_pitch_angle >= 0 and climb_stair_flag != 2: # 有可能是reverse_climb_stair
                    # 遍历每个 stair_pixel_points 点，进行标记和清除
                        for x, y in stair_pixel_points:
                            # 在 _stair_map 上标记为确定的楼梯
                            if 0 <= x < self._up_stair_map.shape[1] and 0 <= y < self._up_stair_map.shape[0] and self._up_stair_map[y, x] == 0:
                                self._up_stair_map[y, x] = 1
                        self._map[self._up_stair_map == 1] = 1
                    elif agent_pitch_angle < 0 and climb_stair_flag != 1: # 有可能是reverse_climb_stair
                        for x, y in stair_pixel_points:
                            # 在 _stair_map 上标记为确定的楼梯
                            if 0 <= x < self._down_stair_map.shape[1] and 0 <= y < self._down_stair_map.shape[0] and self._down_stair_map[y, x] == 0:
                                self._down_stair_map[y, x] = 1 
                        self._map[self._down_stair_map == 1] = 1 # 不可通行范围大一点，减少探索

            ## normal to look for downstair
            ## 反转深度，但发现对短楼梯不好使 
            if agent_pitch_angle <= 0 and reach_stair == False: # 靠近楼梯的时候也要找，不然楼梯间的时候下楼误以为上楼了
                filled_depth_for_stair = fill_small_holes(depth, self._hole_area_thresh)
                inverted_depth_for_stair = max_depth - filled_depth_for_stair * (max_depth - min_depth)
                inverted_mask = inverted_depth_for_stair < 2 # inverted_depth_for_stair < 2 # 3 <= true depth value < max_depth 
                inverted_point_cloud_camera_frame = get_point_cloud(inverted_depth_for_stair, inverted_mask, fx, fy)
                inverted_point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, inverted_point_cloud_camera_frame)
                # below_ground_obstacle_cloud = filter_points_by_height_below_ground(inverted_point_cloud_episodic_frame)
                below_ground_obstacle_cloud_0 = filter_points_by_height_below_ground_0(inverted_point_cloud_episodic_frame)
                below_ground_xy_points = below_ground_obstacle_cloud_0[:, :2] # below_ground_obstacle_cloud[:, :2]
                # 获取需要赋值的点的像素坐标
                below_ground_pixel_points = self._xy_to_px(below_ground_xy_points)
                self._down_stair_map[below_ground_pixel_points[:, 1], below_ground_pixel_points[:, 0]] = 1
                
            # 不爬楼梯的时候标注
            if search_stair_over == True: # reach_stair == False:
                mask = scaled_depth < max_depth
                point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
                point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
                obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

                xy_points = obstacle_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points)

                self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1
            
            self._up_stair_map = self._up_stair_map & (~self._disabled_stair_map)
            self._down_stair_map = self._down_stair_map & (~self._disabled_stair_map)
            
            stair_dilated_mask = (self._up_stair_map == 1) | (self._down_stair_map == 1)
            # stair_dilated_mask = self._up_stair_map == 1 | self._down_stair_map == 1 # ((self._map == 1) & (self._up_stair_map == 1)) | ((self._map == 1) & (self._down_stair_map == 1))
            self._map[stair_dilated_mask] = 0

            dilated_map = cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1
            )
            dilated_map[stair_dilated_mask] = 1
            self._map[stair_dilated_mask] = 1
            # 不让楼梯膨胀
            self._navigable_map = 1 - dilated_map.astype(bool)
            
            strict_dilated_map = cv2.dilate(
                self._map.astype(np.uint8),
                self._strict_navigable_kernel,
                iterations=1
            )
            self._strict_navigable_map = 1 - strict_dilated_map.astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        ) # 传入的是一个假设全都是False的值，如果中间有障碍物，最终只会将观察者视野中未被障碍物遮挡的部分标记为 True
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)

        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0

        # # 原来是在这里只保留一个最近的轮廓
        # # 修改为，当膨胀后的障碍物形成一个闭合区域时，探索区域应该取闭合区域的内部部分
        # 成功但没用，不改了
        # dilated_obstacles = cv2.dilate(
        #     self._map.astype(np.uint8),  # 直接使用原始障碍物地图
        #     self._navigable_kernel,
        #     iterations=1  # 保持与导航地图相同的膨胀参数
        # )

        # # 查找可能形成的闭合区域
        # contours, _ = cv2.findContours(
        #     dilated_obstacles,
        #     cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE
        # )
        # bihe = False
        # for contour in contours:
        #     start_point = contour[0][0]  # 轮廓的起始点
        #     end_point = contour[-1][0]  # 轮廓的结束点
        #     if np.array_equal(start_point, end_point):
        #         bihe = True
        # # 仅当检测到闭合区域时进行处理
        # if bihe: # len(contours) > 0:
        #     # 创建包含所有闭合区域的掩膜
        #     room_mask = np.zeros_like(self.explored_area, dtype=np.uint8)
        #     cv2.drawContours(room_mask, contours, -1, 1, -1)  # 填充所有闭合区域
            
        #     # 将已探索区域限制在闭合区域内部
        #     self.explored_area = np.logical_and(
        #         self.explored_area,
        #         room_mask.astype(bool)
        #     )

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers() # _get_frontiers()
        if len(self._frontiers_px) == 0: 
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

        # Compute stair frontier
            
        if np.sum(self._down_stair_map == 1) > 20:
            self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

            # 应该剔除小的，不连通的区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map, connectivity=8)
            min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
            filtered_map = np.zeros_like(self._down_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
                    # 更新最大面积区域的标签
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._down_stair_map = filtered_map
            self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
            self._has_down_stair = True
            self._look_for_downstair_flag = False
            self._potential_stair_centroid_px = np.array([])
            self._potential_stair_centroid = np.array([])
        else:
            # self._down_stair_frontiers_px = np.array([])  # 没有楼梯区域时
            # self._has_down_stair = False
            if np.sum(self._down_stair_map == 1) > 0:
                # self._down_stair_map = cv2.morphologyEx(self._down_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._down_stair_map.astype(np.uint8), connectivity=8)
                max_area = 0
                max_label = 1
                # 逐个找最大区域的质心，直到有向下楼梯
                for i in range(1, num_labels):  # 从1开始，0是背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        max_label = i
                self._potential_stair_centroid_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                self._potential_stair_centroid = self._px_to_xy(self._potential_stair_centroid_px)
                # self._down_stair_map = filtered_map
                # self._down_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])
                # self._down_stair_frontiers = self._px_to_xy(self._down_stair_frontiers_px)
                # self._has_down_stair = True
                # self._look_for_downstair_flag = False

        if np.sum(self._up_stair_map == 1) > 20:
            self._up_stair_map = cv2.morphologyEx(self._up_stair_map.astype(np.uint8) , cv2.MORPH_CLOSE, self._navigable_kernel,) # 一条细线做先膨胀后腐蚀操作

            # 应该剔除小的，不连通的区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._up_stair_map, connectivity=8)
            min_area_threshold = 10  # 设定最小面积阈值为 10（即小于 10 个像素的区域被视为小连通域）
            filtered_map = np.zeros_like(self._up_stair_map)
            max_area = 0
            max_label = 1
            for i in range(1, num_labels):  # 从1开始，0是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area_threshold:
                    filtered_map[labels == i] = 1  # 保留面积大于阈值的区域
                    # 更新最大面积区域的标签
                    if area > max_area:
                        max_area = area
                        max_label = i

            self._up_stair_map = filtered_map
            self._up_stair_frontiers_px = np.array([[centroids[max_label][0], centroids[max_label][1]]])

            self._up_stair_frontiers = self._px_to_xy(self._up_stair_frontiers_px)
            self._has_up_stair = True
        else:
            self._up_stair_frontiers_px = np.array([])  # 没有楼梯区域时
            self._has_up_stair = False

        if len(self._down_stair_frontiers) == 0 and np.sum(self._down_stair_map) > 0:
            # 标识，提示agent往这边导航
            self._look_for_downstair_flag = True 
    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        # 如果有楼梯间，那么需要更仔细地搜索到另一个楼梯
        # 或者探索完没发现楼梯
        if self._tight_search_thresh:
            frontiers = detect_frontier_waypoints(
                self._navigable_map.astype(np.uint8),
                explored_area,
                -1,
            )
        else:
            frontiers = detect_frontier_waypoints(
                self._navigable_map.astype(np.uint8),
                explored_area,
                self._area_thresh_in_pixels,
            )
        return frontiers

    def visualize(self) -> np.ndarray:
        # 影响画图
        temp_disabled_frontiers = np.atleast_2d(np.array(list(self._disabled_frontiers)))
        if len(temp_disabled_frontiers[0]) > 0:
            self._disabled_frontiers_px = self._xy_to_px(temp_disabled_frontiers)
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw movable obstacles (humans) in pink
        vis_img[self._movable_obstacle_map == 1] = (255,0,255)
        # Draw detected upstair area in purple
        vis_img[self._up_stair_map == 1] = (128,0,128)
        # Draw detected downstair area in orange
        vis_img[self._down_stair_map == 1] = (139, 26, 26)
        
        for carrot in self._carrot_goal_px:
            cv2.circle(vis_img, tuple([int(i) for i in carrot]), 5, (42, 42, 165), 2) # 红色空心圆
        # vis_img[self._temp_down_stair_map == 1] = (139, 69, 19)
        # Draw frontiers in blue (200, 0, 0), 似乎是bgr，不是rgb
        if len(self._down_stair_end) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._down_stair_end]), 5, (0, 255, 255), 2) # 黄色空心圆
        if len(self._up_stair_end) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._up_stair_end]), 5, (0, 255, 255), 2) # 黄色空心圆
        if len(self._down_stair_start) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._down_stair_start]), 5, (101, 96, 127), 2) # 粉色空心圆
        if len(self._up_stair_start) > 0:
            cv2.circle(vis_img, tuple([int(i) for i in self._up_stair_start]), 5, (101, 96, 127), 2) # 粉色空心圆
        for frontier in self._frontiers_px:
            temp = np.array([int(i) for i in frontier])
            if temp not in self._disabled_frontiers_px:
                cv2.circle(vis_img, tuple(temp), 5, (200, 0, 0), 2) # 蓝色空心圆
        # Draw stair frontiers in orange (100, 0, 0)
        for up_stair_frontier in self._up_stair_frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in up_stair_frontier]), 5, (255, 128, 0), 2) # 淡蓝色空心圆
        for down_stair_frontier in self._down_stair_frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in down_stair_frontier]), 5, (19, 69, 139), 2) # 暗橙色空心圆
                    
        for potential_downstair in self._potential_stair_centroid_px:
            cv2.circle(vis_img, tuple([int(i) for i in potential_downstair]), 5, (128, 69, 128), 2) # 紫色空心圆
        # for traj in self._temp_stair_traj:
        #     cv2.circle(vis_img, tuple([int(i) for i in traj]), 5, (128, 69, 128), 2) # 紫色空心圆

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img

# for debug and stair, make a new one 
def reveal_fog_of_war(
    top_down_map: np.ndarray,
    current_fog_of_war_mask: np.ndarray,
    current_point: np.ndarray,
    current_angle: float,
    fov: float = 90,
    max_line_len: float = 100,
    enable_debug_visualization: bool = False,
) -> np.ndarray:
    curr_pt_cv2 = current_point[::-1].astype(int)
    angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))

    cone_mask = cv2.ellipse(
        np.zeros_like(top_down_map),
        curr_pt_cv2,
        (int(max_line_len), int(max_line_len)),
        0,
        angle_cv2 - fov / 2,
        angle_cv2 + fov / 2,
        1,
        -1,
    )

    # Create a mask of pixels that are both in the cone and NOT in the top_down_map, actually the obstacle map
    obstacles_in_cone = cv2.bitwise_and(cone_mask, 1 - top_down_map)

    # Find the contours of the obstacles in the cone
    obstacle_contours, _ = cv2.findContours(
        obstacles_in_cone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if enable_debug_visualization:
        vis_top_down_map = top_down_map * 255
        vis_top_down_map = cv2.cvtColor(vis_top_down_map, cv2.COLOR_GRAY2BGR)
        vis_top_down_map[top_down_map > 0] = (60, 60, 60)
        vis_top_down_map[top_down_map == 0] = (255, 255, 255)
        cv2.circle(vis_top_down_map, tuple(curr_pt_cv2), 3, (255, 192, 15), -1)
        cv2.imshow("vis_top_down_map", vis_top_down_map)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_top_down_map")

        cone_minus_obstacles = cv2.bitwise_and(cone_mask, top_down_map)
        vis_cone_minus_obstacles = vis_top_down_map.copy()
        vis_cone_minus_obstacles[cone_minus_obstacles == 1] = (127, 127, 127)
        cv2.imshow("vis_cone_minus_obstacles", vis_cone_minus_obstacles)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_cone_minus_obstacles")

        vis_obstacles_mask = vis_cone_minus_obstacles.copy()
        cv2.drawContours(vis_obstacles_mask, obstacle_contours, -1, (0, 0, 255), 1)
        cv2.imshow("vis_obstacles_mask", vis_obstacles_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_obstacles_mask")

    if len(obstacle_contours) == 0:
        return current_fog_of_war_mask  # there were no obstacles in the cone

    # Find the two points in each contour that form the smallest and largest angles
    # from the current position
    points = []
    for cnt in obstacle_contours:
        if cv2.isContourConvex(cnt):
            pt1, pt2 = get_two_farthest_points(curr_pt_cv2, cnt, angle_cv2)
            points.append(pt1.reshape(-1, 2))
            points.append(pt2.reshape(-1, 2))
        else:
            # Just add every point in the contour
            points.append(cnt.reshape(-1, 2))
    points = np.concatenate(points, axis=0)

    # Fragment the cone using obstacles and two lines per obstacle in the cone
    visible_cone_mask = cv2.bitwise_and(cone_mask, top_down_map)
    line_points = vectorize_get_line_points(curr_pt_cv2, points, max_line_len * 1.05)
    # Draw all lines simultaneously using cv2.polylines
    cv2.polylines(visible_cone_mask, line_points, isClosed=False, color=0, thickness=2)

    # Identify the contour that is closest to the current position
    final_contours, _ = cv2.findContours(
        visible_cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    visible_area = None
    min_dist = np.inf
    for cnt in final_contours:
        pt = tuple([int(i) for i in curr_pt_cv2])
        dist = abs(cv2.pointPolygonTest(cnt, pt, True))
        if dist < min_dist:
            min_dist = dist
            visible_area = cnt

    if enable_debug_visualization:
        vis_points_mask = vis_obstacles_mask.copy()
        for point in points.reshape(-1, 2):
            cv2.circle(vis_points_mask, tuple(point), 3, (0, 255, 0), -1)
        cv2.imshow("vis_points_mask", vis_points_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_points_mask")

        vis_lines_mask = vis_points_mask.copy()
        cv2.polylines(
            vis_lines_mask, line_points, isClosed=False, color=(0, 0, 255), thickness=2
        )
        cv2.imshow("vis_lines_mask", vis_lines_mask)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_lines_mask")

        vis_final_contours = vis_top_down_map.copy()
        # Draw each contour in a random color
        for cnt in final_contours:
            color = tuple([int(i) for i in np.random.randint(0, 255, 3)])
            cv2.drawContours(vis_final_contours, [cnt], -1, color, -1)
        cv2.imshow("vis_final_contours", vis_final_contours)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_final_contours")

        vis_final = vis_top_down_map.copy()
        # Draw each contour in a random color
        cv2.drawContours(vis_final, [visible_area], -1, (127, 127, 127), -1)
        cv2.imshow("vis_final", vis_final)
        cv2.waitKey(0)
        cv2.destroyWindow("vis_final")

    if min_dist > 3:
        return current_fog_of_war_mask  # the closest contour was too far away

    new_fog = cv2.drawContours(current_fog_of_war_mask, [visible_area], 0, 1, -1)

    return new_fog

def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]

def filter_points_by_height_below_ground_0(points: np.ndarray) -> np.ndarray:
    data = points[(points[:, 2] < 0)] # 0.2 是机器人的max_climb
    return data