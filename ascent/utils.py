import pandas as pd
import numpy as np
import torch
from constants import (
    DIRECT_MAPPING,
    REFERENCE_ROOMS,
)
from typing import Any, List, Tuple, Dict
from habitat.tasks.nav.nav import Success, TopDownMap, HeadingSensor, NavigationEpisode
from habitat.core.simulator import AgentState
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from RedNet.RedNet_model import load_rednet
import torchvision.models as models
from dataclasses import dataclass
from omegaconf import OmegaConf
from omegaconf import DictConfig
from habitat import EmbodiedTask
from habitat.core.registry import registry
from vlfm.policy.habitat_policies import VLFMPolicyConfig, cs
from frontier_exploration.measurements import FrontierExplorationMap, FrontierExplorationMapMeasurementConfig
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.utils.visualizations import fog_of_war, maps
from frontier_exploration.utils.general_utils import habitat_to_xyz
from collections import Counter
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.utils.visualizations.utils import images_to_video
cv2 = try_cv2_import()
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    scene_id: List[str],
    goal_name: str,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
    verbose: bool = True,
    keys_to_include_in_name: Optional[List[str]] = None,
    failure_cause: str = "",
) -> str:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "SPL".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        The saved video name.
    """
    if len(images) < 1:
        return ""
    shape0, shape1 = images[0].shape, images[1].shape
    if any(s1 > s0 for s0, s1 in zip(shape0, shape1)):
        padding_0 = [(0, max(s1 - s0, 0)) for s0, s1 in zip(shape0, shape1)]
        images[0] = np.pad(images[0], pad_width=padding_0, mode='constant', constant_values=0)
    # 仅在 images[0] 大于 images[1] 时裁剪
    if any(s0 > s1 for s0, s1 in zip(shape0, shape1)):
        crop_0 = [slice(0, s1) for s1 in shape1]
        images[0] = images[0][tuple(crop_0)]

    metric_strs = []
    if (
        keys_to_include_in_name is not None
        and len(keys_to_include_in_name) > 0
    ):
        use_metrics_k = [
            k
            for k in metrics
            if any(
                to_include_k in k for to_include_k in keys_to_include_in_name
            )
        ]
    else:
        use_metrics_k = list(metrics.keys())

    for k in use_metrics_k:
        metric_strs.append(f"{k}={metrics[k]:.2f}")

    # video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
    #     metric_strs
    # )
    # my preferred name
    video_name = f"scene={scene_id}-episode={episode_id}-" + f"goal={goal_name}-" + "-".join(
        metric_strs
    ) + "-"+ failure_cause
    if "disk" in video_option:
        assert video_dir is not None
        try:
            images_to_video(
                images, video_dir, video_name, fps=fps, verbose=verbose
            )
        except ValueError as e:
            print(f"Error in images_to_video: {e}")
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )
    return video_name
def xyz_yaw_pitch_roll_to_tf_matrix(xyz: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Converts a given position and yaw, pitch, roll angles to a 4x4 transformation matrix.

    Args:
        xyz (np.ndarray): A 3D vector representing the position.
        yaw (float): The yaw angle in radians (rotation around Z-axis).
        pitch (float): The pitch angle in radians (rotation around Y-axis).
        roll (float): The roll angle in radians (rotation around X-axis).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    x, y, z = xyz
    
    # Rotation matrices for yaw, pitch, roll
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    
    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll
    
    # Construct 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R  # Rotation
    transformation_matrix[:3, 3] = [x, y, z]  # Translation

    return transformation_matrix

def check_stairs_in_upper_50_percent(mask):
    """
    检查在图像的上方30%区域是否有STAIR_CLASS_ID的标记
    参数：
    - mask: 布尔值数组，表示各像素是否属于STARR_CLASS_ID
    
    返回：
    - 如果上方30%区域有True，则返回True，否则返回False
    """
    # 获取图像的高度
    height = mask.shape[0]
    
    # 计算上方50%的区域的高度范围
    upper_50_height = int(height * 0.5)
    
    # 获取上方50%的区域的掩码
    upper_50_mask = mask[:upper_50_height, :]
    
    print(f"Stair upper 50% points: {np.sum(upper_50_mask)}")
    # 检查该区域内是否有True
    if np.sum(upper_50_mask) > 50:  # 如果上方50%区域内有True
        return True
    return False

def load_floor_probabilities(file_path):
    """
    加载楼层和物体分布概率表格。
    """
    df = pd.read_excel(file_path)
    return df
def load_place365_categories(file_path: str) -> tuple:
    """加载 Place365 类别名称。"""
    classes = []
    with open(file_path) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    return tuple(classes)
def load_floor_probabilities_by_dataset(dataset_type: str):
    """根据数据集类型加载楼层概率数据。"""
    if dataset_type == "hm3d":
        return load_floor_probabilities("statistic_priors/hm3d_floor_object_possibility.xlsx")
    elif dataset_type == "mp3d":
        return load_floor_probabilities("statistic_priors/mp3d_floor_object_possibility.xlsx")
    else:
        raise ValueError(f"Dataset type {dataset_type} not recognized")    

def extract_room_categories(top_5_classes):
    """
    提取目标类别或选择 top 1 类别。
    
    参数:
        top_5_classes (list of str): top 5 的类别名称列表。
    
    返回:
        list: 如果 top 5 中有direct_mapping能映射的目标类别，返回排名靠前的目标类别；否则返回 top 1 类别。
    """
    # 遍历 top_5_classes，尝试映射到目标类别
    selected_categories = ""
    for cls_name in top_5_classes:
        if cls_name in DIRECT_MAPPING:  # 如果类别在 direct_mapping 中
            mapped_category = DIRECT_MAPPING[cls_name]
            if mapped_category in REFERENCE_ROOMS:  # 如果映射后的类别是目标类别
                selected_categories = mapped_category
                break  # 只选择第一个匹配的目标类别

    # 如果没有匹配的目标类别，返回 top 1 类别
    return selected_categories if selected_categories else top_5_classes[0]

def get_action_tensor(action_id, device="cuda"):
    return torch.tensor([[action_id]], dtype=torch.long, device=device)

def load_rednet_model(model_path, device):
    """加载 RedNet 模型。"""
    red_sem_pred = load_rednet(
        device, ckpt=model_path, resize=True,
    )
    red_sem_pred.eval()
    return red_sem_pred

def load_place365_model(arch, device):
    """加载 Place365 场景分类模型。"""
    model_file = 'pretrained_weights/%s_places365.pth.tar' % arch
    scene_classify_model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage, weights_only=False)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    
    load_result = scene_classify_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Place365 Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Place365 Unexpected keys: {load_result.unexpected_keys}")

    scene_classify_model = scene_classify_model.to(device)
    scene_classify_model.eval()
    return scene_classify_model

## Additional Config

@dataclass
class AscentPolicyConfig(VLFMPolicyConfig):
    name: str = "AscentPolicy"
    nearby_distance: float = 3.0
    topk: int = 3


@registry.register_measure
class MultiFloorTopDownMap(FrontierExplorationMap):
    def __init__(
        self,
        sim: HabitatSim,
        config: DictConfig,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config, task, *args, **kwargs)
        self._floor_heights = None
        self._saved_maps = None
        self._saved_fogs = None
        self._cur_floor = 0

    def reset_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ) -> None:
        self._floor_heights, self._saved_maps, self._saved_fogs = self.detect_floors(sample_points=100)
        agent_position = self._sim.get_agent_state().position
        agent_height = agent_position[1]
        # Find the floor corresponding to the agent's height
        flag = True  # 标记是否找到匹配楼层
        for ithfloor, floor_height in enumerate(self._floor_heights):
            if self._is_on_same_floor(agent_height, floor_height):
                self._top_down_map = self._saved_maps[ithfloor]
                self._fog_of_war_mask = self._saved_fogs[ithfloor]
                self._cur_floor = ithfloor
                flag = False  # 找到匹配楼层，标记为 False
                break
        if flag:  # 未找到匹配楼层，选择最近的楼层
            closest_floor = min(
                range(len(self._floor_heights)),
                key=lambda idx: abs(self._floor_heights[idx] - agent_height),
            )
            self._top_down_map = self._saved_maps[closest_floor]
            self._fog_of_war_mask = self._saved_fogs[closest_floor]
            self._cur_floor = closest_floor
            
        if self._top_down_map is None:
            self._top_down_map = self._saved_maps[0] # if update, use the previous one

        self._previous_xy_location = [
            None for _ in range(len(self._sim.habitat_config.agents))
        ]

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source and self._is_on_same_floor(agent_height, episode.start_position[1]):
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

        ### For frontier exploration
        assert "task" in kwargs, "task must be passed to reset_metric!"
        self._explorer_sensor = kwargs["task"].sensor_suite.sensors[self._explorer_uuid]
        self._static_metrics = {}

        self.update_metric(episode, None) # need _explorer_sensor to provide fog_mask # 

        self._draw_target_bbox_mask(episode)

        # Expose sufficient info for drawing 3D points on the map
        lower_bound, upper_bound = self._sim.pathfinder.get_bounds()
        episodic_start_yaw = HeadingSensor._quat_to_xy_heading(
            None,  # type: ignore
            quaternion_from_coeff(episode.start_rotation).inverse(),
        )[0]
        x, y, z = habitat_to_xyz(np.array(episode.start_position))
        self._static_metrics["upper_bound"] = (upper_bound[0], upper_bound[2])
        self._static_metrics["lower_bound"] = (lower_bound[0], lower_bound[2])
        self._static_metrics["grid_resolution"] = self._metric["map"].shape[:2]
        self._static_metrics["tf_episodic_to_global"] = np.array(
            [
                [np.cos(episodic_start_yaw), -np.sin(episodic_start_yaw), 0, x],
                [np.sin(episodic_start_yaw), np.cos(episodic_start_yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=0.5, floor_tolerance = 0.2, # ceiling_height=2.0, floor_tolerance = 0.02
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height - floor_tolerance <= height < ref_floor_height + ceiling_height

    def detect_floors(self, output_dir="./", sample_points=100, floor_tolerance=0.5):
        """
        Detect the number of floors and their heights in a scene.

        :param pathfinder: Habitat-sim's PathFinder instance.
        :param sample_points: Number of points to sample per island to calculate its height.
        :param floor_tolerance: Minimum height difference to distinguish different floors.

        :return: List of detected floor heights (sorted) and the total number of floors.
        """
        island_heights = []

        # Iterate through all navigable islands
        for island_index in range(self._sim.pathfinder.num_islands):
            heights = []
            
            # Randomly sample points from the island
            for _ in range(sample_points):
                point = self._sim.pathfinder.get_random_navigable_point(island_index=island_index)
                if point is not None:
                    heights.append(point[1])  # Store the height (y-coordinate)

            # 计算该岛屿的高度
            if heights:
                height_counts = Counter(heights)  # 统计每个高度的出现次数
                
                for height, count in height_counts.items():
                    # 如果某高度的出现次数超过三分之一，记录该高度
                    if count > sample_points // 4:
                        island_heights.append(height)

        # Sort the heights and group them by floor tolerance
        island_heights = sorted(island_heights, reverse=False)
        
        floor_heights = []
        for height in island_heights:
            if not floor_heights or abs(height - floor_heights[-1]) >= floor_tolerance:
                floor_heights.append(height)

        # floor_heights=island_heights

        # Generate and save top-down maps
        saved_maps = []
        saved_fogs = []        
        for idx, height in enumerate(floor_heights):
            # try:
                # Generate top-down view
                # topdown_map = self._sim.pathfinder.get_topdown_view(
                #     meters_per_pixel=1.0 / self._map_resolution, height=height
                # )
            topdown_map = maps.get_topdown_map(
            pathfinder=self._sim.pathfinder,
            height=height,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )
            saved_maps.append(topdown_map)
            if self._config.fog_of_war.draw:
                saved_fogs.append(np.zeros_like(topdown_map)) # 
            else:
                saved_fogs.append(None)

        return floor_heights, saved_maps, saved_fogs
    
    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        if (a_x < self._top_down_map.shape[0] and a_x >= 0) and (a_y < self._top_down_map.shape[1] and a_y >= 0):
            pass
        else:
            return a_x, a_y
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = maps.MAP_SOURCE_POINT_INDICATOR + agent_index * 10 
            # color = 10 + min(
            #     self._step_count * 245 // self._config.max_episode_steps, 245
            # )
            thickness = self.line_thickness
            if self._previous_xy_location[agent_index] is not None:
                cv2.line(
                    self._top_down_map,
                    self._previous_xy_location[agent_index],
                    (a_y, a_x),
                    color,
                    thickness=thickness,
                )
        angle = TopDownMap.get_polar_angle(agent_state)
        if self._fog_of_war_mask.shape == self._explorer_sensor.fog_of_war_mask.shape: # the 0th step can cause error
            self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        agent_height = agent_position[1]
        if self._is_on_same_floor(agent_height, self._floor_heights[self._cur_floor]):
            pass
        else:
            flag = True # do not match any floor
            for ithfloor, floor_height in enumerate(self._floor_heights):
                if self._is_on_same_floor(agent_height, floor_height):
                    self._top_down_map = self._saved_maps[ithfloor]
                    self._fog_of_war_mask = self._saved_fogs[ithfloor]
                    self._cur_floor = ithfloor
                    flag = False
                    break
            if flag: # maybe at stair
                pass
            else:
                if hasattr(episode, "goals"):
                # draw source and target parts last to avoid overlap
                    self._draw_goals_view_points(episode)
                    self._draw_goals_aabb(episode)
                    self._draw_goals_positions(episode)
                    self._draw_shortest_path(episode, agent_position)
                if self._config.draw_source and self._is_on_same_floor(episode.start_position[1], self._floor_heights[self._cur_floor]):
                    self._draw_point(
                        episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
                    )
        map_positions: List[Tuple[float]] = []
        map_angles = []
        if 'human_num' in episode.info:
            for agent_index in range(episode.info['human_num']):
                agent_state = self._sim.get_agent_state(agent_index)
                if self._is_on_same_floor(agent_state.position[1], self._floor_heights[self._cur_floor]) : # for human, filter the one in other floor
                    map_positions.append(self.update_map(agent_state, agent_index))
                    map_angles.append(MultiFloorTopDownMap.get_polar_angle(agent_state))
        else:
            for agent_index in range(len(self._sim.habitat_config.agents)):
                agent_state = self._sim.get_agent_state(agent_index)
                if self._is_on_same_floor(agent_state.position[1], self._floor_heights[self._cur_floor]) : # for human, filter the one in other floor
                    map_positions.append(self.update_map(agent_state, agent_index))
                    map_angles.append(MultiFloorTopDownMap.get_polar_angle(agent_state))
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_positions,
            "agent_angle": map_angles,
        }

        ### For frontier exploration
        # Update the map with visualizations of the frontier waypoints
        new_map = self._metric["map"].copy()
        circle_size = 20 * self._map_resolution // 1024
        thickness = max(int(round(3 * self._map_resolution / 1024)), 1)
        selected_frontier = self._explorer_sensor.closest_frontier_waypoint

        if self._draw_waypoints:
            next_waypoint = self._explorer_sensor.next_waypoint_pixels
            if next_waypoint is not None:
                cv2.circle(
                    new_map,
                    tuple(next_waypoint[::-1].astype(np.int32)),
                    circle_size,
                    maps.MAP_INVALID_POINT,
                    1,
                )

        for waypoint in self._explorer_sensor.frontier_waypoints:
            if np.array_equal(waypoint, selected_frontier):
                color = maps.MAP_TARGET_POINT_INDICATOR
            else:
                color = maps.MAP_SOURCE_POINT_INDICATOR
            cv2.circle(
                new_map,
                waypoint[::-1].astype(np.int32),
                circle_size,
                color,
                1,
            )

        beeline_target = getattr(self._explorer_sensor, "beeline_target_pixels", None)
        if beeline_target is not None:
            cv2.circle(
                new_map,
                tuple(beeline_target[::-1].astype(np.int32)),
                circle_size * 2,
                maps.MAP_SOURCE_POINT_INDICATOR,
                thickness,
            )
        self._metric["map"] = new_map
        self._metric["is_feasible"] = self._is_feasible
        # if not self._is_feasible:
        #     self._task._is_episode_active = False

        # Update self._metric with the static metrics
        self._metric.update(self._static_metrics)

@dataclass
class MultiFloorTopDownMapMeasurementConfig(FrontierExplorationMapMeasurementConfig):
    type: str = "MultiFloorTopDownMap"

cs.store(
    group="habitat_baselines/rl/policy",
    name="ascent_policy",
    node={
        "main_agent": AscentPolicyConfig  # 注意：传类，不是实例
    }
)

# cs.store(group="habitat_baselines/rl/policy", name="ascent_policy", node=AscentPolicyConfig)

# main_agent_ascent_policy = {
#     'main_agent': OmegaConf.structured(AscentPolicyConfig())
# }

# cs.store(group="habitat_baselines/rl/policy", name="main_agent_ascent_policy", node=main_agent_ascent_policy)

cs.store(
    package="habitat.task.measurements.multi_floor_map",
    group="habitat/task/measurements",
    name="multi_floor_map",
    node=MultiFloorTopDownMapMeasurementConfig,
)