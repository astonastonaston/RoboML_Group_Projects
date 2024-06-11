"""
At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto run by ManiSkill.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents, in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
import sapien
from transforms3d.euler import euler2quat
from mani_skill.envs.utils import randomization

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

import matplotlib.pyplot as plt
import gymnasium as gym

@register_env("myenv-v1", max_episode_steps=50)
class PutCubeIntoBinEnv(BaseEnv):
    """
    Task Description
    ----------------
    

    Randomizations
    --------------
    
    Success Conditions
    ------------------
    
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    cube_half_size = 0.02
    side_half_size = cube_half_size/8
    block_half_size = [side_half_size, 2*side_half_size+cube_half_size, 2*side_half_size+cube_half_size] # the block of the bin
    edge_block_half_size = [side_half_size, 2*side_half_size+cube_half_size, 2*side_half_size] # the block of the bin
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
        
    def _build_bin(self, cube_half_size):
        builder = self.scene.create_actor_builder()
        
        # init the basic block: with x-axis length small and other-axis length larger
        dx = self.block_half_size[1] - self.block_half_size[0] 
        dy = self.block_half_size[1] - self.block_half_size[0] 
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

	      # build bin blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

      	# build the kinematic bin which is not collidable
        return builder.build_kinematic(name="bin")

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )
        # self.obj = actors.build_sphere(
        #     self.scene,
        #     radius=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="sphere",
        #     body_type="dynamic",
        # )

        self.goal_site = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        
        # self.goal_site = actors.build_sphere(
        #     self.scene,
        #     radius=self.cube_half_size,
        #     color=[0, 1, 0, 1],
        #     name="goal_site",
        #     body_type="kinematic",
        #     add_collision=False,
        # )
        self.bin = self._build_bin(self.cube_half_size)



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # init the cube in the first 1/4 zone (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            # print(f"xyz.shape is {xyz.shape, xyz[..., 0].shape, (torch.rand((b, 1)) * 0.1 - 0.2).shape}")
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0] # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[..., 0] # spanning all possible ys
            xyz[..., 2] = self.cube_half_size # on the table
            qs_obj = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            # initialize all equivalent poses
            obj_pose = Pose.create_from_pq(p=xyz, q=qs_obj)
            self.obj.set_pose(obj_pose)

            # init the bin in the last 1/2 zone (so that it doesn't collide the cube)
            pos = torch.zeros((b, 3))
            qs_bin = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            pos[:, 0] = torch.rand((b, 1))[..., 0] * 0.1 # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1 # spanning all possible ys
            pos[:, 2] = self.block_half_size[0] # on the table
            bin_pose = Pose.create_from_pq(p=pos, q=qs_bin)
            self.bin.set_pose(bin_pose)
            
            # init the goal position
            goal_xyz = pos.clone()
            goal_xyz[:, 2] = goal_xyz[:, 2] + self.block_half_size[0] + self.cube_half_size
            goal_pose = Pose.create_from_pq(p=goal_xyz, q=qs_bin)
            self.goal_site.set_pose(goal_pose)


    def evaluate(self):

        p_diff = self.goal_site.pose.p - self.obj.pose.p
        q_diff = self.goal_site.pose.q - self.obj.pose.q
        is_obj_placed = (torch.linalg.norm(p_diff[..., :2], axis=1) < 0.01)
        is_obj_placed = is_obj_placed.logical_and(torch.linalg.norm(q_diff, axis=1) < 0.01)

        is_robot_static = self.agent.is_static(0.2)
        is_grasped = self.agent.is_grasping(self.obj)
        
        
        return {
            "success": is_obj_placed & is_robot_static & ~is_grasped,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped
        }

    def _get_obs_extra(self, info: Dict):
        # tcp: (tool center point) which is the point between the grippers of the robot
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            bin_pos=self.bin.pose.p
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        cube_to_tcp_dist = torch.linalg.norm(self.agent.tcp.pose.p - self.obj.pose.p, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cube_to_tcp_dist))

        # # grasp and reach top reward 
        # bin_top_pos_p = self.goal_site.pose.p.clone()
        # bin_top_pos_p[..., 2] = bin_top_pos_p[..., 2] + self.block_half_size[1]*3
        # cube_to_bin_top_dist = torch.linalg.norm(bin_top_pos_p - self.obj.pose.p, axis=1)  
        # place_reward = 1 - torch.tanh(5.0 * cube_to_bin_top_dist)

        # reward[info["is_grasped"]] = (4 + place_reward)[info["is_grasped"]]

        # # align pose reward
        # is_on_top = (cube_to_bin_top_dist < 0.03)
        # bin_top_pos_q = self.goal_site.pose.q
        # q_dist = torch.linalg.norm(bin_top_pos_q - self.obj.pose.q, axis=1) # TODO: do min-of-N loss for q
        # align_reward = 1 - torch.tanh(5.0 * q_dist)
        # reward[is_on_top] = (6 + align_reward)[is_on_top]
        
        # grasp and reach reward 
        p_diff = self.goal_site.pose.p - self.obj.pose.p
        q_diff = self.goal_site.pose.q - self.obj.pose.q
        cube_to_bin_top_pq_dist = torch.linalg.norm(p_diff, axis=1)
        cube_to_bin_top_pq_dist += torch.linalg.norm(q_diff, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cube_to_bin_top_pq_dist)

        reward[info["is_grasped"]] = (4 + place_reward)[info["is_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~info["is_grasped"]] = 1.0
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward[info["is_obj_placed"]] = (
            8 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_obj_placed"]]
        
        # success reward
        reward[info["success"]] = 10

        return reward

    def __compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # reaching object reward
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        # print(f"reaching rw {reaching_reward}")
        reward = reaching_reward

        # grasping reward
        is_grasped = info["is_grasped"]
        # print(f"grasping rw {is_grasped}")
        reward += is_grasped

        # reaching the bin top reward (use the xy distance / angle approx pi/2 rewards)
        obj_to_goal_diff = self.obj.pose.p - self.goal_site.pose.p
        obj_to_goal_q_diff = self.obj.pose.q - self.goal_site.pose.q
        obj_to_goal_dist_xy = torch.linalg.norm(obj_to_goal_diff[:, :2], axis=1)
        obj_to_goal_dist_q = torch.linalg.norm(obj_to_goal_q_diff, axis=1)
        # move_top_reward = 1 - torch.tanh(5 * obj_to_goal_dist_xy)
        # print(f"top moving rw {move_top_reward * is_grasped}")
        # reward += move_top_reward * is_grasped
        
        obj_to_goal_dist_xyz = torch.linalg.norm(obj_to_goal_diff, axis=1)
        b, v = obj_to_goal_diff.shape
        # print(obj_to_goal_diff.shape)
        z_unit = torch.tensor([[0, 0, 1]]*b, device=torch.device('cuda:0')).view(b, v, 1)
        # print(z_unit.shape)
        # print(obj_to_goal_dist_xyz.shape)
        cosine_diff_z = torch.bmm(obj_to_goal_diff.view(b, 1, v).float(), z_unit.float()).view(b) / obj_to_goal_dist_xyz
        # print(cosine_diff_z.shape)
        # print(is_grasped.shape)
        # cosine_diff_z.view(8, 1)
        move_top_reward = cosine_diff_z # theta approaching 90 degrees
        move_top_reward = move_top_reward + 1 - torch.tanh(5 * obj_to_goal_dist_xy) # get closed to the bin
        pose_align_reward = obj_to_goal_dist_q
        # print(move_top_reward.shape, is_grasped.view(8, 1).shape, move_top_reward * is_grasped.view(8, 1), reward.shape)
        reward += ((move_top_reward + pose_align_reward) * is_grasped)
        
        # release cube reward (TODO)
        is_cube_on_top = ((obj_to_goal_dist_xy < 1e-9).logical_and(obj_to_goal_dist_q < 1e-9))
        is_released = torch.logical_not(is_grasped)
        # print(f"cube top rw {is_cube_on_top * is_released * 2}") 
        # print(f"cube top rw {(is_cube_on_top * is_released * 2).shape}") 
        reward += is_cube_on_top * is_released * 2
        
        # static end state keeping reward
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        # print(f"static robot rw {static_reward * info['is_obj_placed']}")
        reward += static_reward * info["is_obj_placed"]
        
        # success reward
        reward[info["success"]] = 7
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 7.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward



if __name__ == "__main__":
    env = gym.make(id="myenv-v1", render_mode="sensors")
    env.reset()
    while True:
    	env.render_human()
    #img = env.render()
    #img = np.squeeze(img)
    #plt.imshow(img)
    #plt.show()


