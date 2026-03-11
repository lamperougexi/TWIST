from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class G1MimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 23
        obs_type = 'priv' # 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9) # Hardcode for now, 9 is base, 9 is the number of key bodies
        n_mimic_obs = 8 + 23 # 23 for dof pos
        n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single

        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        rand_reset = True
        track_root = False
     
        dof_err_w = [1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Left Leg
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Right Leg
                     0.6, 0.6, 0.6, # waist yaw, roll, pitch
                     0.8, 0.8, 0.8, 1.0, # Left Arm (unlocked)
                     0.8, 0.8, 0.8, 1.0, # Right Arm (unlocked)
                     ]

        # Default to full-body tracking (arms are no longer locked).
        lower_body_only_tracking = False
        locked_dof_names = []
        

        
        global_obs = False
        # global_obs = True
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        # height = [0, 0.02]
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.2,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.4,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.2,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.4,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.4,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.4,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 150,
                     'shoulder': 40,
                     'elbow': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 4,
                     'shoulder': 5,
                     'elbow': 5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        
        action_scale = 0.5
        decimation = 10
        # decimation = 4
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002 # 1/500
        # dt = 1/200 # 0.005
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        # file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision.urdf'
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/g1/g1_custom_collision_with_fixed_hand.urdf'
        
        # for both joint and link name
        torso_name: str = 'pelvis'  # humanoid pelvis part
        chest_name: str = 'imu_in_torso'  # humanoid chest part

        # for link name
        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll_link'  # foot_pitch is not used
        waist_name: list = ['torso_link', 'waist_roll_link', 'waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_link'
        hand_name: list = ['right_rubber_hand', 'left_rubber_hand']

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ['torso_link']
        
        
        # ========================= Inertia =========================
        # shoulder, elbow, and ankle: 0.139 * 1e-4 * 16**2 + 0.017 * 1e-4 * (46/18 + 1)**2 + 0.169 * 1e-4 = 0.003597
        # waist, hip pitch & yaw: 0.489 * 1e-4 * 14.3**2 + 0.098 * 1e-4 * 4.5**2 + 0.533 * 1e-4 = 0.0103
        # knee, hip roll: 0.489 * 1e-4 * 22.5**2 + 0.109 * 1e-4 * 4.5**2 + 0.738 * 1e-4 = 0.0251
        # wrist: 0.068 * 1e-4 * 25**2 = 0.00425
        
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + [0.003597] * 8
        
        # dof_armature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001] * 2 + [0.0] * 3 + [0.0] * 8
        
        # ========================= Inertia =========================
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.2
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.4
            
            # alive = 0.5

            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            # collision = -10.0
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            
            # feet_height = 5.0
            feet_air_time = 5.0
            
            
            ang_vel_xy = -0.01
            total_angular_momentum = -0.01
            # orientation = -0.4
            
            # base_acc = -5e-7
            # orientation = -1.0
            
            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2
            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            
            # ankle_action = -0.02
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        # push_end_effector = False
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link", "head_mocap"] # 9 key bodies
        upper_key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_elbow_link", "right_elbow_link", "head_mocap"]
        
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist_dataset.yaml"
        
        reset_consec_frames = 30
    

class G1MimicStuCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 23
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9) # Hardcode for now, 9 is the number of key bodies
        n_mimic_obs = 8 + 23 # 23 for dof pos
        
        n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single

class G1MimicStuRLCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 23
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*9) # Hardcode for now, 9 is the number of key bodies
        n_mimic_obs = 8 + 23 # 23 for dof pos
        
        n_priv_info = 3 + 1 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        class scales:
            tracking_joint_dof = 0.65
            tracking_joint_vel = 0.22
            tracking_root_pose = 0.6
            tracking_root_vel = 1.3
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.6
            
            # alive = 0.5

            feet_slip = -0.055 # lower penalty to avoid overly conservative policy
            feet_contact_forces = -4e-4
            # collision = -10.0
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.005

            feet_air_time = 4.5
            
            
            ang_vel_xy = -0.01
            total_angular_momentum = -0.01
            # orientation = -0.4
            
            # base_acc = -5e-7
            # orientation = -1.0
            
            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2
            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            
            # ankle_action = -0.02
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2


class G1MimicStuRLCMGBaseCfg(G1MimicStuRLCfg):
    """Student RL config aligned with CMG-generated references."""

    class terrain(G1MimicStuRLCfg.terrain):
        # Match CMG teacher terrain setup.
        mesh_type = 'plane'

    class motion(G1MimicStuRLCfg.motion):
        use_cmg = True
        cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
        cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/dataloader/cmg_training_data.pt"
        cmg_dt = 0.02
        motion_curriculum = False

        # Default ranges are overridden by speed-specific subclasses.
        cmg_vx_range = [0.5, 1.5]
        cmg_vy_range = [-0.3, 0.3]
        cmg_yaw_range = [-0.5, 0.5]

        # Optional single-switch command pattern per episode: cmd_A -> cmd_B.
        enable_cmd_switch = False
        cmd_switch_time_range_s = [0.1, 1.0]
        cmd_switch_once_per_episode = True
        cmd_switch_min_delta = [0.25, 0.1, 0.1]
        cmd_switch_post_window_s = 0.5

    class env(G1MimicStuRLCfg.env):
        track_root = False
        rand_reset = False

    class rewards(G1MimicStuRLCfg.rewards):
        class scales(G1MimicStuRLCfg.rewards.scales):
            # Align primary tracking scales with CMG teacher task.
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.2
            tracking_root_vel = 0.8
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_upper = 0.3

            # Explicit command-tracking objectives for CMG.
            tracking_cmd_vel = 1.5
            tracking_cmd_yaw = 1.0
            action_symmetry = 0.1

            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01


class G1MimicStuRLCMGSlowCfg(G1MimicStuRLCMGBaseCfg):
    class motion(G1MimicStuRLCMGBaseCfg.motion):
        cmg_vx_range = [0.5, 1.5]
        cmg_vy_range = [-0.3, 0.3]
        cmg_yaw_range = [-0.5, 0.5]


class G1MimicStuRLCMGMediumCfg(G1MimicStuRLCMGBaseCfg):
    class motion(G1MimicStuRLCMGBaseCfg.motion):
        cmg_vx_range = [1.5, 2.5]
        cmg_vy_range = [-0.5, 0.5]
        cmg_yaw_range = [-0.5, 0.5]


class G1MimicStuRLCMGFastCfg(G1MimicStuRLCMGBaseCfg):
    class motion(G1MimicStuRLCMGBaseCfg.motion):
        cmg_vx_range = [2.5, 3.5]
        cmg_vy_range = [-0.5, 0.5]
        cmg_yaw_range = [-0.5, 0.5]


class G1MimicStuRLCMGFastLowCfg(G1MimicStuRLCMGBaseCfg):
    """
    Fast task low-speed extension:
    keep fast upper bound while exposing lower forward commands.
    """
    class motion(G1MimicStuRLCMGBaseCfg.motion):
        cmg_vx_range = [1.0, 3.5]
        cmg_vy_range = [-0.5, 0.5]
        cmg_yaw_range = [-0.5, 0.5]

    class domain_rand(G1MimicStuRLCMGBaseCfg.domain_rand):
        # Low-speed adaptation is more sensitive to strong random pushes.
        push_interval_s = 6
        max_push_vel_xy = 0.6
        push_end_effector = False
        max_push_force_end_effector = 12.0


class G1MimicStuRLCMGFastLowCmdSwitchCfg(G1MimicStuRLCMGFastLowCfg):
    """
    Fast-low command-switch variant:
    sample cmd_A at reset, then switch to cmd_B once after a random 0.1-1.0s delay.
    """

    class env(G1MimicStuRLCMGFastLowCfg.env):
        # Lift horizon from 500 -> 600 steps (dt=0.02) for better late-stage discrimination.
        episode_length_s = 12

    class motion(G1MimicStuRLCMGFastLowCfg.motion):
        # Slightly narrow high-speed tails to reduce collapse around command switch.
        cmg_vx_range = [1.0, 3.4]
        cmg_vy_range = [-0.45, 0.45]
        cmg_yaw_range = [-0.45, 0.45]

        enable_cmd_switch = True
        # Delay switching a bit to let policy settle before cmd_B.
        cmd_switch_time_range_s = [0.2, 1.2]
        cmd_switch_once_per_episode = True
        # Keep switch meaningful but reduce instantaneous shock magnitude.
        cmd_switch_min_delta = [0.25, 0.12, 0.12]
        cmd_switch_post_window_s = 0.7

    class domain_rand(G1MimicStuRLCMGFastLowCfg.domain_rand):
        # For cmdswitch stage, reduce random push intensity to isolate switch robustness.
        push_interval_s = 8
        max_push_vel_xy = 0.45
        push_end_effector = False
        max_push_force_end_effector = 10.0


class G1MimicStuRLCMGFastNarrowCfg(G1MimicStuRLCMGBaseCfg):
    """
    Fast task curriculum warm-up:
    keep command ranges tighter at the beginning to stabilize episode length growth.
    """
    class motion(G1MimicStuRLCMGBaseCfg.motion):
        cmg_vx_range = [2.2, 3.0]
        cmg_vy_range = [-0.3, 0.3]
        cmg_yaw_range = [-0.25, 0.25]


class G1MimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 30_002 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        # Transformer params
        # learning_rate = 1e-4 #1.e-3 #5.e-4
        # schedule = 'fixed' # could be adaptive, fixed
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        


class G1MimicStuRLCfgDAgger(G1MimicStuRLCfg):
    seed = 1

    class teachercfg(G1MimicPrivCfgPPO):
        pass

    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 30_002
        warm_iters = 100

        # logging
        save_interval = 100
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None

        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        learning_rate = 8e-5
        desired_kl = 0.004
        num_learning_epochs = 3
        entropy_coef = 0.005

        dagger_coef_anneal_steps = 50000

        dagger_coef = 0.03
        dagger_coef_min = 0.01
        # dagger_coef = 0.0
        # dagger_coef_min = 0.0  # Minimum value for dagger_coef

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128


class G1MimicStuRLCMGCfgDAgger(G1MimicStuRLCfgDAgger):
    """
    DAgger config specialized for CMG student tasks.
    Uses stronger teacher imitation to stabilize early-stage training.
    """

    class algorithm(G1MimicStuRLCfgDAgger.algorithm):
        # Stronger teacher guidance for CMG student alignment.
        dagger_coef = 0.10
        dagger_coef_min = 0.03
        dagger_coef_anneal_steps = 80000


class G1MimicStuRLCMGFastCfgDAgger(G1MimicStuRLCMGCfgDAgger):
    """DAgger/PPO settings specialized for full fast CMG fine-tuning stability."""

    class algorithm(G1MimicStuRLCMGCfgDAgger.algorithm):
        # Match fast_low stability settings: smaller updates, fixed LR, tighter clipping.
        learning_rate = 6e-5
        schedule = 'fixed'
        num_learning_epochs = 3
        clip_param = 0.15
        max_grad_norm = 0.8
        entropy_coef = 0.004
        desired_kl = 0.003


class G1MimicStuRLCMGFastLowCfgDAgger(G1MimicStuRLCMGCfgDAgger):
    """DAgger/PPO settings specialized for fast_low adaptation stability."""

    class algorithm(G1MimicStuRLCMGCfgDAgger.algorithm):
        # Keep updates conservative to reduce value spikes during broad-range fine-tuning.
        learning_rate = 6e-5
        schedule = 'fixed'
        num_learning_epochs = 3
        clip_param = 0.15
        max_grad_norm = 0.8
        entropy_coef = 0.004
        desired_kl = 0.003


class G1MimicStuRLCMGFastLowCmdSwitchCfgDAgger(G1MimicStuRLCMGFastLowCfgDAgger):
    """Stabilized DAgger/PPO settings for cmdswitch fine-tuning near long-horizon plateau."""

    class algorithm(G1MimicStuRLCMGFastLowCfgDAgger.algorithm):
        # Slightly tighter policy updates and lower entropy reduce random late-episode falls.
        desired_kl = 0.0025
        entropy_coef = 0.0035


# ==================== CMG-based Configurations ====================

class G1MimicCMGBaseCfg(G1MimicPrivCfg):
    """Base configuration for CMG-based motion generation."""

    class terrain(G1MimicPrivCfg.terrain):
        # Use simple plane terrain to reduce GPU memory usage
        mesh_type = 'plane'

    class motion(G1MimicPrivCfg.motion):
        # Enable CMG motion generation
        use_cmg = True
        cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
        cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/dataloader/cmg_training_data.pt"

        # CMG operates at 50 Hz
        cmg_dt = 0.02

        # Default velocity ranges (overridden by speed-specific configs)
        cmg_vx_range = [0.5, 1.5]
        cmg_vy_range = [-0.3, 0.3]
        cmg_yaw_range = [-0.5, 0.5]

        # Disable motion curriculum for CMG (not applicable)
        motion_curriculum = False

    class env(G1MimicPrivCfg.env):
        # For CMG, we don't track root position since it's generated
        track_root = False
        # Random reset not applicable for CMG
        rand_reset = False

        # CMG uses full-body DOF tracking by default.
        dof_err_w = [1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Left Leg (unchanged)
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Right Leg (unchanged)
                     0.6, 0.6, 0.6,                   # Waist (unchanged)
                     0.8, 0.8, 0.8, 1.0,             # Left Arm (unlocked)
                     0.8, 0.8, 0.8, 1.0,             # Right Arm (unlocked)
                     ]

    class rewards(G1MimicPrivCfg.rewards):
        """CMG-specific rewards including velocity command tracking."""
        class scales(G1MimicPrivCfg.rewards.scales):
            # Inherit all existing reward scales
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.2
            tracking_root_vel = 0.8
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_upper = 0.3

            # CMG velocity command tracking rewards
            tracking_cmd_vel = 1.5   # Track vx, vy commands
            tracking_cmd_yaw = 1.0   # Track yaw rate command

            # Action symmetry: weak reward encouraging left-right symmetric actions
            action_symmetry = 0.1

            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01


class G1MimicCMGSlowCfg(G1MimicCMGBaseCfg):
    """Slow speed training configuration (1 m/s)."""

    class motion(G1MimicCMGBaseCfg.motion):
        use_cmg = True
        cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
        cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/dataloader/cmg_training_data.pt"
        cmg_dt = 0.02
        motion_curriculum = False

        # Slow speed: ~1 m/s forward
        cmg_vx_range = [0.5, 1.5]
        cmg_vy_range = [-0.3, 0.3]
        cmg_yaw_range = [-0.5, 0.5]


class G1MimicCMGMediumCfg(G1MimicCMGBaseCfg):
    """Medium speed training configuration (2 m/s)."""

    class motion(G1MimicCMGBaseCfg.motion):
        use_cmg = True
        cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
        cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/dataloader/cmg_training_data.pt"
        cmg_dt = 0.02
        motion_curriculum = False

        # Medium speed: ~2 m/s forward
        cmg_vx_range = [1.5, 2.5]
        cmg_vy_range = [-0.5, 0.5]
        cmg_yaw_range = [-0.5, 0.5]


class G1MimicCMGFastCfg(G1MimicCMGBaseCfg):
    """Fast speed training configuration (3 m/s)."""

    class motion(G1MimicCMGBaseCfg.motion):
        use_cmg = True
        cmg_model_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/runs/cmg_20260123_194851/cmg_final.pt"
        cmg_data_path = f"{LEGGED_GYM_ROOT_DIR}/../cmg_workspace/dataloader/cmg_training_data.pt"
        cmg_dt = 0.02
        motion_curriculum = False

        # Fast speed: ~3 m/s forward
        cmg_vx_range = [2.5, 3.5]
        cmg_vy_range = [-0.5, 0.5]
        cmg_yaw_range = [-0.5, 0.5]


# PPO configurations for CMG environments
class G1MimicCMGSlowCfgPPO(G1MimicPrivCfgPPO):
    seed = 1

    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 30_002
        save_interval = 100
        experiment_name = 'cmg_slow'


class G1MimicCMGMediumCfgPPO(G1MimicPrivCfgPPO):
    seed = 1

    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 30_002
        save_interval = 100
        experiment_name = 'cmg_medium'


class G1MimicCMGFastCfgPPO(G1MimicPrivCfgPPO):
    seed = 1

    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 30_002
        save_interval = 100
        experiment_name = 'cmg_fast'

    class algorithm(G1MimicPrivCfgPPO.algorithm):
        # Stabilize late-stage fast-teacher training:
        # use gentler PPO updates to reduce episodic collapse spikes.
        learning_rate = 8e-5
        schedule = 'fixed'
        num_learning_epochs = 3
        clip_param = 0.15
        max_grad_norm = 0.8
        entropy_coef = 0.003
