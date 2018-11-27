import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_pose = init_pose
        #self.target_pos=target_pos
        # Goal
        '''if init_pose is None:
            print("pose setting is auto")
            self.sim.init_pose = init_pose if init_pose is not None else np.array([0., 0., 0.]) '''

    def get_reward(self):
        """,同时对z速度进行奖励"""
        reward = np.tanh(1-0.0001*(self.sim.v[0]+0.5*self.sim.v[1])+0.0002*self.sim.v[2]).sum()
        return reward

    def step(self, rotor_speeds):
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            if done:
                reward+=10
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state