import numpy as np
import gym
import dartpy as dart
from dartHelper import buildFromFile 
import xml.etree.ElementTree as ET
from bvhparser import MyBVH



class Env(gym.Env):
    def __init__(self, metadata = "data/env.xml"):
        
        self.world = dart.simulation.World()
        
        self.skel = None
        self.target_skel = None
        self.ground = None 
        
        self.simulationHz = 480
        self.controlHz = 30
        self.bvhs = None
        self.bvh_idx = 0

        self.ees_name = ["Head", "HandL", "HandR", "TalusL", "TalusR"]


        self.loading_xml(metadata)

        self.world.setTimeStep(1.0 / self.simulationHz)
        self.world.setGravity([0, -9.81, 0])
        
        self.target_pos = None
        self.target_vel = None
        
        self.target_displacement = np.zeros(self.skel.getNumDofs() - self.skel.getJoint(0).getNumDofs())
        
        self.cur_obs = None
        self.cur_reward = 0.0
        self.cur_root_T = None
        self.cur_root_T_inv = None

        self.reset()
        
        self.num_obs = len(self.get_obs())
        self.num_action = len(self.get_zero_action())
        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_obs,))
        self.action_space = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.num_action,))

        self.action_scale = 0.04


    def loading_xml(self, metadata):
        ## xml loading
        doc = ET.parse(metadata)
        root = doc.getroot()
        bvh_info = None
        for child in root:
            if child.tag == "skeleton":
                self.skel, bvh_info = buildFromFile(child.text)
                self.target_skel = self.skel.clone()
                self.world.addSkeleton(self.skel)
            elif child.tag == "ground":
                self.ground, _ = buildFromFile(child.text)
                self.ground.setMobile(False)
                self.world.addSkeleton(self.ground)
            elif child.tag == "simHz":
                self.simulationHz = int(child.text)
            elif child.tag == "controlHz":
                self.controlHz = int(child.text)
            elif child.tag == "bvh":
                if child.text[-3:] == "bvh":
                    self.bvhs = [MyBVH(child.text, bvh_info, self.skel)]
                else:
                    import os
                    files = os.listdir(child.text)
                    self.bvhs = [MyBVH(f, bvh_info, self.skel) for f in files if f[-3:] == "bvh"]
        
    def get_zero_action(self):
        return np.zeros(self.skel.getNumDofs() - self.skel.getJoint(0).getNumDofs())

    def get_root_T(self):
        root_y = np.array([0, 1, 0])
        root_z = self.skel.getRootBodyNode().getTransform().rotation() @ np.array([0, 0, 1])
        root_z[1] = 0.0
        root_z = root_z / np.linalg.norm(root_z)
        root_x = np.cross(root_y, root_z)

        root_rot = np.array([root_x, root_y, root_z]).transpose()

        root_T = np.identity(4)
        root_T[:3, :3] = root_rot
        root_T[:3,  3] = self.skel.getRootBodyNode().getTransform().translation()
        root_T[1,   3] = 0.0
        return root_T

    def get_obs(self):
        return self.cur_obs
        
    def update_target(self, time):
        self.target_pos = self.bvhs[self.bvh_idx].getPose(time)
        pos_next = self.bvhs[self.bvh_idx].getPose(time + 1.0 / self.controlHz)
        self.target_vel = self.skel.getPositionDifferences(pos_next, self.target_pos) * self.controlHz
        self.target_skel.setPositions(self.target_pos)
        self.target_skel.setVelocities(self.target_vel)

    def update_obs(self):
        # Skeleton Information 
        bn_lin_pos = []
        bn_lin_vel = []
        bn_6d_orientation = []
        bn_ang_vel = []

        w_bn_ang_vel = 0.1
        
        for bn in self.skel.getBodyNodes():
            p = np.ones(4)
            p[:3] = bn.getWorldTransform().translation()
            bn_lin_pos.append((self.cur_root_T_inv @ p)[:3])  
            bn_lin_vel.append(w_bn_ang_vel * self.cur_root_T_inv[:3,:3] @ bn.getCOMLinearVelocity())        
            bn_6d_orientation.append((self.cur_root_T_inv[:3,:3]@(bn.getWorldTransform().rotation())).flatten()[:6])
            bn_ang_vel.append(self.cur_root_T_inv[:3,:3] @ bn.getAngularVelocity())
        

        # Target 
        
        # phase = (self.world.getTime() - self.bvhs[self.bvh_idx].bvh_time * (self.world.getTime() // self.bvhs[self.bvh_idx].bvh_time)) / self.bvhs[self.bvh_idx].bvh_time
        # phase *= 2 * np.pi
        # phase = [np.array([np.sin(phase), np.cos(phase)])]
        
        target_bn = []
        for bn in self.target_skel.getBodyNodes():
            p = np.ones(4)
            p[:3] = bn.getWorldTransform().translation()
            target_bn.append((self.cur_root_T_inv @ p)[:3])

        self.cur_obs = np.concatenate(bn_lin_pos + bn_lin_vel + bn_6d_orientation + bn_ang_vel + target_bn) 

    def update_root_T(self):
        self.cur_root_T = self.get_root_T()
        self.cur_root_T_inv = np.linalg.inv(self.cur_root_T)

    def reset(self):
        # dynamics reset 
        time = (np.random.rand() % 1.0) * self.bvhs[self.bvh_idx].bvh_time
        self.update_target(time)

        solver = self.world.getConstraintSolver()
        solver.setCollisionDetector(dart.collision.BulletCollisionDetector())
        solver.clearLastCollisionResult()

        self.skel.setPositions(self.target_pos)
        self.skel.setVelocities(self.target_vel)

        self.skel.clearInternalForces()
        self.skel.clearExternalForces()
        self.skel.clearConstraintImpulses()

        self.world.setTime(time)
        
        self.update_root_T()
        self.update_obs()
        return self.get_obs()
    
    def getSPDForces(self, p_desired):
        kp = 300.0 * np.ones(self.skel.getNumDofs())
        kv = 20.0 * np.ones(self.skel.getNumDofs())
        
        kp[:self.skel.getJoint(0).getNumDofs()] = 0.0
        kv[:self.skel.getJoint(0).getNumDofs()] = 0.0

        q = self.skel.getPositions()
        dq = self.skel.getVelocities()
        dt = 1.0 / self.simulationHz

        M_inv = np.linalg.inv(self.skel.getMassMatrix() + dt * np.diag(kv))
        qdqdt = q + dq * dt

        p_diff = -kp * self.skel.getPositionDifferences(qdqdt, p_desired)
        v_diff = -kv * dq

        ddq = M_inv @ (-self.skel.getCoriolisAndGravityForces() + p_diff + v_diff + self.skel.getConstraintForces())
        tau = p_diff + v_diff - dt * kv * ddq
        tau[:self.skel.getJoint(0).getNumDofs()] = 0.0

        return tau

    def get_reward(self):
        # Joint reward
        q_diff = self.skel.getPositionDifferences(self.skel.getPositions(), self.target_pos)
        r_q = np.exp(-20.0 * np.linalg.norm(q_diff)**2 / len(q_diff))

        # EE reward 
        ee_diff = np.concatenate([(self.skel.getBodyNode(ee).getCOM() - self.target_skel.getBodyNode(ee).getCOM()) for ee in self.ees_name])
        r_ee = np.exp(-40 * np.linalg.norm(ee_diff)**2 / len(ee_diff))

        # COM reward 
        com_diff = self.skel.getCOM() - self.target_skel.getCOM()
        r_com = np.exp(-10 * np.linalg.norm(com_diff)**2 / len(com_diff))

        w_alive = 0.05
        return (w_alive + r_q * (1.0 - w_alive)) * (w_alive + r_ee * (1.0 - w_alive)) * (w_alive + r_com * (1.0 - w_alive))
    

    def step(self, action):
        self.target_displacement = self.action_scale * action
        self.update_target(self.world.getTime())

        pd_target = self.target_pos.copy()
        pd_target[self.skel.getJoint(0).getNumDofs():] += self.target_displacement

        for _ in range(int(self.simulationHz / self.controlHz)):
            tau = self.getSPDForces(pd_target)
            self.skel.setForces(tau)
            self.world.step()

        self.update_root_T()
        self.update_obs()

        self.cur_reward = self.get_reward()
        
        info = {}
        info["end"] = 0 

        if self.cur_reward < 0.1:
            info["end"] = 1
        
        if self.world.getTime() > 10.0:
            info["end"] = 3 
        
        return self.get_obs(), self.cur_reward, info["end"] != 0, info
