import bvh
import numpy as np
import quaternion

import dartpy as dart
import math

from numpy.linalg import matrix_power

def inplaneXY(R):
    y = np.array([0, 1, 0])
    z = R[:3,:3] @ np.array([0, 0, 1])
    z[1] = 0
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)

    return np.array([x, y, z]).transpose()


def eulerToMatrix(angle, order):
    order = order.upper()
    angle = angle * math.pi / 180.0
    if order == "XYZ":
        return dart.math.eulerXYZToMatrix(angle)
    elif order == "XZY":
        return dart.math.eulerXZYToMatrix(angle)
    elif order == "YXZ":
        return dart.math.eulerYXZToMatrix(angle)
    elif order == "YZX":
        return dart.math.eulerYZXToMatrix(angle)
    elif order == "ZXY":
        return dart.math.eulerZXYToMatrix(angle)
    elif order == "ZYX":
        return dart.math.eulerZYXToMatrix(angle)
    else:
        print("Unknown Euler order")
        return None
    
class MyBVH():
    def __init__(self, filename, bvh_info, skel, T_frame=None):
        with open(filename, 'r') as f:
            self.mocap = bvh.Bvh(f.read())
        if self.mocap == None:
            print("BVH file not found")
            return
        
        self.Tpose_frame = T_frame
        
        self.skel = skel
        self.bvh_info = bvh_info
        self.bvh_time = self.mocap.frame_time * self.mocap.nframes
        self.mocap_refs = np.zeros([self.mocap.nframes, self.skel.getNumDofs()])

        self.mocap_refs_mat = {}
        self.Tpose_mat = {}

        self.root_T = np.identity(4)
        self.root_jn = None

        for jn in self.mocap.get_joints():
            self.mocap_refs_mat[jn.name] = []
            joint_chs = self.mocap.joint_channels(jn.name)
            if len(joint_chs) == 6:
                xpos = [ch_n for ch_n in joint_chs if ch_n.upper() == "XPOSITION"][0]
                ypos = [ch_n for ch_n in joint_chs if ch_n.upper() == "YPOSITION"][0]
                zpos = [ch_n for ch_n in joint_chs if ch_n.upper() == "ZPOSITION"][0]

                rots = [ch_n for ch_n in joint_chs if ch_n.upper()[1:] == "ROTATION"]
                euler_order = "".join([ch_n[0]for ch_n in joint_chs if ch_n.upper()[1:] == "ROTATION"])

                pos = np.array(self.mocap.frames_joint_channels(jn.name, [xpos, ypos, zpos])) * 0.01
                rot = np.array(self.mocap.frames_joint_channels(jn.name, rots))

                for i in range(self.mocap.nframes):
                    T_mat = np.identity(4)
                    T_mat[ :3, :3] = eulerToMatrix(rot[i], euler_order)
                    T_mat[0:3,  3] = pos[i]

                    if (self.Tpose_frame != None and i == self.Tpose_frame) or (self.Tpose_frame == None and i == 0):
                        self.Tpose_mat[jn.name] = T_mat.copy()
                    self.mocap_refs_mat[jn.name].append(T_mat)

            elif len(joint_chs) == 3:
                rots = [ch_n for ch_n in joint_chs if ch_n.upper()[1:] == "ROTATION"]
                euler_order = "".join([ch_n[0] for ch_n in joint_chs if ch_n.upper()[1:] == "ROTATION"])
                rot = np.array(self.mocap.frames_joint_channels(jn.name, rots))
                for i in range(self.mocap.nframes):
                    if jn.parent == None:
                        T_mat = eulerToMatrix(rot[i], euler_order)
                    else:
                        T_mat = self.mocap_refs_mat[jn.parent.name][i][:3, :3] @ eulerToMatrix(rot[i], euler_order)
                    self.mocap_refs_mat[jn.name].append(T_mat)

                    if self.Tpose_frame != None and i == self.Tpose_frame:
                        self.Tpose_mat[jn.name] = T_mat.copy()

        self.setRefs()

    def setRefs(self):
        for skel_jn_i in range(self.skel.getNumJoints()):
            skel_jn = self.skel.getJoint(skel_jn_i)

            if skel_jn.getName() in self.bvh_info.keys() and self.bvh_info[skel_jn.getName()] in self.mocap_refs_mat.keys():
                for i in range(len(self.mocap_refs_mat[self.bvh_info[skel_jn.getName()]])):
                    T = self.mocap_refs_mat[self.bvh_info[skel_jn.getName()]][i].copy()
                    if T.shape == (4, 4):
                        T[:3, 3] -= self.Tpose_mat[self.bvh_info[skel_jn.getName()]][:3, 3]
                        if self.Tpose_frame != None:
                            T[:3, :3] = T[:3, :3] @ self.Tpose_mat[self.bvh_info[skel_jn.getName()]][:3, :3].transpose()

                    else:
                        if self.Tpose_frame != None:
                            T = T @ self.Tpose_mat[self.bvh_info[skel_jn.getName()]].transpose()

                    T_parent = np.identity(4)
                    if self.mocap.joint_parent_index(self.bvh_info[skel_jn.getName()]) != -1:
                        T_parent = self.mocap_refs_mat[self.mocap.joint_parent(self.bvh_info[skel_jn.getName()]).name][i][:3, :3]
                        if self.Tpose_frame != None:
                            T_parent = T_parent @ self.Tpose_mat[self.mocap.joint_parent(self.bvh_info[skel_jn.getName()]).name][:3, :3].transpose()

                    T_net = T_parent.transpose() @ T

                    if skel_jn.getNumDofs() == 1:
                        self.mocap_refs[i, skel_jn.getIndexInSkeleton(0):skel_jn.getIndexInSkeleton(0) + skel_jn.getNumDofs()] = dart.math.AngleAxis(T_net).angle()
                    else:
                        self.mocap_refs[i, skel_jn.getIndexInSkeleton(0):skel_jn.getIndexInSkeleton(0) + skel_jn.getNumDofs()] = skel_jn.convertToPositions(T_net)

        self.root_jn = self.skel.getJoint(0)
        root_0 = self.root_jn.convertToTransform(self.mocap_refs[0, 0:self.root_jn.getNumDofs()]).matrix()
        root_0[:3, :3] = inplaneXY(root_0)

        self.root_T = self.root_jn.convertToTransform(self.mocap_refs[-1, 0:self.root_jn.getNumDofs()]).matrix()
        self.root_T[:3, :3] = inplaneXY(self.root_T)

        self.root_T = self.root_T @ np.linalg.inv(root_0)
        self.root_T[1, 3] = 0.0

    def getLowerBoundPose(self, time):
        iter = time // self.bvh_time
        net_time = time - self.bvh_time * iter
        frame = int(net_time / self.mocap.frame_time)

        res = self.mocap_refs[frame].copy()
        T = matrix_power(self.root_T, int(iter))
        res[:self.root_jn.getNumDofs()] = self.root_jn.convertToPositions(T @ self.root_jn.convertToTransform(res[:self.root_jn.getNumDofs()]).matrix())

        return res
    
    def getPose(self, time):
        cur_pose = self.getLowerBoundPose(time)
        next_pose = self.getLowerBoundPose(time + self.mocap.frame_time)
        slerp_pose = np.zeros(cur_pose.shape)
        alpha = (time - ((time // self.mocap.frame_time) * self.mocap.frame_time)) / self.mocap.frame_time

        for i in range(self.skel.getNumJoints()):
            jn = self.skel.getJoint(i)
            if jn.getNumDofs() == 1:
                slerp_pose[jn.getIndexInSkeleton(0)] = cur_pose[jn.getIndexInSkeleton(0)] * (1.0 - alpha) + next_pose[jn.getIndexInSkeleton(0)] * alpha
            elif jn.getNumDofs() == 3:
                q1 = quaternion.from_rotation_vector(cur_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + 3])
                q2 = quaternion.from_rotation_vector(next_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + 3])
                slerp_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0)+ 3] = quaternion.as_rotation_vector(quaternion.slerp(q1, q2, 0.0, 1.0, alpha))
            elif jn.getNumDofs() == 6:
                q1 = quaternion.from_rotation_vector(cur_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + 3])
                q2 = quaternion.from_rotation_vector(next_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + 3])
                slerp_pose[jn.getIndexInSkeleton(0):jn.getIndexInSkeleton(0) + 3] = quaternion.as_rotation_vector(quaternion.slerp(q1, q1, 0.0, 1.0, alpha))
                slerp_pose[jn.getIndexInSkeleton(3):jn.getIndexInSkeleton(3) + 3] = cur_pose[jn.getIndexInSkeleton(3):jn.getIndexInSkeleton(3) + 3] * (1.0 - alpha) + next_pose[jn.getIndexInSkeleton(3):jn.getIndexInSkeleton(3) + 3] * alpha

        return slerp_pose

