# -*-coding:utf-8-*-
import math
import numpy as np
import math as m


class calculate_tool:
    def __init__(self):
        self.Gr_user = 40
        self.velocity = 3e8
        self.frequency = 2e10   # Hz
        self.user_number = 0
        self.Hgeo = 35786000   # m
        self.path_loss = -209.53778  # dBw
        self.gama = 0.5
        self.noisy = 2.5118864315095823e-12
        self.bw = 500e6
        self.rbg_number = 6
        self.rb_bw = self.bw / self.rbg_number
        self.PowerT_beam = [24.2041, 23.2720, 23.7058, 24.1472, 24.5833, 21.2876, 25.3695, 23.6879, 22.0189, 24.9629, 25.6658, 22.7529]
        self.G_peak = [39.3776, 39.8437, 39.6268, 39.4061, 39.1880, 40.8359, 38.7949, 39.6357, 40.4702, 38.9983, 38.6468, 40.1032]
        self.onedB = [0.4822, 0.4570, 0.4686, 0.4807, 0.4929, 0.4077, 0.5157, 0.4681, 0.4252, 0.5038, 0.5246, 0.4436]

    def get_loss_path(self, position_info):
        self.user_number = len(position_info)
        LOSS_PATH = np.zeros((1, self.user_number))
        for i in range(self.user_number):
            user_position = position_info[i][1:4]
            beam_position = position_info[i][4:7]
            distance_ub = np.sqrt(np.sum((user_position - beam_position) ** 2))
            distance_us = np.sqrt(distance_ub ** 2 + self.Hgeo ** 2)
            loss_path = ((4 * m.pi * self.frequency * distance_us) / self.velocity) ** (-2)    # W
            Loss_path = 10 * np.log10(loss_path)   # dBw
            LOSS_PATH[0][i] = Loss_path
        return LOSS_PATH

    def get_gain(self, position_info):
        self.user_number = len(position_info)
        beam_number_connect = self.user_number
        theta_matrix = np.zeros((self.user_number, beam_number_connect))
        Gain_matrix = np.zeros((self.user_number, beam_number_connect))
        distance_matrix = np.zeros((self.user_number, beam_number_connect))
        for i in range(self.user_number):
            user_position = position_info[i][1:4]
            for j in range(beam_number_connect):
                beam_position = position_info[j][4:7]
                beam_number = int(position_info[j][10])
                distance = np.sqrt(np.sum((user_position - beam_position) ** 2))
                distance_matrix[i][j] = distance
                theta = np.degrees(np.arctan(distance / self.Hgeo))
                theta_matrix[i][j] = theta
                Gain_matrix[i][j] = self.G_peak[beam_number-1] - ((12 * (10 ** (self.G_peak[beam_number-1] / 10))) / self.gama) * np.square(theta_matrix[i][j] / (70 * np.pi))
        Gain_matrix = 10 ** (Gain_matrix / 10)
        return Gain_matrix

    def get_sinr(self, action, position_info, traffic_info):
        rbgnumber = action.shape[1]
        Gain_matrix = self.get_gain(position_info)
        Path_loss_matrxi = self.get_loss_path(position_info)
        sinr_matrix = np.zeros((self.user_number, rbgnumber))
        capa_matrix = np.zeros((self.user_number, rbgnumber))
        traffic_info = np.asarray(traffic_info).reshape(180, -1)
        traffic_info = traffic_info[np.where(traffic_info[:, 7] == 1), :]
        angle = traffic_info[:, :, 4].reshape(-1)
        for i in range(self.user_number):
            for j in range(rbgnumber):
                if action[i][j] == 0:
                    continue
                else:
                    index = np.where(action[:, j] == 1)
                    Gain_self = 10 * np.log10(Gain_matrix[i][i])
                    number = int(position_info[i][10])
                    power_self = 10 ** ((Gain_self + self.Gr_user + Path_loss_matrxi[0][i]) / 10) * (10 ** (self.PowerT_beam[number-1]/10))
                    if len(index[0]) == 1:
                        sinr = power_self / (self.noisy)
                        sinr_matrix[i][j] = sinr
                        continue
                    index2 = np.where(index[0] == i)
                    other_user_interference_index = np.delete(index[0], index2[0])
                    interference = 0
                    for k in range(len(other_user_interference_index)):
                        beam_number = int(position_info[other_user_interference_index[k]][10])
                        if angle[other_user_interference_index[k]] < self.onedB[beam_number - 1]:
                            interf = 0
                        else:
                            Gain_interf = 10 * np.log10(Gain_matrix[i][other_user_interference_index[k]])
                            interf = 10 ** ((Gain_interf) / 10) * (10 ** (self.PowerT_beam[beam_number - 1] / 10))
                        interference += interf
                    interference = 10 ** ((self.Gr_user + Path_loss_matrxi[0][i]) / 10) * interference
                    sinr = power_self / (self.noisy + interference)
                    capa = power_self / self.noisy
                    sinr_matrix[i][j] = sinr
                    capa_matrix[i][j] = capa
        return sinr_matrix, capa_matrix


def get_tb(action, position_info, traffic_info):
    tool = calculate_tool()
    # 有请求用户的位置信息,position_info,(用户编号,用户位置,波束位置,星下点位置)
    sinr, capa = tool.get_sinr(action, position_info, traffic_info)
    sinr = np.log2(sinr + 1) * tool.rb_bw / 1000
    capacity = np.log2(capa + 1) * tool.rb_bw / 1000
    tb = np.sum(sinr, axis=1)
    capacity = np.sum(capacity, axis=1)
    rbglist=np.sum(action, axis=1)
    return tb, rbglist, sinr, capacity


def get_capacity(beam_num):
    tool = calculate_tool()
    cap_label = []
    for i in range(beam_num):
        power_self = 10 ** ((tool.G_peak[i] + tool.Gr_user + tool.path_loss) / 10) * (
                    10 ** (tool.PowerT_beam[i] / 10))
        sinr = power_self / tool.noisy
        cap = np.log2(sinr + 1) * tool.bw / 1000
        cap_label.append(cap)
    return cap_label


if __name__ == '__main__':
    label = []
    tool = calculate_tool()
    for i in range(12):
        power_self = 10 ** ((tool.G_peak[i] + tool.Gr_user + tool.path_loss) / 10) * (10 ** (tool.PowerT_beam[i]/10))
        sinr = power_self / tool.noisy
        cap = np.log2(sinr + 1) * tool.bw / 1000
        label.append(cap)
    print("label", label)
    label_1 = [1170427.0388427917, 1108976.7032083324, 1137409.6666052616, 1166643.4797618384, 1195810.576419014, 982913.1870934572, 1249107.7273640872, 1136223.9501351079, 1028573.0350210491, 1221446.8528336838, 1269427.1150338312, 1075346.1870396722]
    print("****************", np.sum(np.array(label_1))*50)