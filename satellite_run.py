import numpy as np
import pandas as pd
from user import *
from GEO_BeamDesign import *
import matplotlib.pyplot as plt
from core import *
from SINR_Calculate import *


class Env:
    def __init__(self):
        self.beam, self.lat_log = setInitBeamCenterPos(0, [0, 0, 0], type='IRIDIUM')
        self.maxdistance = 380000
        self.user_number = 180
        self.user_per_beam = 15
        self.beam_num = 12
        self.beam_list = list(range(0, len(self.beam), 1))
        self.userlist = 0
        self.request_list = 0
        self.tti = 1
        self.rbgnumber = 6
        self.cqi = np.random.randint(15, 16, size=self.user_number)
        self.sbcqi = np.random.randint(15, 16, size=(self.user_number, self.rbgnumber))
        self.InvFlag = np.random.randint(1, 2, size=(len(self.beam_list), self.user_number))
        self.bler = np.zeros(self.user_number)
        self.current_cqi_reqest = 0
        self.current_bler_request = 0
        self.request_position_xyz_info = 0
        self.cellid = np.random.randint(0, 1, size=(self.user_number))
        self.observation_space = {'Requests': (self.user_number, 15), 'RbgMap': (len(self.beam_list), self.rbgnumber),
                                  'InvFlag': (len(self.beam_list), self.user_number)}
        self.action_space = (self.rbgnumber * len(self.beam_list), self.user_number + 1)
        self.extra_infor = {}
        self.last_tti_state = 0
        self.onedB = np.asarray([0.4814, 0.4562, 0.4675, 0.4799, 0.4921, 0.4068, 0.5146, 0.4669, 0.4242, 0.5031, 0.5241, 0.4426])

    def reset(self, on, off):
        self.extra_infor = {}
        self.tti = 1
        self.bler = np.zeros(self.user_number)
        self.cqi = np.random.randint(15, 16, size=self.user_number)
        self.sbcqi = np.random.randint(15, 16, size=(self.user_number, self.rbgnumber))
        self.userlist = initial_all_user(self.maxdistance, self.user_number, ontime=on, offtime=off)
        for i in range(len(self.userlist)):
            self.userlist[i].model2_update(tb=0, capacity=0)
        position_xyz0, position_log_lat0 = get_user_position(self.userlist)
        S0, self.request_list = get_user_traffic_info(self.userlist)
        cat_reqandposition_xyz, beam_number = userconnectsate(position_xyz0, self.beam, self.request_list, self.user_number)
        self.request_position_xyz_info = cat_reqandposition_xyz
        S0['beam_number'] = beam_number
        self.last_tti_state = S0
        self.edge_ratio = self.generate_edge_ratio(S0['Angle'].to_numpy(), S0['waitingdata'].to_numpy())
        self.RbgMap = self.generate_RbgMap(self.edge_ratio)
        self.InvFlag = self.generate_InvFlag(S0['beam_number'].to_numpy(), S0['Angle'].to_numpy())
        S_PPO_0 = {'Requests': S0.iloc[:, 0:15].to_numpy().flatten(), 'RbgMap': self.RbgMap.flatten(),
                   'InvFlag': self.InvFlag.flatten()}
        return S0, S_PPO_0

    def step(self,  obs,  action=0):
        self.extra_infor = {}
        S0, self.request_list = get_user_traffic_info(self.userlist)
        last_time_request = self.request_list
        action = self.reshape_act_tensor(action, last_time_request)
        tb_list, rbg_list, sinr, capacity = get_tb(action, self.request_position_xyz_info, obs)
        position_xyz, position_log_lat, next_state, self.request_list = updata(self.userlist, tb_list,
                                                                               last_time_request, capacity)
        cat_reqandposition_xyz, beam_number = userconnectsate(position_xyz, self.beam, self.request_list,
                                                              self.user_number)
        self.request_position_xyz_info = cat_reqandposition_xyz
        next_state['beam_number'] = beam_number

        self.last_tti_state.iloc[:, 8] = next_state.iloc[:, 8]
        self.last_tti_state.iloc[:, 15] = next_state.iloc[:, 15]
        self.extra_infor = self.generate_extra_info(self.last_tti_state, rbg_list, last_time_request, beam_number, tb_list, capacity)
        self.last_tti_state = next_state
        self.edge_ratio = self.generate_edge_ratio(next_state['Angle'].to_numpy(), next_state['waitingdata'].to_numpy())
        self.RbgMap = self.generate_RbgMap(self.edge_ratio)
        self.InvFlag = self.generate_InvFlag(next_state['beam_number'].to_numpy(), next_state['Angle'].to_numpy())
        done = False
        S_PPO_next = {'Requests': next_state.iloc[:, 0:15].to_numpy().flatten(),
                      'RbgMap': self.RbgMap.flatten(),
                      'InvFlag': self.InvFlag.flatten()}
        self.tti += 1
        return next_state, S_PPO_next, self.extra_infor, done

    def generate_extra_info(self, state, rbg_list, req, beam_number, tblist, capacity):
        self.rbg_capacity = 1480
        beam_user_connectlist = state['beam_number'].to_numpy()
        user_rbgbumber_dict = dict(zip(req, rbg_list))

        for i in range(int(max(beam_user_connectlist))):
            enb_info = state[state['beam_number'] == i + 1]
            if enb_info.empty:
                continue
            else:
                index = np.where(beam_user_connectlist == i + 1)
                rbg_number_used = 0
                enb_req_total = len(index[0])
                unassigned_total = 0
                enb_rbg_list = []
                for j in index[0]:
                    rbg_number_used += user_rbgbumber_dict[j]
                    enb_rbg_list.append(user_rbgbumber_dict[j])
                    if user_rbgbumber_dict[j] == 0:
                        unassigned_total += 1

                enb_last_txdata = enb_info['last_time_txdata'].to_numpy()
                enb_beam_number = enb_info['beam_number'].to_numpy()
                if np.sum(enb_beam_number) == 0:
                    utili_per_rbg = 0
                else:
                    mask = enb_beam_number != 0
                    enb_last_txdata = enb_last_txdata[mask]
                    txed_per_rbg = [(enb_last_txdata[i] / enb_rbg_list[i]) if enb_rbg_list[i] != 0 else 0 for i in
                                    range(len(enb_rbg_list))]
                    txed_per_rbg = np.array(txed_per_rbg) / self.rbg_capacity
                    if np.sum(np.array(enb_rbg_list)) == 0:
                        utili_per_rbg = 0
                    else:
                        mask1 = np.array(enb_rbg_list) != 0
                        utili_per_rbg = np.mean(txed_per_rbg[mask1])

                self.extra_infor['enb' + str(i + 1)] = {'enb': i+1,
                                                        'enb_req_total': enb_req_total,
                                                        'unassigned_total': unassigned_total,
                                                        'number_of_rbg_nedded': enb_info['number_of_rbg_nedded'].sum(),
                                                        'rbg_used': rbg_number_used,
                                                        'newdata': enb_info['newdata'].sum(),
                                                        'waitingdata': enb_info['waitingdata'].sum(),
                                                        'last_time_txdata': enb_info['last_time_txdata'].sum(),
                                                        # 'time_duration': enb_info['time_duration'].sum(),
                                                        'total_txdata': enb_info['total_txdata'].sum(),
                                                        'average_throughput': enb_info['average_throughput'].sum(),
                                                        'rbg_usable': self.rbgnumber,
                                                        'utili_per_rbg': utili_per_rbg,
                                                        'capacity': enb_info['capacity'].sum()}
        return self.extra_infor

    def printposition_xyz(self):
        for i in range(len(self.userlist)):
            print('user{0} position_xyz{1}'.format(i, self.userlist[i].position_xyz))

    def generate_InvFlag(self, data, angle):
        flag = np.random.randint(1, 2, size=(len(self.beam_list), self.user_number))
        for i in range(len(self.beam_list)):
            b = np.where(data == i + 1)
            flag[i][b] = 0
        for i in range(self.beam_num):
            for j in range(self.user_number):
                if flag[i][j] == 0:
                    if angle[j] > self.onedB[int(j//self.user_per_beam)]:
                        flag[i][j] = -1
        return flag

    def generate_edge_ratio(self, angel, traffic):
        edge_ratio = []
        for i in range(self.beam_num):
            edge_traffic = 0
            beam_traffic = 0
            for j in range(self.user_per_beam):
                beam_traffic += traffic[i * self.user_per_beam + j]
                if angel[i * self.user_per_beam + j] > self.onedB[i]:
                    edge_traffic += traffic[i * self.user_per_beam + j]
            ratio = edge_traffic / beam_traffic
            edge_ratio.append(ratio)
        edge_ratio = np.asarray(edge_ratio)
        return edge_ratio

    def generate_RbgMap(self, ratio):
        self.RbgMap = np.zeros((len(self.beam_list), self.rbgnumber))
        ratio_sum = np.sum(ratio)
        if ratio_sum > 1:
            ratio = ratio / ratio_sum
        rbg_index = 0
        for i in range(self.beam_num):
            if ratio[i] > 0:
                rbg_num = (ratio[i] * self.rbgnumber) // 1
                self.RbgMap[i, int(rbg_index): int(rbg_index+rbg_num)] = 1
                rbg_index = rbg_index + rbg_num
        return self.RbgMap

    def reshape_act_tensor(self, act, request_list):
        act_matrix = np.zeros((len(request_list), self.rbgnumber), dtype='int64')
        assert len(act.shape) == 1, "act维度不为(x,)"
        for i in range(len(request_list)):
            index = np.where(act == request_list[i] + 1)
            index = index[0]
            for y in range(len(index)):
                act_matrix[i][index[y] % self.rbgnumber] = 1
        return act_matrix

    def step_SA(self,  obs,  action, position, userlist):
        self.extra_infor = {}
        S0, self.request_list = get_user_traffic_info(userlist)
        action = self.reshape_act_tensor(action, self.request_list)
        tb_list, rbg_list, sinr, capacity = get_tb(action, position, obs)
        return sinr


if __name__ == '__main__':
    env = Env()
    piou = 0
    on = 9
    off = 1
    for i in range(500):
        S0, _ = env.reset(on, off)
        S0 = np.asarray(S0)
        S0 = S0[:, 1:6]
        np.savetxt("./" + str(piou) + ".txt", S0, fmt="%.18f", delimiter=" ")
        piou += 1

