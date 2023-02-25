import platform
import random
import matplotlib.cbook
import torch
import time
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from user import *
import math as m

np.set_printoptions(threshold=np.inf)

ay, az, ap, ac, ap, ag, adB, adb, ai = [], [], [], [], [], [], [], [], []

plt.ion()
localtime = time.asctime(time.localtime(time.time()))

# 超参数
EPOCH = 500
BATCH_SIZE = 500
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

mm = MinMaxScaler()
train_data = []
train_data_label = []
for num in range(2000, 2500, 1):
    action = []
    data = np.loadtxt("./UserRequest/" + str(num) + ".txt")
    data = data.reshape(-1, 1)
    train_data_label.append(data.reshape(-1))
    train_data.append(data.reshape(-1))
    num += 1
train_data = np.asarray(train_data)
train_data_label = np.asarray(train_data_label)
train = Data.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_data_label).float())
train_loader = Data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=1)

        self.power = nn.Sequential(
            nn.Linear(900, 800),
            nn.Tanh(),
            nn.Linear(800, 700),
            nn.Tanh(),
            nn.Linear(700, 600),
            nn.Tanh(),
            nn.Linear(600, 500),
            nn.Tanh(),
            nn.Linear(500, 400),
            nn.Tanh(),
            nn.Linear(400, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 48),
            nn.Tanh(),
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Linear(24, 12),
            nn.Sigmoid())

    def run(self, x):
        power = self.power(x)
        power_distribution = power.view(-1, 12)
        power_softmax = self.softmax(power_distribution)
        return power_distribution, power_softmax


# 根据发射功率，计算中心用户与边缘用户的比例
def calculate_interf_ration(power_selection, beam_gain, b_y, beam_traffic):
    height = 35786000
    theta_3dB = []
    theta_1dB = []
    # gain_decrease = []
    for item in range(12):
        theta_3dB_i = 70 * m.pi / (m.sqrt(2 * (m.pow(10, beam_gain[item] / 10))))
        theta_3dB.append(theta_3dB_i)
        theta_1dB_i = 70 * m.pi * m.sqrt(0.5 / (12 * m.pow(10, beam_gain[item] / 10)))
        theta_1dB.append(theta_1dB_i)
    angle = b_y[:, 3]
    traffic = b_y[:, 4]

    beam_angele_1 = angle[0:15]
    beam_angele_2 = angle[15:30]
    beam_angele_3 = angle[30:45]
    beam_angele_4 = angle[45:60]
    beam_angele_5 = angle[60:75]
    beam_angele_6 = angle[75:90]
    beam_angele_7 = angle[90:105]
    beam_angele_8 = angle[105:120]
    beam_angele_9 = angle[120:135]
    beam_angele_10 = angle[135:150]
    beam_angele_11 = angle[150:165]
    beam_angele_12 = angle[165:180]

    beam_traffic_1 = traffic[0:15]
    beam_traffic_2 = traffic[15:30]
    beam_traffic_3 = traffic[30:45]
    beam_traffic_4 = traffic[45:60]
    beam_traffic_5 = traffic[60:75]
    beam_traffic_6 = traffic[75:90]
    beam_traffic_7 = traffic[90:105]
    beam_traffic_8 = traffic[105:120]
    beam_traffic_9 = traffic[120:135]
    beam_traffic_10 = traffic[135:150]
    beam_traffic_11 = traffic[150:165]
    beam_traffic_12 = traffic[165:180]

    center_traffic_beam1 = center_traffic_beam2 = center_traffic_beam3 = \
        center_traffic_beam4 = center_traffic_beam5 = center_traffic_beam6 = \
        center_traffic_beam7 = center_traffic_beam8 = center_traffic_beam9 = \
        center_traffic_beam10 = center_traffic_beam11 = center_traffic_beam12 = 0

    for i in range(len(beam_angele_1)):
        if beam_angele_1[i] <= theta_1dB[0]:
            center_traffic_beam1 += beam_traffic_1[i]
    for i in range(len(beam_angele_2)):
        if beam_angele_2[i] <= theta_1dB[1]:
            center_traffic_beam2 += beam_traffic_2[i]
    for i in range(len(beam_angele_3)):
        if beam_angele_3[i] <= theta_1dB[2]:
            center_traffic_beam3 += beam_traffic_3[i]
    for i in range(len(beam_angele_4)):
        if beam_angele_4[i] <= theta_1dB[3]:
            center_traffic_beam4 += beam_traffic_4[i]
    for i in range(len(beam_angele_5)):
        if beam_angele_5[i] <= theta_1dB[4]:
            center_traffic_beam5 += beam_traffic_5[i]
    for i in range(len(beam_angele_6)):
        if beam_angele_6[i] <= theta_1dB[5]:
            center_traffic_beam6 += beam_traffic_6[i]
    for i in range(len(beam_angele_7)):
        if beam_angele_7[i] <= theta_1dB[6]:
            center_traffic_beam7 += beam_traffic_7[i]
    for i in range(len(beam_angele_8)):
        if beam_angele_8[i] <= theta_1dB[7]:
            center_traffic_beam8 += beam_traffic_8[i]
    for i in range(len(beam_angele_9)):
        if beam_angele_9[i] <= theta_1dB[8]:
            center_traffic_beam9 += beam_traffic_9[i]
    for i in range(len(beam_angele_10)):
        if beam_angele_10[i] <= theta_1dB[9]:
            center_traffic_beam10 += beam_traffic_10[i]
    for i in range(len(beam_angele_11)):
        if beam_angele_11[i] <= theta_1dB[10]:
            center_traffic_beam11 += beam_traffic_11[i]
    for i in range(len(beam_angele_12)):
        if beam_angele_12[i] <= theta_1dB[11]:
            center_traffic_beam12 += beam_traffic_12[i]

    center_traffic = [center_traffic_beam1, center_traffic_beam2, center_traffic_beam3,
                      center_traffic_beam4, center_traffic_beam5, center_traffic_beam6,
                      center_traffic_beam7, center_traffic_beam8, center_traffic_beam9,
                      center_traffic_beam10, center_traffic_beam11, center_traffic_beam12]

    beam_traffic = [torch.sum(traffic[0:15]), torch.sum(traffic[15:30]), torch.sum(traffic[30:45]),
                    torch.sum(traffic[45:60]), torch.sum(traffic[60:75]), torch.sum(traffic[75:90]),
                    torch.sum(traffic[90:105]), torch.sum(traffic[105:120]), torch.sum(traffic[120:135]),
                    torch.sum(traffic[135:150]), torch.sum(traffic[150:165]), torch.sum(traffic[165:180])]

    ratio = [((beam_traffic[0] - center_traffic[0]) / beam_traffic[0]).item(),
             ((beam_traffic[1] - center_traffic[1]) / beam_traffic[1]).item(),
             ((beam_traffic[2] - center_traffic[2]) / beam_traffic[2]).item(),
             ((beam_traffic[3] - center_traffic[3]) / beam_traffic[3]).item(),
             ((beam_traffic[4] - center_traffic[4]) / beam_traffic[4]).item(),
             ((beam_traffic[5] - center_traffic[5]) / beam_traffic[5]).item(),
             ((beam_traffic[6] - center_traffic[6]) / beam_traffic[6]).item(),
             ((beam_traffic[7] - center_traffic[7]) / beam_traffic[7]).item(),
             ((beam_traffic[8] - center_traffic[8]) / beam_traffic[8]).item(),
             ((beam_traffic[9] - center_traffic[9]) / beam_traffic[9]).item(),
             ((beam_traffic[10] - center_traffic[10]) / beam_traffic[10]).item(),
             ((beam_traffic[11] - center_traffic[11]) / beam_traffic[11]).item()]
    for i in range(12):
        if beam_traffic[i] == 0:
            ratio[i] = 0
        if ratio[i] <= 0:
            ratio[i] = 0
    return ratio, theta_3dB, theta_1dB


def calculate_beam_asign_traffic(power_selection, gain_selection, ratio):
    receive_gain = 40  # dBi
    path_loss = -213.498  # dB
    Noise = -228.6 + 29 + 84  # dB
    beam_asign_traffic = []
    useful_power = power_selection + gain_selection + receive_gain + path_loss
    power_array = power_selection.t()
    ratio_array = ratio.t()
    gain_array = gain_selection.t()
    interf = []
    ratio_sum = torch.sum(ratio_array, dim=0)
    for i in range(BATCH_SIZE):
        if ratio_sum[i] <= 1:
            interf_power_beam = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        else:
            interf_power_beam = []
            for j in range(12):
                interf_power_array = power_array[torch.arange(power_array.size(0)) != j]
                interf_power_array = interf_power_array.t()
                interf_ratio_array = ratio_array[torch.arange(ratio_array.size(0)) != j]
                interf_ratio_array = interf_ratio_array.t()
                interf_gain_array = gain_array[torch.arange(gain_array.size(0)) != j]
                interf_gain_array = interf_gain_array.t()
                for k in range(interf_ratio_array.size(1)):
                    if interf_ratio_array[i][k] > ratio[i][j]:
                        interf_ratio_array[i][k] = ratio[i][j]
                interf_power = 0
                for k in range(interf_ratio_array.size(1)):
                    interf_power += interf_ratio_array[i][k] * torch.pow(10, (
                            interf_power_array[i][k] + interf_gain_array[i][k]) / 10)
                interf_power = 10 * torch.log10(interf_power)
                interf_power = interf_power + receive_gain + path_loss
                interf_power = torch.pow(10, interf_power / 10)
                interf_power_beam.append(interf_power)
        interf.append(interf_power_beam)
    interf = torch.tensor(interf, requires_grad=True)
    Noise = m.pow(10, Noise / 10)
    total_interf = interf + Noise
    for i in range(BATCH_SIZE):
        for j in range(12):
            total_interf[i][j] = 10 * m.log10(total_interf[i][j])
    SINR = useful_power - total_interf
    return SINR, interf


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss(reduce=True, size_average=True)

fig, ay = plt.subplots()
fig, az = plt.subplots()
fig, ac = plt.subplots()
fig, ap = plt.subplots()
fig, ag = plt.subplots()
fig, adB = plt.subplots()
fig, adb = plt.subplots()
fig, ai = plt.subplots()

xx, yy, zz, cc = [], [], [], []
p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = [], [], [], [], [], [], [], [], [], [], [], []
g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12 = [], [], [], [], [], [], [], [], [], [], [], []
dB1, dB2, dB3, dB4, dB5, dB6, dB7, dB8, dB9, dB10, dB11, dB12 = [], [], [], [], [], [], [], [], [], [], [], []
db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12 = [], [], [], [], [], [], [], [], [], [], [], []
i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12 = [], [], [], [], [], [], [], [], [], [], [], []

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 900)
        # 计算各个波束的业务量
        b_y = b_label.view(-1, 180, 5)
        traffic = b_y[:, :, 4]
        batch_beam_traffic = []
        for i in range(traffic.shape[0]):
            beam_traffic = []
            beam_traffic.append(torch.sum(traffic[i, 0:15]))
            beam_traffic.append(torch.sum(traffic[i, 15:30]))
            beam_traffic.append(torch.sum(traffic[i, 30:45]))
            beam_traffic.append(torch.sum(traffic[i, 45:60]))
            beam_traffic.append(torch.sum(traffic[i, 60:75]))
            beam_traffic.append(torch.sum(traffic[i, 75:90]))
            beam_traffic.append(torch.sum(traffic[i, 90:105]))
            beam_traffic.append(torch.sum(traffic[i, 105:120]))
            beam_traffic.append(torch.sum(traffic[i, 120:135]))
            beam_traffic.append(torch.sum(traffic[i, 135:150]))
            beam_traffic.append(torch.sum(traffic[i, 150:165]))
            beam_traffic.append(torch.sum(traffic[i, 165:180]))
            batch_beam_traffic.append(beam_traffic)
        batch_beam_traffic = torch.Tensor(batch_beam_traffic)
        power_distribution, power_softmax = net.run(b_x)
        gain_selection = power_distribution * 20 + 35
        power_selection1 = 68 - gain_selection - power_distribution * 30
        power_W = torch.pow(10, power_selection1 / 10)
        power_sum = torch.sum(power_W, dim=1)
        power_W1 = torch.zeros_like(power_W)
        for i in range(BATCH_SIZE):
            if power_sum[i] > 3000:
                power_W1[i] = power_W[i] / power_sum[i] * 3000
            else:
                power_W1[i] = power_W[i]
            if (batch_beam_traffic[i, :] > 0).all():
                continue
            else:
                index = torch.where(batch_beam_traffic[i, :] == 0)[0]
                power_W11 = power_W1[i].clone()
                for k in range(len(index)):
                    power_W11[index[k]] = 0
                power_W11SUM = torch.sum(power_W11)
                i_index = 0
                for n in range(len(index)):
                    i_index += power_W1[i][index[n]].item()
                power_W1[i] = power_W1[i] + i_index * power_W11 / power_W11SUM
                for mm in range(len(index)):
                    power_W1[i][index[mm]] = 0
        power_selection = 10 * torch.log10(power_W1)
        beam_ratio, beam_3dB, beam_1dB = [], [], []
        for i in range(BATCH_SIZE):
            ratio, threedB, onedB = calculate_interf_ration(power_selection[i, :], gain_selection[i, :], b_y[i, :, :], batch_beam_traffic[i, :])
            beam_ratio.append(ratio)
            beam_3dB.append(threedB)
            beam_1dB.append(onedB)
        beam_ratio = np.asarray(beam_ratio)
        beam_ratio = torch.Tensor(beam_ratio)
        beam_3dB = np.asarray(beam_3dB)
        beam_1dB = np.asarray(beam_1dB)
        beam_tra_allo, interf = calculate_beam_asign_traffic(power_selection, gain_selection, beam_ratio)
        bandwidth = 500e6  # Hz
        batch_beam_traffic = batch_beam_traffic.view(-1)
        beam_traffic_label = []
        for i in range(len(batch_beam_traffic)):
            if batch_beam_traffic[i] == 0:
                beam_traffic = 0
            else:
                beam_traffic = m.pow(2, batch_beam_traffic[i] * 1000 / bandwidth) - 1
                beam_traffic = 10 * m.log10(beam_traffic)
            beam_traffic_label.append(beam_traffic)
        beam_traffic_label = np.asarray(beam_traffic_label)
        beam_traffic_label = beam_traffic_label.reshape(BATCH_SIZE, -1)
        beam_traffic_label = torch.Tensor(beam_traffic_label)
        tral = torch.zeros((beam_traffic_label.size()[0], beam_traffic_label.size()[1]))
        for i in range(tral.size()[0]):
            for j in range(tral.size()[1]):
                if beam_traffic_label[i][j] == 0:
                    tral[i][j] = 1
        mask = tral.bool()
        beam_tra_allo = beam_tra_allo.masked_fill_(mask, 0)
        loss = loss_func(beam_tra_allo.float(), beam_traffic_label.float())  # mean square error
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # 计算准确率
        b_label_m = np.asarray(beam_traffic_label).reshape(-1)
        b_allo_m = beam_tra_allo.detach().numpy().reshape(-1)
        b_label = b_label_m.copy()
        b_allo = b_allo_m.copy()
        accuaracy = 0
        for i in range(b_label.size):
            if b_label[i] > (68 + 40 - 213.498) / (-228.6 - 29 - 84):
                b_label[i] = (68 + 40 - 213.498) / (-228.6 - 29 - 84)
            b_label[i] = m.pow(10, b_label[i] / 10)
            b_allo[i] = m.pow(10, b_allo[i] / 10)
            if b_label[i] == 0:
                acc = 1
            else:
                acc = min(b_allo[i], b_label[i]) / b_label[i]
            accuaracy += acc
        accuaracy = accuaracy / (12 * BATCH_SIZE)

        # 计算各个波束的发射功率
        power_selection = power_selection.detach().numpy()
        power_0 = power_1 = power_2 = power_3 = power_4 = power_5 = power_6 = power_7 = power_8 = power_9 = power_10 = power_11 = 0
        for i in range(BATCH_SIZE):
            power_0 += power_selection[i][0]
            power_1 += power_selection[i][1]
            power_2 += power_selection[i][2]
            power_3 += power_selection[i][3]
            power_4 += power_selection[i][4]
            power_5 += power_selection[i][5]
            power_6 += power_selection[i][6]
            power_7 += power_selection[i][7]
            power_8 += power_selection[i][8]
            power_9 += power_selection[i][9]
            power_10 += power_selection[i][10]
            power_11 += power_selection[i][11]
        power_beam0 = power_0 / BATCH_SIZE
        power_beam1 = power_1 / BATCH_SIZE
        power_beam2 = power_2 / BATCH_SIZE
        power_beam3 = power_3 / BATCH_SIZE
        power_beam4 = power_4 / BATCH_SIZE
        power_beam5 = power_5 / BATCH_SIZE
        power_beam6 = power_6 / BATCH_SIZE
        power_beam7 = power_7 / BATCH_SIZE
        power_beam8 = power_8 / BATCH_SIZE
        power_beam9 = power_9 / BATCH_SIZE
        power_beam10 = power_10 / BATCH_SIZE
        power_beam11 = power_11 / BATCH_SIZE

        # 计算各个波束的发射增益
        gain_selection = gain_selection.detach().numpy()
        print(gain_selection.shape)
        for i in range(gain_selection.shape[0]):
            for j in range(gain_selection.shape[1]):
                if beam_traffic_label[i][j] == 0:
                    gain_selection[i][j] = np.inf
        gain_0 = gain_1 = gain_2 = gain_3 = gain_4 = gain_5 = gain_6 = gain_7 = gain_8 = gain_9 = gain_10 = gain_11 = 0
        for i in range(BATCH_SIZE):
            gain_0 += gain_selection[i][0]
            gain_1 += gain_selection[i][1]
            gain_2 += gain_selection[i][2]
            gain_3 += gain_selection[i][3]
            gain_4 += gain_selection[i][4]
            gain_5 += gain_selection[i][5]
            gain_6 += gain_selection[i][6]
            gain_7 += gain_selection[i][7]
            gain_8 += gain_selection[i][8]
            gain_9 += gain_selection[i][9]
            gain_10 += gain_selection[i][10]
            gain_11 += gain_selection[i][11]
        gain_beam0 = gain_0 / BATCH_SIZE
        gain_beam1 = gain_1 / BATCH_SIZE
        gain_beam2 = gain_2 / BATCH_SIZE
        gain_beam3 = gain_3 / BATCH_SIZE
        gain_beam4 = gain_4 / BATCH_SIZE
        gain_beam5 = gain_5 / BATCH_SIZE
        gain_beam6 = gain_6 / BATCH_SIZE
        gain_beam7 = gain_7 / BATCH_SIZE
        gain_beam8 = gain_8 / BATCH_SIZE
        gain_beam9 = gain_9 / BATCH_SIZE
        gain_beam10 = gain_10 / BATCH_SIZE
        gain_beam11 = gain_11 / BATCH_SIZE

        # 计算各个波束的3dB角
        for i in range(beam_3dB.shape[0]):
            for j in range(beam_3dB.shape[1]):
                if beam_traffic_label[i][j] == 0:
                    beam_3dB[i][j] = 0
        threedB_0 = threedB_1 = threedB_2 = threedB_3 = threedB_4 = threedB_5 = threedB_6 = threedB_7 = threedB_8 = threedB_9 = threedB_10 = threedB_11 = 0
        for i in range(BATCH_SIZE):
            threedB_0 += beam_3dB[i][0]
            threedB_1 += beam_3dB[i][1]
            threedB_2 += beam_3dB[i][2]
            threedB_3 += beam_3dB[i][3]
            threedB_4 += beam_3dB[i][4]
            threedB_5 += beam_3dB[i][5]
            threedB_6 += beam_3dB[i][6]
            threedB_7 += beam_3dB[i][7]
            threedB_8 += beam_3dB[i][8]
            threedB_9 += beam_3dB[i][9]
            threedB_10 += beam_3dB[i][10]
            threedB_11 += beam_3dB[i][11]
        threedB_beam0 = threedB_0 / BATCH_SIZE
        threedB_beam1 = threedB_1 / BATCH_SIZE
        threedB_beam2 = threedB_2 / BATCH_SIZE
        threedB_beam3 = threedB_3 / BATCH_SIZE
        threedB_beam4 = threedB_4 / BATCH_SIZE
        threedB_beam5 = threedB_5 / BATCH_SIZE
        threedB_beam6 = threedB_6 / BATCH_SIZE
        threedB_beam7 = threedB_7 / BATCH_SIZE
        threedB_beam8 = threedB_8 / BATCH_SIZE
        threedB_beam9 = threedB_9 / BATCH_SIZE
        threedB_beam10 = threedB_10 / BATCH_SIZE
        threedB_beam11 = threedB_11 / BATCH_SIZE

        # 计算各个波束的1dB角
        for i in range(beam_1dB.shape[0]):
            for j in range(beam_1dB.shape[1]):
                if beam_traffic_label[i][j] == 0:
                    beam_1dB[i][j] = 0
        onedB_0 = onedB_1 = onedB_2 = onedB_3 = onedB_4 = onedB_5 = onedB_6 = onedB_7 = onedB_8 = onedB_9 = onedB_10 = onedB_11 = 0
        for i in range(BATCH_SIZE):
            onedB_0 += beam_1dB[i][0]
            onedB_1 += beam_1dB[i][1]
            onedB_2 += beam_1dB[i][2]
            onedB_3 += beam_1dB[i][3]
            onedB_4 += beam_1dB[i][4]
            onedB_5 += beam_1dB[i][5]
            onedB_6 += beam_1dB[i][6]
            onedB_7 += beam_1dB[i][7]
            onedB_8 += beam_1dB[i][8]
            onedB_9 += beam_1dB[i][9]
            onedB_10 += beam_1dB[i][10]
            onedB_11 += beam_1dB[i][11]
        onedB_beam0 = onedB_0 / BATCH_SIZE
        onedB_beam1 = onedB_1 / BATCH_SIZE
        onedB_beam2 = onedB_2 / BATCH_SIZE
        onedB_beam3 = onedB_3 / BATCH_SIZE
        onedB_beam4 = onedB_4 / BATCH_SIZE
        onedB_beam5 = onedB_5 / BATCH_SIZE
        onedB_beam6 = onedB_6 / BATCH_SIZE
        onedB_beam7 = onedB_7 / BATCH_SIZE
        onedB_beam8 = onedB_8 / BATCH_SIZE
        onedB_beam9 = onedB_9 / BATCH_SIZE
        onedB_beam10 = onedB_10 / BATCH_SIZE
        onedB_beam11 = onedB_11 / BATCH_SIZE

        # 计算各个波束的干扰
        interf = beam_tra_allo.detach().numpy()
        interf_0 = interf_1 = interf_2 = interf_3 = interf_4 = interf_5 = interf_6 = interf_7 = interf_8 = interf_9 = interf_10 = interf_11 = 0
        for i in range(BATCH_SIZE):
            interf_0 += interf[i][0]
            interf_1 += interf[i][1]
            interf_2 += interf[i][2]
            interf_3 += interf[i][3]
            interf_4 += interf[i][4]
            interf_5 += interf[i][5]
            interf_6 += interf[i][6]
            interf_7 += interf[i][7]
            interf_8 += interf[i][8]
            interf_9 += interf[i][9]
            interf_10 += interf[i][10]
            interf_11 += interf[i][11]
        interf_beam0 = interf_0 / BATCH_SIZE
        interf_beam1 = interf_1 / BATCH_SIZE
        interf_beam2 = interf_2 / BATCH_SIZE
        interf_beam3 = interf_3 / BATCH_SIZE
        interf_beam4 = interf_4 / BATCH_SIZE
        interf_beam5 = interf_5 / BATCH_SIZE
        interf_beam6 = interf_6 / BATCH_SIZE
        interf_beam7 = interf_7 / BATCH_SIZE
        interf_beam8 = interf_8 / BATCH_SIZE
        interf_beam9 = interf_9 / BATCH_SIZE
        interf_beam10 = interf_10 / BATCH_SIZE
        interf_beam11 = interf_11 / BATCH_SIZE

        # 计算系统的总干扰
        t_interf = 0
        for i in range(BATCH_SIZE):
            for j in range(12):
                t_interf += beam_tra_allo[i][j]
        system_interf = t_interf / (BATCH_SIZE) / 12

    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

    print('Epoch: ', epoch, '| accuaracy: %.4f' % accuaracy)

    print('Epoch: ', epoch, '| power_0: %.4f' % power_beam0)
    print('Epoch: ', epoch, '| power_1: %.4f' % power_beam1)
    print('Epoch: ', epoch, '| power_2: %.4f' % power_beam2)
    print('Epoch: ', epoch, '| power_3: %.4f' % power_beam3)
    print('Epoch: ', epoch, '| power_4: %.4f' % power_beam4)
    print('Epoch: ', epoch, '| power_5: %.4f' % power_beam5)
    print('Epoch: ', epoch, '| power_6: %.4f' % power_beam6)
    print('Epoch: ', epoch, '| power_7: %.4f' % power_beam7)
    print('Epoch: ', epoch, '| power_8: %.4f' % power_beam8)
    print('Epoch: ', epoch, '| power_9: %.4f' % power_beam9)
    print('Epoch: ', epoch, '| power_10: %.4f' % power_beam10)
    print('Epoch: ', epoch, '| power_11: %.4f' % power_beam11)

    print('Epoch: ', epoch, '| gain_0: %.4f' % gain_beam0)
    print('Epoch: ', epoch, '| gain_1: %.4f' % gain_beam1)
    print('Epoch: ', epoch, '| gain_2: %.4f' % gain_beam2)
    print('Epoch: ', epoch, '| gain_3: %.4f' % gain_beam3)
    print('Epoch: ', epoch, '| gain_4: %.4f' % gain_beam4)
    print('Epoch: ', epoch, '| gain_5: %.4f' % gain_beam5)
    print('Epoch: ', epoch, '| gain_6: %.4f' % gain_beam6)
    print('Epoch: ', epoch, '| gain_7: %.4f' % gain_beam7)
    print('Epoch: ', epoch, '| gain_8: %.4f' % gain_beam8)
    print('Epoch: ', epoch, '| gain_9: %.4f' % gain_beam9)
    print('Epoch: ', epoch, '| gain_10: %.4f' % gain_beam10)
    print('Epoch: ', epoch, '| gain_11: %.4f' % gain_beam11)

    print('Epoch: ', epoch, '| threedB_0: %.4f' % threedB_beam0)
    print('Epoch: ', epoch, '| threedB_1: %.4f' % threedB_beam1)
    print('Epoch: ', epoch, '| threedB_2: %.4f' % threedB_beam2)
    print('Epoch: ', epoch, '| threedB_3: %.4f' % threedB_beam3)
    print('Epoch: ', epoch, '| threedB_4: %.4f' % threedB_beam4)
    print('Epoch: ', epoch, '| threedB_5: %.4f' % threedB_beam5)
    print('Epoch: ', epoch, '| threedB_6: %.4f' % threedB_beam6)
    print('Epoch: ', epoch, '| threedB_7: %.4f' % threedB_beam7)
    print('Epoch: ', epoch, '| threedB_8: %.4f' % threedB_beam8)
    print('Epoch: ', epoch, '| threedB_9: %.4f' % threedB_beam9)
    print('Epoch: ', epoch, '| threedB_10: %.4f' % threedB_beam10)
    print('Epoch: ', epoch, '| threedB_11: %.4f' % threedB_beam11)

    print('Epoch: ', epoch, '| onedB_0: %.4f' % onedB_beam0)
    print('Epoch: ', epoch, '| onedB_1: %.4f' % onedB_beam1)
    print('Epoch: ', epoch, '| onedB_2: %.4f' % onedB_beam2)
    print('Epoch: ', epoch, '| onedB_3: %.4f' % onedB_beam3)
    print('Epoch: ', epoch, '| onedB_4: %.4f' % onedB_beam4)
    print('Epoch: ', epoch, '| onedB_5: %.4f' % onedB_beam5)
    print('Epoch: ', epoch, '| onedB_6: %.4f' % onedB_beam6)
    print('Epoch: ', epoch, '| onedB_7: %.4f' % onedB_beam7)
    print('Epoch: ', epoch, '| onedB_8: %.4f' % onedB_beam8)
    print('Epoch: ', epoch, '| onedB_9: %.4f' % onedB_beam9)
    print('Epoch: ', epoch, '| onedB_10: %.4f' % onedB_beam10)
    print('Epoch: ', epoch, '| onedB_11: %.4f' % onedB_beam11)

    print('Epoch: ', epoch, '| interf_0: %.4f' % interf_beam0)
    print('Epoch: ', epoch, '| interf_1: %.4f' % interf_beam1)
    print('Epoch: ', epoch, '| interf_2: %.4f' % interf_beam2)
    print('Epoch: ', epoch, '| interf_3: %.4f' % interf_beam3)
    print('Epoch: ', epoch, '| interf_4: %.4f' % interf_beam4)
    print('Epoch: ', epoch, '| interf_5: %.4f' % interf_beam5)
    print('Epoch: ', epoch, '| interf_6: %.4f' % interf_beam6)
    print('Epoch: ', epoch, '| interf_7: %.4f' % interf_beam7)
    print('Epoch: ', epoch, '| interf_8: %.4f' % interf_beam8)
    print('Epoch: ', epoch, '| interf_9: %.4f' % interf_beam9)
    print('Epoch: ', epoch, '| interf_10: %.4f' % interf_beam10)
    print('Epoch: ', epoch, '| interf_11: %.4f' % interf_beam11)

    print('Epoch: ', epoch, '| interf: %.4f' % system_interf)

    # 保存
    torch.save(Net, './model.pt')
    xx.append(epoch + 1)
    yy.append(loss)
    zz.append(accuaracy)
    cc.append(system_interf)

    p1.append(power_beam0)
    p2.append(power_beam1)
    p3.append(power_beam2)
    p4.append(power_beam3)
    p5.append(power_beam4)
    p6.append(power_beam5)
    p7.append(power_beam6)
    p8.append(power_beam7)
    p9.append(power_beam8)
    p10.append(power_beam9)
    p11.append(power_beam10)
    p12.append(power_beam11)

    g1.append(gain_beam0)
    g2.append(gain_beam1)
    g3.append(gain_beam2)
    g4.append(gain_beam3)
    g5.append(gain_beam4)
    g6.append(gain_beam5)
    g7.append(gain_beam6)
    g8.append(gain_beam7)
    g9.append(gain_beam8)
    g10.append(gain_beam9)
    g11.append(gain_beam10)
    g12.append(gain_beam11)

    dB1.append(threedB_beam0)
    dB2.append(threedB_beam1)
    dB3.append(threedB_beam2)
    dB4.append(threedB_beam3)
    dB5.append(threedB_beam4)
    dB6.append(threedB_beam5)
    dB7.append(threedB_beam6)
    dB8.append(threedB_beam7)
    dB9.append(threedB_beam8)
    dB10.append(threedB_beam9)
    dB11.append(threedB_beam10)
    dB12.append(threedB_beam11)

    db1.append(onedB_beam0)
    db2.append(onedB_beam1)
    db3.append(onedB_beam2)
    db4.append(onedB_beam3)
    db5.append(onedB_beam4)
    db6.append(onedB_beam5)
    db7.append(onedB_beam6)
    db8.append(onedB_beam7)
    db9.append(onedB_beam8)
    db10.append(onedB_beam9)
    db11.append(onedB_beam10)
    db12.append(onedB_beam11)

    i1.append(interf_beam0)
    i2.append(interf_beam1)
    i3.append(interf_beam2)
    i4.append(interf_beam3)
    i5.append(interf_beam4)
    i6.append(interf_beam5)
    i7.append(interf_beam6)
    i8.append(interf_beam7)
    i9.append(interf_beam8)
    i10.append(interf_beam9)
    i11.append(interf_beam10)
    i12.append(interf_beam11)

    ay.cla()  # clear plot
    ay.plot(xx, yy, 'b', lw=1)  # draw line chart
    ay.set(xlabel='epoch', ylabel='training loss')

    az.cla()  # clear plot
    az.plot(xx, zz, 'b', lw=1)
    az.set(xlabel='epoch', ylabel='training accuracy')

    ac.cla()  # clear plot
    ac.plot(xx, cc, 'b', lw=1)
    ac.set(xlabel='epoch', ylabel='SINR of the system')

    ap.cla()
    ap.plot(xx, p1, 'r-', lw=1)
    ap.plot(xx, p2, 'y-', lw=1)
    ap.plot(xx, p3, 'g-', lw=1)
    ap.plot(xx, p4, 'k-', lw=1)
    ap.plot(xx, p5, 'b-', lw=1)
    ap.plot(xx, p6, 'm-', lw=1)
    ap.plot(xx, p7, 'c-', lw=1)
    ap.plot(xx, p8, 'darkolivegreen', lw=1)
    ap.plot(xx, p9, 'blueviolet', lw=1)
    ap.plot(xx, p10, 'lightseagreen', lw=1)
    ap.plot(xx, p11, 'lime', lw=1)
    ap.plot(xx, p12, 'deeppink', lw=1)
    ap.set(xlabel='epoch', ylabel='transmit power')
    ap.legend(('Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8', 'Beam9', 'Beam10', 'Beam11', 'Beam12'), loc='best')

    ag.cla()
    ag.plot(xx, g1, 'r-', lw=1)
    ag.plot(xx, g2, 'y-', lw=1)
    ag.plot(xx, g3, 'g-', lw=1)
    ag.plot(xx, g4, 'k-', lw=1)
    ag.plot(xx, g5, 'b-', lw=1)
    ag.plot(xx, g6, 'm-', lw=1)
    ag.plot(xx, g7, 'c-', lw=1)
    ag.plot(xx, g8, 'darkolivegreen', lw=1)
    ag.plot(xx, g9, 'blueviolet', lw=1)
    ag.plot(xx, g10, 'lightseagreen', lw=1)
    ag.plot(xx, g11, 'lime', lw=1)
    ag.plot(xx, g12, 'deeppink', lw=1)
    ag.set(xlabel='epoch', ylabel='transmit gain')
    ag.legend(('Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8', 'Beam9', 'Beam10', 'Beam11', 'Beam12'), loc='best')

    adB.cla()
    adB.plot(xx, dB1, 'r-', lw=1)
    adB.plot(xx, dB2, 'y-', lw=1)
    adB.plot(xx, dB3, 'g-', lw=1)
    adB.plot(xx, dB4, 'k-', lw=1)
    adB.plot(xx, dB5, 'b-', lw=1)
    adB.plot(xx, dB6, 'm-', lw=1)
    adB.plot(xx, dB7, 'c-', lw=1)
    adB.plot(xx, dB8, 'darkolivegreen', lw=1)
    adB.plot(xx, dB9, 'blueviolet', lw=1)
    adB.plot(xx, dB10, 'lightseagreen', lw=1)
    adB.plot(xx, dB11, 'lime', lw=1)
    adB.plot(xx, dB12, 'deeppink', lw=1)
    adB.set(xlabel='epoch', ylabel='3dB beam width')
    adB.legend(('Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8', 'Beam9', 'Beam10', 'Beam11', 'Beam12'), loc='best')

    adb.cla()
    adb.plot(xx, db1, 'r-', lw=1)
    adb.plot(xx, db2, 'y-', lw=1)
    adb.plot(xx, db3, 'g-', lw=1)
    adb.plot(xx, db4, 'k-', lw=1)
    adb.plot(xx, db5, 'b-', lw=1)
    adb.plot(xx, db6, 'm-', lw=1)
    adb.plot(xx, db7, 'c-', lw=1)
    adb.plot(xx, db8, 'darkolivegreen', lw=1)
    adb.plot(xx, db9, 'blueviolet', lw=1)
    adb.plot(xx, db10, 'lightseagreen', lw=1)
    adb.plot(xx, db11, 'lime', lw=1)
    adb.plot(xx, db12, 'deeppink', lw=1)
    adb.set(xlabel='epoch', ylabel='1dB beam width')
    adb.legend(('Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8', 'Beam9', 'Beam10', 'Beam11', 'Beam12'), loc='best')

    ai.cla()
    ai.plot(xx, i1, 'r-', lw=1)
    ai.plot(xx, i2, 'y-', lw=1)
    ai.plot(xx, i3, 'g-', lw=1)
    ai.plot(xx, i4, 'k-', lw=1)
    ai.plot(xx, i5, 'b-', lw=1)
    ai.plot(xx, i6, 'm-', lw=1)
    ai.plot(xx, i7, 'c-', lw=1)
    ai.plot(xx, i8, 'darkolivegreen', lw=1)
    ai.plot(xx, i9, 'blueviolet', lw=1)
    ai.plot(xx, i10, 'lightseagreen', lw=1)
    ai.plot(xx, i11, 'lime', lw=1)
    ai.plot(xx, i12, 'deeppink', lw=1)
    ai.set(xlabel='epoch', ylabel='SINR of the beam')
    ai.legend(('Beam1', 'Beam2', 'Beam3', 'Beam4', 'Beam5', 'Beam6', 'Beam7', 'Beam8', 'Beam9', 'Beam10', 'Beam11', 'Beam12'), loc='best')

    plt.pause(0.1)
    plt.savefig('./' + str(localtime))

plt.ioff()
plt.show()
