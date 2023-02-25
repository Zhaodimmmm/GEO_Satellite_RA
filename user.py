import numpy as np
import pandas as pd
import math as m
import random


class user:
    def __init__(self, maxdistfromorigin, lat_long_areacenter, cbrrate, ontime=10, offtime=2):
        """
        :param maxdistfromorigin: 距离中心点的最大距离单位米
        :param ontime: 单位ms,指数分布的均值
        :param offtime: 单位ms,指数分布的均值
        """
        self.maxdistance = maxdistfromorigin  # 距离波束中心的最大距离
        self.position = np.array([0, 0, 0], dtype="float64")  # 以地心为原点建立的笛卡尔坐标系
        self.lat_long_areacenter = lat_long_areacenter
        self.log_lat_coordinates = np.array([0, 0, 0], dtype="float64")  # 用户参数，共三个参数，分别代表纬度，经度，距离地表的海拔高度
        self.nextjingweiposision = np.array([0, 0, 0], dtype="float64")  # 更新后的经纬度坐标
        # 业务信息
        self.throughput = 0  # Mbps
        self.request = 0  # 0表示无请求 1表示有请求
        self.ontime = 0  # 业务持续时间
        self.offtime_restore = offtime
        self.offtime = np.random.exponential(offtime)
        self.traffictype = {'text': ontime/4, 'voice': ontime/2, 'video': ontime}  # 业务类型指数分布均值参数
        self.qci_type = {'None': 0, 'text': 1, 'voice': 2, 'video': 3}
        self.qci = 0
        self.waiting_data_finally = 0  # 采取动作后剩余数据
        self.waitingbit = 0  # 当前时刻剩余待传数据
        self.cbrrate = cbrrate  # 单位bit 每毫秒
        self.transmit_rate_one_channal = 10
        self.newarrivaldata = 0  # 新到数据
        self.current_txdata = 0
        self.total_txdata = 0
        self.type = None  # 最终生成的业务类型
        self.number_of_rbg_nedded = 0
        self.max_number_of_rbg = 12
        self.current_waiting_data = 0
        self.index = 0
        self.average_throughput = 0.000001  # 1bytes
        self.capacity = 0
        # 随机位置
        self.movespeed = 30  # 用户移动速度 米每秒
        self.earth_radius = 6371000  # 地球半径
        self.earth_semimajor_axis = 6378137   # 由GRS80和WGS84定义的以米为单位的地球半长轴
        self.earth_grs80_eccentricity = 0.0818191910428158   # GRS80定义的地球第一偏心率
        self.earth_wgs84_eccentricity = 0.0818191908426215   # WGS84定义的地球第一偏心率
        self.earthspheroidtype = {'sphere': 0, 'grs80': 1, 'wgs84': 2}  # 三种地球模型
        self.initial_random_position(self.lat_long_areacenter)  # 初始化用户位置函数
        self.movedistance = 0  # 每次更新用户移动距离
        self.randomangle = self.random_angle()  # 产生基于用户速度的经纬度的变化角度
        self.time_delay = 0   # 时延
        self.height = 35786000
        self.angle_user = 0
        self.capacity = 0
        self.generate_angele_user(self.lat_long_areacenter, self.position, self.height)

    # 产生基于用户速度的经纬度的变化角度
    def random_angle(self):
        direction = np.cos(np.random.uniform(0, m.pi, size=3))
        speed = self.movespeed
        randomangle = speed * direction
        randomangle = (randomangle / (2 * np.pi * self.earth_radius)) * 360
        zaxischangerate = np.random.uniform(-1, 1)
        randomangle[2] = 0
        return randomangle

    # 更新模型1，每次更新选择随机的移动方向和行进速率
    def model1_update(self, tb, bler, cqi, time_duration=0.001):
        self.randomangle = self.random_angle()
        currentpositionxyz = self.position
        self.log_lat_coordinates[0] += self.randomangle[0] * time_duration
        self.log_lat_coordinates[1] += self.randomangle[1] * time_duration
        self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
        if self.log_lat_coordinates[2] < 0:
            self.log_lat_coordinates[2] = 0
        userxyz_afterupdate = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                   self.log_lat_coordinates[1],
                                                                   self.log_lat_coordinates[2],
                                                                   self.earthspheroidtype['sphere'])
        areacenterxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                             self.lat_long_areacenter[2],
                                                             self.earthspheroidtype['sphere'])
        user_beamcenter_distance = np.sum(np.square(userxyz_afterupdate - areacenterxyz)) ** 0.5
        if user_beamcenter_distance >= self.maxdistance:
            self.randomangle = -self.randomangle
            self.log_lat_coordinates[0] += self.randomangle[0] * time_duration
            self.log_lat_coordinates[1] += self.randomangle[1] * time_duration
            self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
            if self.log_lat_coordinates[2] < 0:
                self.log_lat_coordinates[2] = 0
            self.position = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                 self.log_lat_coordinates[1],
                                                                 self.log_lat_coordinates[2],
                                                                 self.earthspheroidtype['sphere'])
        self.position = userxyz_afterupdate
        updatepositionxyz = self.position
        self.movedistance = np.sum(np.square(updatepositionxyz - currentpositionxyz)) ** 0.5
        self.traffic_updata(tb, bler, cqi)

    # 更新模型2 按照恒定的行进速率和方向前进，直到到达边界，然后重新调用random_angle函数产生随机方向和距离
    def model2_update(self, tb, capacity, time_duration=0.001):
        currentpositionxyz = self.position
        self.log_lat_coordinates[0] = self.log_lat_coordinates[0] + self.randomangle[0] * time_duration
        self.log_lat_coordinates[1] = self.log_lat_coordinates[1] + self.randomangle[1] * time_duration
        self.log_lat_coordinates[2] += self.randomangle[2] * time_duration
        if self.log_lat_coordinates[2] < 0:
            self.log_lat_coordinates[2] = 0
        userxyz_afterupdate = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                   self.log_lat_coordinates[1],
                                                                   self.log_lat_coordinates[2],
                                                                   self.earthspheroidtype['sphere'])
        areacenterxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                             self.lat_long_areacenter[2],
                                                             self.earthspheroidtype['sphere'])
        user_beamcenter_distance = np.sum(np.square(userxyz_afterupdate - areacenterxyz)) ** 0.5
        if user_beamcenter_distance <= self.maxdistance:
            self.position = userxyz_afterupdate
        else:
            while (True):
                self.randomangle = self.random_angle()
                self.log_lat_coordinates[0] = self.log_lat_coordinates[0] + self.randomangle[0] * time_duration
                self.log_lat_coordinates[1] = self.log_lat_coordinates[1] + self.randomangle[1] * time_duration
                self.log_lat_coordinates[2] = self.log_lat_coordinates[2] + self.randomangle[2] * time_duration
                if self.log_lat_coordinates[2] < 0:
                    self.log_lat_coordinates[2] = 0
                userxyz_afterupdate2 = self.GeographicTocartesianCoordinate(self.log_lat_coordinates[0],
                                                                            self.log_lat_coordinates[1],
                                                                            self.log_lat_coordinates[2],
                                                                            self.earthspheroidtype['sphere'])
                user_areacenter_distance2 = np.sum(np.square(userxyz_afterupdate2 - areacenterxyz)) ** 0.5
                if user_areacenter_distance2 <= self.maxdistance:
                    self.position = userxyz_afterupdate2
                    break
        updatepositionxyz = self.position
        self.movedistance = np.sum(np.square(updatepositionxyz - currentpositionxyz)) ** 0.5
        self.traffic_updata(tb, capacity)

    # 随机选择三种业务并按照指数分布随机产生业务的持续时间
    def trafficduration(self):
        type = 'None'
        if self.offtime > 0:
            self.offtime -= 1
            if self.offtime < 0:
                self.offtime = 0
                ################
                traffic_choice = np.random.choice([1, 2, 3])
                if traffic_choice == 1:
                    self.ontime = np.random.exponential(self.traffictype['text'])
                    type = 'text'
                    self.qci = self.qci_type[type]
                elif traffic_choice == 2:
                    self.ontime = np.random.exponential(self.traffictype['voice'])
                    type = 'voice'
                    self.qci = self.qci_type[type]
                else:
                    self.ontime = np.random.exponential(self.traffictype['video'])
                    type = 'video'
                    self.qci = self.qci_type[type]
        elif self.offtime == 0 and self.ontime > 0:
            self.ontime -= 1
            if self.ontime < 0:
                self.ontime = 0
                self.offtime = np.random.exponential(self.offtime_restore)
                self.qci = 0
        return self.ontime

    def RbgCountRequired(self, bytes):
        data = 100000
        rbg_need = m.ceil(bytes / data)
        return rbg_need

    def waiting_data(self, tb, capacity):
        self.waiting_data_finally = self.waitingbit + self.newarrivaldata - tb
        if self.request == 1 and self.waiting_data_finally >= 0:
            self.current_txdata = tb
        elif self.request == 1 and self.waiting_data_finally < 0:
            self.current_txdata = self.waitingbit + self.newarrivaldata
        else:
            self.current_txdata = 0
        if self.current_txdata == 0 and self.request == 1:
            self.time_delay += 1
        if self.waiting_data_finally < 0:
            self.waiting_data_finally = 0
        self.throughput = ((self.current_txdata/0.001))/(1024**2)
        self.waitingbit = self.waiting_data_finally
        self.newarrivaldata = self.cbrrate * 1 if self.ontime > 1 else self.cbrrate * self.ontime
        self.current_waiting_data = self.waitingbit+self.newarrivaldata
        current_data_total = self.waitingbit + self.newarrivaldata
        if current_data_total > 0:
            self.request = 1
        else:
            self.request = 0
        self.capacity = capacity
        self.number_of_rbg_nedded = self.RbgCountRequired(current_data_total)   # 判断传输完等待数据所需资源块数目
        self.index += 1
        self.total_txdata = self.total_txdata+self.current_txdata
        self.average_throughput = (self.total_txdata / (self.index / 1000)) / 1024 ** 2

    def traffic_updata(self, tb, capacity):
        self.trafficduration()
        self.waiting_data(tb, capacity)

    # 以波束中心点为中心，随机在self.maxdistance范围内产生经纬度坐标，即用户初始位置
    def initial_random_position(self, beampara):
        originlatitude = beampara[0]
        originlongitude = beampara[1]
        maxaltitude = beampara[2]
        # 除去南北极
        if originlatitude >= 90:
            originlatitude = 89.999
        elif originlatitude <= -90:
            originlatitude = -89.999
        if maxaltitude < 0:
            maxaltitude = 0
        originlatituderadians = originlatitude * (np.pi / 180)
        originlongituderadians = originlongitude * (np.pi / 180)
        origincolatitude = (np.pi / 2) - originlatituderadians
        # 圆心角弧度数的最大值
        a = 0.99 * self.maxdistance / self.earth_radius
        if a > np.pi:
            a = np.pi
        d = np.random.uniform(0, self.earth_radius - self.earth_radius * np.cos(a))
        phi = np.random.uniform(0, np.pi * 2)
        alpha = m.acos((self.earth_radius - d) / self.earth_radius)
        theta = np.pi / 2 - alpha
        randpointlatitude = m.asin(
            m.sin(theta) * m.cos(origincolatitude) + m.cos(theta) * m.sin(origincolatitude) * m.sin(phi))
        intermedlong = m.asin((m.sin(randpointlatitude) * m.cos(origincolatitude) - m.sin(theta)) / (
                m.cos(randpointlatitude) * m.sin(origincolatitude)))
        intermedlong = intermedlong + np.pi / 2
        if phi > (np.pi / 2) and phi <= ((3 * np.pi) / 2):
            intermedlong = -intermedlong
        randpointlongtude = intermedlong + originlongituderadians
        randaltitude = np.random.uniform(0, maxaltitude)

        self.position = self.GeographicTocartesianCoordinate(randpointlatitude * (180 / np.pi),
                                                             randpointlongtude * (180 / np.pi), randaltitude,
                                                             self.earthspheroidtype['sphere'])
        self.log_lat_coordinates = [randpointlatitude * (180 / np.pi), randpointlongtude * (180 / np.pi),
                                    randaltitude]  # 度数为单位
        return self.position, self.log_lat_coordinates

    def generate_angele_user(self, beam_center, user_position, height):
        beam_position = self.GeographicTocartesianCoordinate(beam_center[0], beam_center[1], beam_center[2], self.earthspheroidtype['sphere'])
        beam2user = m.sqrt((beam_position[0]-user_position[0])**2+(beam_position[1]-user_position[1])**2
                           +(beam_position[2]-user_position[2])**2)
        angle_user = m.atan(beam2user/height)
        self.angle_user = m.degrees(angle_user)
        return self.angle_user

    # 将经纬度坐标转换为笛卡尔坐标系
    def GeographicTocartesianCoordinate(self, latitude, longitude, altitude, sphType):
        latitudeRadians = latitude * m.pi / 180
        longitudeRadians = longitude * m.pi / 180
        # a: semi - major axis of earth
        # e: first eccentricity of earth
        EARTH_RADIUS = 6371e3
        EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
        EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
        EARTH_SEMIMAJOR_AXIS = 6378137
        EARTH_SEMIMAJOR_BXIS = 6356752.3142451793
        if sphType == "SPHERE":
            a = EARTH_RADIUS
            e = 0
        if sphType == "GRS80":
            a = EARTH_SEMIMAJOR_AXIS
            e = EARTH_GRS80_ECCENTRICITY
        else:  # if sphType == WGS84
            a = EARTH_SEMIMAJOR_AXIS
            e = EARTH_WGS84_ECCENTRICITY
        Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))  # radius of  curvature
        x = (Rn + altitude) * m.cos(latitudeRadians) * m.cos(longitudeRadians)
        y = (Rn + altitude) * m.cos(latitudeRadians) * m.sin(longitudeRadians)
        z = (Rn + altitude) * m.sin(latitudeRadians)
        cartesianCoordinates = np.array([x, y, z], dtype='float64')
        return cartesianCoordinates

    def get_distance(self):
        areaxyz = self.GeographicTocartesianCoordinate(self.lat_long_areacenter[0], self.lat_long_areacenter[1],
                                                       self.lat_long_areacenter[2], self.earthspheroidtype['sphere'])
        distance = (np.sum(np.square(self.position - areaxyz))) ** 0.5
        return distance


def updata(user, tb, last_time_request, capacity):
    user_list = user
    tb_list = np.zeros(len(user_list))
    capacity_list = np.zeros(len(user_list))
    for i in range(len(last_time_request)):
        tb_list[last_time_request[i]] = tb[i]
        capacity_list[last_time_request[i]] = capacity[i]
    for i in range(len(user_list)):
        user_list[i].model2_update(tb_list[i], capacity_list[i])
    user_position_xyz, user_position_log_lat_coordinates = get_user_position(user_list)
    traffic_info, user_request = get_user_traffic_info(user_list)
    return user_position_xyz, user_position_log_lat_coordinates, traffic_info, user_request


def initial_all_user(maxdistance, numofuser, ontime=10, offtime=2):
    userlist1 = [user(maxdistance, np.array([-2.25821726e+00,  3.89018562e+00,  0.00000000e+00], dtype='float64'),
                      random.randint(1000000, 1100000), ontime, offtime) for i in range(15)]
    userlist2 = [user(maxdistance, np.array([-2.25821726e+00, -3.89018562e+00,  0.00000000e+00], dtype='float64'),
                      random.randint(800000, 900000), ontime, offtime) for i in range(15)]
    userlist3 = [user(maxdistance, np.array([4.51418857e+00, -1.10053519e-15,  0.00000000e+00], dtype='float64'),
                      random.randint(900000, 1000000), ontime, offtime) for i in range(15)]
    userlist4 = [user(maxdistance, np.array([1.03405998e+01,  8.74798540e+00,  9.31322575e-10], dtype='float64'),
                      random.randint(1000000, 1100000), ontime, offtime) for i in range(15)]
    userlist5 = [user(maxdistance, np.array([2.33434799e+00,  1.32767170e+01,  0.00000000e+00], dtype='float64'),
                      random.randint(1100000, 1200000), ontime, offtime) for i in range(15)]
    userlist6 = [user(maxdistance, np.array([-6.73239677e+00,  1.17186367e+01,  9.31322575e-10], dtype='float64'),
                      random.randint(500000, 600000), ontime, offtime) for i in range(15)]
    userlist7 = [user(maxdistance, np.array([-1.27130928e+01,  4.67862337e+00, -9.31322575e-10], dtype='float64'),
                      random.randint(1300000, 1400000), ontime, offtime) for i in range(15)]
    userlist8 = [user(maxdistance, np.array([-1.27130928e+01, -4.67862337e+00, -9.31322575e-10], dtype='float64'),
                      random.randint(900000, 1000000), ontime, offtime) for i in range(15)]
    userlist9 = [user(maxdistance, np.array([-6.73239677e+00, -1.17186367e+01,  9.31322575e-10], dtype='float64'),
                      random.randint(600000, 700000), ontime, offtime) for i in range(15)]
    userlist10 = [user(maxdistance, np.array([2.33434799e+00, -1.32767170e+01,  0.00000000e+00], dtype='float64'),
                       random.randint(1200000, 1300000), ontime, offtime) for i in range(15)]
    userlist11 = [user(maxdistance, np.array([1.03405998e+01, -8.74798540e+00,  0.00000000e+00], dtype='float64'),
                       random.randint(1400000, 1500000), ontime, offtime) for i in range(15)]
    userlist12 = [user(maxdistance, np.array([1.35410837e+01, -3.35733680e-15,  0.00000000e+00], dtype='float64'),
                       random.randint(700000, 800000), ontime, offtime) for i in range(15)]
    userlist = userlist1 + userlist2 + userlist3 + userlist4 + userlist5 + userlist6 + userlist7 + userlist8 + userlist9 + userlist10 + userlist11 + userlist12
    return userlist


# 获取发起业务请求用户的位置和编号
def get_user_position(user):
    # 初始化用户和泊松分布均值
    userlist = user
    user_position_XYZ = []
    user_position_log_lat_coordinates = []
    # 随机选择len(index)个用户来产生业务，len(index)服从泊松分布
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            position = userlist[i].position
            position2 = userlist[i].log_lat_coordinates
            user_position_XYZ.append(position)  # 只保留位置信息
            user_position_log_lat_coordinates.append(position2)
    return user_position_XYZ, user_position_log_lat_coordinates


def get_user_log_lat_coordinates(user):
    userlist = user
    user_position_list=[]
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            position=userlist[i].log_lat_coordinates
            user_position_list.append(position)
    return user_position_list


def get_user_traffic_info(user):
    userlist = user
    user_request = []
    traffic_info = []
    for i in range(len(userlist)):
        if userlist[i].request == 1:
            user_request.append(i)
        traffic_info.append(
            (i, userlist[i].log_lat_coordinates[0], userlist[i].log_lat_coordinates[1], userlist[i].log_lat_coordinates[2],
             userlist[i].angle_user, userlist[i].current_waiting_data, userlist[i].newarrivaldata, userlist[i].request,
             userlist[i].current_txdata, userlist[i].time_delay, userlist[i].qci,
             userlist[i].number_of_rbg_nedded, userlist[i].total_txdata, userlist[i].throughput, userlist[i].average_throughput,
             userlist[i].capacity))
    traffic_info = np.array(traffic_info, dtype='float')
    traffic_info = pd.DataFrame(traffic_info,
                                columns=['Use_ID', 'Lon', 'Lat', 'Alt', 'Angle', 'waitingdata', 'newdata',  'Req_ID',
                                         'last_time_txdata', 'time_delay', 'CQI', 'number_of_rbg_nedded', 'total_txdata',
                                         'throughput(mbps)', 'average_throughput', 'capacity'])
    return traffic_info, user_request


def get_all_user_position_and_request(user):
    userlist = user
    position_and_req = []
    for i in range(len(userlist)):
        position = userlist[i].position.tolist()
        position_and_req.append((i, position[0], position[1], position[2], userlist[i].request))
    position_and_req = np.array(position_and_req, dtype='float')
    return position_and_req