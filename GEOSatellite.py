import numpy as np
import math as m
import GEO_BeamDesign
from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs72
# // https://celestrak.com/NORAD


def seesate(epoch, step):
    """
    在限定用户区域范围内寻找可见星;
    输入参数: 
    返回：
    seesate为可见星坐标[lat, lon, alt], 
    seesateId为可见星编号+可见星坐标[num, x, y, z], 
    BeamSeeAlloocate包含四部分: (1)可见星坐标[lat, lon, alt], (2)小区编号与波束中心位置[num, x, y, z], (3)小区编号与波束中心位置[num, lat, lon, alt], (4)卫星坐标[x, y, z]
    """
    TLE_PATH = "./Intelsat.txt"
    satlist, tle_str1, tle_str2 = read_tle_file(TLE_PATH)  # satlist卫星列表，tle_str1TLE文件第一行,tle_str2TLE文件第二行
    position, SateId, SateBLH, undersate, beamallocate = sateposition(satlist, tle_str1, tle_str2, epoch, step)   # position为卫星位置[x, y, z], SateId为卫星编号+卫星位置, undersate为星下点坐标集合[x, y, z], allocate为卫星坐标[lat, lon, alt]+小区编号与波束中心[num, x, y, z]和[num, lat, lon, alt]
    centerLatLonAlt, radius = limitquare()
    BeamSeeAlloocate, seesate, seesateId = [], [], []
    for i in range(len(satlist)):
        distance = m.sqrt((undersate[i][0]-centerLatLonAlt[0])**2+(undersate[i][1]-centerLatLonAlt[1])**2+(undersate[i][2]-centerLatLonAlt[2])**2)     # distance 卫星星下点与用户收集区域中心之间的距离
        sate2undersate = m.sqrt((position[i][0]-undersate[i][0])**2+(position[i][1]-undersate[i][1])**2+(position[i][2]-undersate[i][2])**2)      # sate2undersate 卫星与星下点之间的距离
        sate2center = m.sqrt((position[i][0]-centerLatLonAlt[0])**2+(position[i][1]-centerLatLonAlt[1])**2+(position[i][2]-centerLatLonAlt[2])**2)    # sate2undersate 卫星与用户收集区域中心之间的距离
        # equatorRadius = 6378137.0
        # A = m.acos(equatorRadius/(equatorRadius+SateBLH[i][2]))*180/m.pi
        # L = A*m.pi*equatorRadius/180 
        if distance < 235000-radius:    # 寻找可见星
            seesate.append(SateBLH[i])
            seesateId.append(SateId[i])
            for j in range(len(beamallocate)):
                if beamallocate[j][0] == SateBLH[i]:
                    BeamSeeAlloocate.append(beamallocate[j] + [position[i]])
    # seesate为可见星的坐标集合[lat, lon, alt], seesateId为可见星编号+可见星坐标[num, x, y, z], BeamSeeAlloocate为卫星坐标[lat, lon, alt]+小区编号与波束中心[num, x, y, z]和[num, lat, lon, alt]+可见星坐标[x, y, z]
    return seesate, seesateId, BeamSeeAlloocate


def read_tle_file(TLE_PATH):
    """
    限定用户收集信息区域;
    SimpleScenario: 以ueCenterLatLonAlt中的经纬度为中心划定用户区域;
    输入参数: 用户区域半径, 用户区域的中心坐标[lat, lon, alt];
    返回: 用户区域的中心坐标[x, y, z]和用户区域半径;
    """
    satlist, tle_str1, tle_str2 = [], [], []
    f = open(TLE_PATH)
    lines = f.readlines()
    for i in range(len(lines) // 3):
        satlist.append(lines[3 * i])
        tle_str1.append(lines[3 * i + 1])
        tle_str2.append(lines[3 * i + 2])
    return satlist, tle_str1, tle_str2


def sateposition(satlist, tle_str1, tle_str2, epoch, step):
    """
    获取所有卫星的详细信息；
    输入参数: satlist卫星列表, tle_str1为TLE文件第一行, tle_str2为TLE文件第二行; epoch, step为强化学习概念;
    返回: 
    position: 卫星位置, [x, y, z];
    SateId: 卫星编号+卫星位置, 列表[num, [x, y, z]];
    SateBLH: 卫星位置, [latitude, longitude, altitude];
    undersate:卫星星下点位置, [x, y, z];
    allocate: 小区编号+波束中心, [[num, x, y, z], [num, latitude, longitude, altitude]];
    velocity: 速度;
    """
    position, SateId, SateBLH, undersate, allocate, velocity = [], [], [], [], [], []
    for i in range(len(satlist)):
        time = (step + epoch * 100) * 0.001  # 假设每一个epoch有100个step,1个step为1ms(或者1s),可以修改；time的单位是s
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = (time - hours * 3600 - minutes * 60)
        satellite = twoline2rv(tle_str1[i], tle_str2[i], wgs72)
        position1, velocity1 = satellite.propagate(2020, 6, 29, hours, minutes, seconds)  # 时间可到秒, 时间应该设置为可变, position是位置[x,y,z]
        position1 = (position1[0] * 1000, position1[1] * 1000, position1[2] * 1000)  # position是位置[x,y,z], km变成m
        position.append(position1)
        SateId.append([satlist[i], position1])
        velocity.append(velocity1)
        SateBLH1 = GEO_BeamDesign.ConstructFromVector(position1[0], position1[1], position1[2], "GRS80")      # 卫星位置, [latitude, longitude, altitude]
        SateBLH.append(SateBLH1)
        underposition = GEO_BeamDesign.GeographicToCartesianCoordinates(SateBLH1[0], SateBLH1[1], 0, "GRS80")   # 星下点坐标xy, [x,y,z]
        undersate.append(underposition)
        allocate.extend(GEO_BeamDesign.setInitBeamCenterPos(i, SateBLH1, "INTELSAT"))     # 小区坐标, [x,y,z], [latitude, longitude, altitude]
    # position为卫星位置[x, y, z], SateId为卫星编号+卫星位置, undersate为星下点坐标集合[x, y, z], allocate为卫星坐标[lat, lon, alt]+小区编号与波束中心[num, x, y, z]和[num, lat, lon, alt]
    return position, SateId, SateBLH, undersate, allocate


def limitquare():
    """
    限定用户收集信息区域;
    SimpleScenario: 以ueCenterLatLonAlt中的经纬度为中心划定用户区域;
    返回: 用户区域的中心坐标[x, y, z]和用户区域半径;
    """
    radius = 80000     # 小区辐射半径, 单位是m
    centerLatLonAlt = GEO_BeamDesign.GeographicToCartesianCoordinates(0, 0, 0, "GRS80")  # 用户区域中心，[lat, lon, alt] = [0, 0, 0]
    return centerLatLonAlt, radius


def userconnectsate(ueposition, epoch, step):
    """
    判断用户接入那个波束（就近原则）
    """
    # seesate为可见星的坐标集合，seesateId为卫星编号+卫星坐标BLH,BeamSeeAlloocate为卫星坐标（BLH）+小区编号+波束中心+卫星坐标（x,y,z）
    seesatenum, SeesateId, beamallocate = seesate(epoch, step)
    # 先找可见卫星
    satuedict, Beam, UeLinkSate = [], [], []
    for i in range(len(ueposition)):
        ue , min, link, beam = [], [], [], []
        for j in range(len(beamallocate)):
            uebeamdistance = m.sqrt((beamallocate[j][2][0]-ueposition[i][0])**2+(beamallocate[j][2][1]-ueposition[i][1])**2+(beamallocate[j][2][2]-ueposition[i][2])**2)
            if len(ue) != 0:
                mindistance = ue[0]
                if mindistance > uebeamdistance:
                    ue.remove(mindistance)
                    ue.append(uebeamdistance)
                    min.remove(min[0])
                    min.append([ueposition[i]]+ beamallocate[j])
                    link.remove(link[0])
                    link.append([ueposition[i], beamallocate[j][1]])
                    beam.remove(beam[0])
                    beam.append([beamallocate[j][1], beamallocate[j][2]])
            else:
                ue.append(uebeamdistance)
                min.append([ueposition[i]] + beamallocate[j])
                link.append([ueposition[i], beamallocate[j][1]])
                beam.append([beamallocate[j][1], beamallocate[j][2]])
        satuedict.append(min[0])  # 用户位置+卫星位置(B,L,H)+波束编号+波束中心+卫星位置（x,y,z）
        Beam.append(beam[0])
        UeLinkSate.append(link[0])  # 波束编号+波束中心
    return satuedict, Beam, UeLinkSate


if __name__ == "__main__":
    seesate, seesateId, BeamSeeAlloocate = seesate(0, 0)
    print('seesate', seesate)
    print('seesateId', seesateId)