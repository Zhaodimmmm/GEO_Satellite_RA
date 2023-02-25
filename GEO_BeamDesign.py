import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class beam_design():
    def __init__(self, SateBLH):
        self.SateBLH = SateBLH

    def createInitBeamCenterPos(self, satnum, numPointsof, SateBLH, numPoints, arcDist):
        beamCenterAlloc = []
        beamCenterAlloc_lat_log = []
        for i in range(numPoints):
            # 方位角设计
            cellid = satnum * 48 + numPointsof + i
            azimuth = (i + 1) * 360.0 / numPoints
            a = 6378137.0
            b = 6356752.3142
            f = 1.0 / 298.257223563
            alpha1 = azimuth * m.pi / 180.0
            sinAlpha1 = m.sin(alpha1)
            cosAlpha1 = m.cos(alpha1)
            tanU1 = (1 - f) * m.tan(SateBLH[0] * m.pi / 180.0)
            cosU1 = 1 / m.sqrt((1 + tanU1 * tanU1))
            sinU1 = tanU1 * cosU1
            sigma1 = m.atan2(tanU1, cosAlpha1)
            sinAlpha = cosU1 * sinAlpha1
            cosSqAlpha = 1 - sinAlpha * sinAlpha
            uSq = cosSqAlpha * (a * a - b * b) / (b * b)
            A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
            B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
            sigma = arcDist / (b * A)
            sigmaP = 2 * m.pi
            sinSigma = m.sin(sigma)
            cosSigma = m.cos(sigma)
            cos2SigmaM = m.cos(2 * sigma1 + sigma)
            while (abs(sigma - sigmaP) > 1e-12):
                deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM) -
                                    B / 6 * cos2SigmaM * ( -3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)))
                sigmaP = sigma
                sigma = arcDist / (b * A) + deltaSigma
            tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1
            lat2 = m.atan(
                (sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1) / ((1 - f) * m.sqrt(sinAlpha * sinAlpha + tmp * tmp)))
            lambd = m.atan((sinSigma * sinAlpha1) / (cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1))
            C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
            L = lambd - (1 - C) * f * sinAlpha * (
                    sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)))
            pointPosition = GeographicToCartesianCoordinates(lat2 * 180 / m.pi, SateBLH[1] + L * 180 / m.pi, 0, "GRS80")
            lat_log = ConstructFromVector(pointPosition[0], pointPosition[1], pointPosition[2], "GRS80")
            beamCenterAlloc.append([cellid, pointPosition[0], pointPosition[1], pointPosition[2]])
            beamCenterAlloc_lat_log.append([cellid, lat_log[0], lat_log[1], lat_log[2]])
        return beamCenterAlloc, beamCenterAlloc_lat_log


def GeographicToCartesianCoordinates(latitude, longitude, altitude, sphType):
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
    cartesianCoordinates = [x, y, z]
    return cartesianCoordinates


def ConstructFromVector(x, y, z, sphType):
    # a: semi - major axis of earth
    # e: first eccentricity of earth
    EARTH_RADIUS = 6371e3
    EARTH_SEMIMAJOR_AXIS = 6378137
    EARTH_GRS80_ECCENTRICITY = 0.0818191910428158
    EARTH_WGS84_ECCENTRICITY = 0.0818191908426215
    if sphType == "SPHERE":
        a = EARTH_RADIUS
        e = 0
    if sphType == "GRS80":
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_GRS80_ECCENTRICITY
    else:  # if sphType == WGS84
        a = EARTH_SEMIMAJOR_AXIS
        e = EARTH_WGS84_ECCENTRICITY

    latitudeRadians = m.asin(z / m.sqrt(x ** 2 + y ** 2 + z ** 2))
    latitude = latitudeRadians * 180 / m.pi

    if x == 0 and y > 0:
        longitude = 90
    elif x == 0 and y < 0:
        longitude = -90
    elif x < 0 and y >= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi + 180
    elif x < 0 and y <= 0:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi - 180
    else:
        longitudeRadians = m.atan(y / x)
        longitude = longitudeRadians * 180 / m.pi

    Rn = a / (m.sqrt(1 - pow(e, 2) * pow(m.sin(latitudeRadians), 2)))
    altitude = m.sqrt(x**2+y**2+z**2)-Rn
    return [latitude, longitude, altitude]


def setInitBeamCenterPos(i, SateBLH, type):
    if type == "IRIDIUM":
        # 波束的数量以及功率大小
        numPointsofC1 = 3
        numPointsofC2 = 9
        # 设定每一圈里中心距离orbit inclination : 86.402 degree, 1.508 rad
        # maxArcDistFromSubPos = 40008080.0 / 20.0  # 3334km：单星覆盖arc长度
        maxArcDistFromSubPos = 2000000
        arcDist1 = maxArcDistFromSubPos * 1 / 4
        arcDist2 = maxArcDistFromSubPos * 3 / 4
        sate = beam_design(SateBLH)
        # 设置每一层
        allocate1, lat_log1 = sate.createInitBeamCenterPos(i, 0, SateBLH, numPointsofC1, arcDist1)
        allocate2, lat_log2 = sate.createInitBeamCenterPos(i, numPointsofC1, SateBLH, numPointsofC2, arcDist2)
        beam = np.array(allocate1 + allocate2)
        lat_log = np.array(lat_log1 + lat_log2)
    return beam, lat_log


def userconnectsate(userposition, beamposition, request, usernumer):
    all_connect_info = []
    beam_number = np.zeros(usernumer)
    for i in range(len(userposition)):
        user = userposition[i]
        distance_max = np.inf
        connect_beam_position = 0
        for j in range(len(beamposition)):
            beam = beamposition[j][1:]
            distance = np.sqrt(np.sum((user-beam)**2))
            if distance < distance_max:
                distance_max = distance
                connect_beam_position = beam
                beam_number[request[i]] = beamposition[j][0]+1
        user_connect_info = np.hstack((request[i], user, connect_beam_position, [0, 0, 0], beam_number[request[i]]))
        all_connect_info.append(user_connect_info)
    all_connect_info = np.array(all_connect_info)
    return all_connect_info, beam_number


if __name__ == "__main__":

    _, allocate = setInitBeamCenterPos(0, [0, 0, 0], "IRIDIUM")
    print(allocate)

    x0 = 0
    y0 = 0
    x1 = allocate[:, 1]
    y1 = allocate[:, 2]
    user = np.loadtxt('2001.txt', encoding='utf-8', delimiter=" ")
    x2 = user[:, 0]
    y2 = user[:, 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x0, y0, c='r', marker='^')
    ax1.scatter(x1, y1, c='r', marker='x')
    ax1.scatter(x2, y2, c='b', marker='.')
    threedB = [1.8603, 1.4941, 1.5433, 1.5711, 1.8851, 1.3499, 1.6659, 1.5355, 1.7594, 1.6371, 1.6957, 1.4500]
    onedB = [0.5370, 0.4313, 0.4455, 0.4535, 0.5442, 0.3897, 0.4809, 0.4432, 0.5079, 0.4726, 0.4895, 0.4186]
    bigonedB = [0.5370, 0, 0, 0, 0.5442, 0, 0, 0, 0.5079, 0, 0, 0]

    for i in range(12):
        cir1 = Circle(xy=(x1[i], y1[i]), radius=35786000 * m.tan(m.radians(onedB[i]))/111000, alpha=0.4)
        ax1.add_patch(cir1)
        cir2 = Circle(xy=(x1[i], y1[i]), radius=35786000 * m.tan(m.radians(threedB[i]))/111000, alpha=0.2)
        ax1.add_patch(cir2)
        cir3 = Circle(xy=(x1[i], y1[i]), radius=35786000 * m.tan(m.radians(bigonedB[i])) / 111000, alpha=0.7)
        ax1.add_patch(cir3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

