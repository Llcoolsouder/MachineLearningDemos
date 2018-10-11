import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dates = []
time = []
diningRoomTemps = []
roomTemps = []
forecastTemp = []
co2DiningRoom = []
co2Room = []
relativehumidtyDiningRoom = []
relativehumidtyRoom = []
lightingDiningRoom = []
lightingRoom = []
rain = []
sunDusk = []
wind = []
sunlightWest = []
sunlightEast = []
sunlightSouth = []
sunIrradiance = []
enthalpicMotor1 = []
enthalpicMotor2 = []
enthalpicMotorTurbo = []
outdoorTemp = []
outdoorRelativeHumidity = []
dayOfWeek = []


with open('NEW-DATA-1.T15.txt', 'r') as dataFile:
    for line in dataFile:
        columns = line.split()
        dates.append(columns[0])
        time.append(columns[1])
        diningRoomTemps.append(columns[2])
        roomTemps.append(columns[3])
        forecastTemp.append(columns[4])
        co2DiningRoom.append(columns[5])
        co2Room.append(columns[6])
        relativehumidtyDiningRoom.append(columns[7])
        relativehumidtyRoom.append(columns[8])
        lightingDiningRoom.append(columns[9])
        lightingRoom.append(columns[10])
        rain.append(columns[11])
        sunDusk.append(columns[12])
        wind.append(columns[13])
        sunlightWest.append(columns[14])
        sunlightEast.append(columns[15])
        sunlightSouth.append(columns[16])
        sunIrradiance.append(columns[17])
        enthalpicMotor1.append(columns[18])
        enthalpicMotor2.append(columns[19])
        enthalpicMotorTurbo.append(columns[20])
        outdoorTemp.append(columns[21])
        outdoorRelativeHumidity.append(columns[22])
        dayOfWeek.append(columns[23])
    
dates = dates[1:len(dates)]
time = time[1:len(time)]
diningRoomTemps = diningRoomTemps[1:len(diningRoomTemps)]
roomTemps = roomTemps[1:len(roomTemps)]
forecastTemp = forecastTemp[1:len(forecastTemp)]
co2DiningRoom = co2DiningRoom[1:len(co2DiningRoom)]
co2Room = co2Room[1:len(co2Room)]
relativehumidtyDiningRoom = relativehumidtyDiningRoom[1:len(relativehumidtyDiningRoom)]
relativehumidtyRoom = relativehumidtyRoom[1:len(relativehumidtyRoom)]
lightingDiningRoom = lightingDiningRoom[1:len(lightingDiningRoom)]
lightingRoom = lightingRoom[1:len(lightingRoom)]
rain = rain[1:len(rain)]
sunDusk = sunDusk[1:len(sunDusk)]
wind = wind[1:len(wind)]
sunlightWest = sunlightWest[1:len(sunlightWest)]
sunlightEast = sunlightEast[1:len(sunlightEast)]
sunlightSouth = sunlightSouth[1:len(sunlightSouth)]
sunIrradiance = sunIrradiance[1:len(sunIrradiance)]
enthalpicMotor1 = enthalpicMotor1[1:len(enthalpicMotor1)]
enthalpicMotor2 = enthalpicMotor2[1:len(enthalpicMotor2)]
enthalpicMotorTurbo = enthalpicMotorTurbo[1:len(enthalpicMotorTurbo)]
outdoorTemp = outdoorTemp[1:len(outdoorTemp)]
outdoorRelativeHumidity = outdoorRelativeHumidity[1:len(outdoorRelativeHumidity)]
dayOfWeek = dayOfWeek[1:len(dayOfWeek)]

outdoorRelativeHumidity = list(map(float, outdoorRelativeHumidity))
outdoorTemp = list(map(float, outdoorTemp))

regr = linear_model.LinearRegression()

outdoorTemp_train = outdoorTemp[0:int(len(outdoorTemp)*0.7)]
outdoorRelativeHumidity_train = outdoorRelativeHumidity[0:int(len(outdoorRelativeHumidity)*0.7)]

regr.fit(np.asarray(outdoorTemp_train).reshape(-1,1), np.asarray(outdoorRelativeHumidity_train).reshape(-1,1))

outdoorTemp_test = outdoorTemp[len(outdoorTemp_train):len(outdoorTemp)]
print(np.asarray(outdoorTemp_test).reshape(-1,1))
outdoorRelativeHumidity_pred = regr.predict(np.asarray(outdoorTemp_test).reshape(-1,1))

plt.scatter(outdoorTemp, outdoorRelativeHumidity, color='blue')
plt.plot(outdoorTemp_test, outdoorRelativeHumidity_pred, color='red', linewidth=3)
plt.title('Outdoor Humidity vs Outdoor Temp')
plt.xlabel('Temperature [C]')
plt.ylabel('Humidity [%]')
plt.show()

####

co2DiningRoom = list(map(float, co2DiningRoom))
co2Room = list(map(float, co2Room))

regr = linear_model.LinearRegression()

co2Room_train = co2Room[0:int(len(co2Room)*0.7)]
co2DiningRoom_train = co2DiningRoom[0:int(len(co2DiningRoom)*0.7)]

regr.fit(np.asarray(co2Room_train).reshape(-1,1), np.asarray(co2DiningRoom_train).reshape(-1,1))

co2Room_test = co2Room[len(co2Room_train):len(co2Room)]
print(np.asarray(outdoorTemp_test).reshape(-1,1))
co2DiningRoom_pred = regr.predict(np.asarray(co2Room_test).reshape(-1,1))

plt.scatter(co2Room, co2DiningRoom, color='blue')
plt.plot(co2Room_test, co2DiningRoom_pred, color='red', linewidth=3)
plt.title('Dining Room CO2 vs Room CO2')
plt.xlabel('Room CO2 [ppm]')
plt.ylabel('Dining Room CO2 [ppm]')
plt.show()