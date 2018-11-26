import aifc
import os
import struct
import pandas
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

rawDataDir = 'Data/Raw/'
parsedDataDir = 'Data/Parsed/'

def read_AIFFs_to_csv():
    rawFiles = os.listdir(rawDataDir)

    soundDatas = []

    for file in rawFiles:
        print(file)
        label = file.split('.')[0]
        aiffData = aifc.open(rawDataDir + file, 'r')

        numFrames = aiffData.getnframes()
        i = 0
        channel1 = []
        channel2 = []
        while i < numFrames:
            frame = bytearray(aiffData.readframes(1))
            # print(frame)
            data = list(map(int, frame))
            # print(data)
            channel1.append((data[0]<<16) + (data[1]<<8) + data[2])
            channel2.append((data[3]<<16) + (data[4]<<8) + data[5])
            # print((data[0]<<16) + (data[1]<<8) + data[2])
            # print((data[3]<<16) + (data[4]<<8) + data[5])
            i += 1

        plt.plot(channel1)
        plt.show()
        
        # data = bytearray(aiffData.readframes(aiffData.getnframes()))
        # data = list(map(int, data))
        # msb = [data[i] for i in range(0, len(data), 3)]
        # midb = [data[i] for i in range(1, len(data), 3)]
        # lsb = [data[i] for i in range(2, len(data), 3)]
        # data = list(map(lambda m, mid, l: m*65536 + mid*256 + l, msb, midb, lsb))
        # channel1 = [data[i] for i in range(0, len(data), 2)]
        # channel2 = [data[i] for i in range(1, len(data), 2)]
        # plt.plot(channel1)
        # plt.show()
        # soundDatas.append(channel1)
        # soundDatas.append(channel2)
        aiffData.close()

    print('Done reading AIFFs')
    df = pandas.DataFrame(soundDatas)
    df.to_csv(parsedDataDir + 'WaveData.csv')
    print('Done writing CSV')





if __name__ == '__main__':
    read_AIFFs_to_csv()
    data = pandas.read_csv(parsedDataDir + 'WaveData.csv')
    audio = data.iloc[1, 1:].values
    write('test.wav', 44100, audio)