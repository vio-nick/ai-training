import numpy as np
import scipy.io as sio


def readLogFileArduinoCalib(Filename, Nbeacon, fs):

    samplesBias = int(fs * 20)

    # Read logfile-----------------------------------------------------------------------
    with open(Filename, 'r') as file:
        print(file)
        lines = file.readlines()
    del(lines[0:fs * 3])

    # Read Calibration-----------------------------------------------------------------------
    calib = sio.loadmat('Calibration/calib_' + Nbeacon + '.mat')
    calib_struct = calib['calib_est'][0, 0]

    AccMiss = calib_struct[0]
    AccScale = calib_struct[1]
    AccBias = calib_struct[2][0]

    GyrMiss = calib_struct[3]
    GyrScale = calib_struct[4]
    GyrBias = calib_struct[5]

    clockError = calib_struct[6][0][0]

    fs = fs / clockError

    # Process data-----------------------------------------------------------------------
    Acc = np.zeros((len(lines), 3))
    Gyr = np.zeros((len(lines), 3))
    timeStamps = np.zeros(len(lines))

    row = 0
    for line in lines:
        data = line.split(' ')
        #data = line.split(',')
        Acc[row, 0] = float(data[0])
        Acc[row, 1] = float(data[1])
        Acc[row, 2] = float(data[2])

        Gyr[row, 0] = float(data[3])
        Gyr[row, 1] = float(data[4])
        l = len(data[5])
        Gyr[row, 2] = float(data[5][0:l - 1])
        #Gyr[row, 2] = float(data[5][0:l - 2])
        #Gyr [row,2] = float(data[5])

        if row > 0:
            timeStamps[row] = timeStamps[row - 1] + (1 / fs)

        # Calibration--------------------------------------------------------------------
        Acc[row, :] = np.dot(AccMiss, np.dot(
            AccScale, (Acc[row, :] - AccBias).T)).T
        Gyr[row, :] = np.dot(GyrMiss, np.dot(
            GyrScale, (Gyr[row, :] - GyrBias).T)).T

        row = row + 1

    bias_Gyr = [np.mean(Gyr[0:samplesBias, 0], axis=0), np.mean(
        Gyr[0:samplesBias, 1], axis=0), np.mean(Gyr[0:samplesBias, 2], axis=0)]

    for i in range(3):
        Gyr[:, i] = Gyr[:, i] - bias_Gyr[i]

    return Acc, Gyr, timeStamps, fs
