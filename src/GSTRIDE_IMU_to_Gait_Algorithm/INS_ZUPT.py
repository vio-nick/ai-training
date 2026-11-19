import numpy as np
from scipy.linalg import expm


class INS_ZUPT:
    # INS_ZUPT implementation of Antonio Ramon Jimenez Ruiz algorithm

    # For more information see:
    # https://lopsi.weebly.com/personal-indoor-localization.html

    # e.g. How to use it in main script:
    # 1) Init object:       ZUPT=INS_ZUPT(fs,len(Step_events))
    #    with the positions of the steps detected previously, in the imu data array
    # 2) Pass all IMU data  ZUPT.UPDATE_ZUPT(Acc,Gyr,Step_events,Still)
    #-----------------------------constructor----------------------
    def __init__(self, fs, num_steps):
        self.fs = fs                          # IMU sample rate
        self.num_steps = num_steps            # Number of steps previously detected
        # Window for gravitational acceleration estimation
        self.window = int(np.floor(15 * fs))
    #%================================Update======================================

    def UPDATE_ZUPT(self, Acc, Gyr, Step_event, Still):

        # Acc = [num_samples,3]         Accelerometer data
        # Gyr = [num_samples,3]         Gyroscope data
        # Step_event = [num_steps]      Position of the detected steps
        # Still=[num_samples]           Still condition vector

        num_samples = Acc.shape[0]

        # Init parameters
        Acc_G = np.zeros((num_samples, 3))
        Speed_G = np.zeros((num_samples, 3))
        self.Speed_G_corr = np.zeros((num_samples, 3))

        # Atleast during 20 seconds there must be no movement to estimate gravity vector
        cont = 0
        for i in range(len(Still)):
            if Still[i] == 1:
                cont = cont + 1
        gravedades = np.zeros(cont)
        cont = 0
        for i in range(len(Still)):
            if Still[i] == 1:
                gravedades[cont] = np.sqrt(
                    Acc[i, 0]**2 + Acc[i, 1]**2 + Acc[i, 2]**2)
                cont = cont + 1
        gravedad = np.mean(gravedades)

        # 1) Linear Accelerations from SENSOR (S) to GLOBAL (G) coordinates
        # Init rotation matrix:
        Acc_fs = Acc[0:self.window, :]
        Acc_mean = np.mean(Acc_fs, axis=0)
        # Roll en grados
        Roll_ini = (np.arctan2(Acc_mean[1], Acc_mean[2])) * 180 / (np.pi)
        Pitch_ini = -np.arctan2(Acc_mean[0], np.sqrt(Acc_mean[1]
                                                     ** 2 + Acc_mean[2]**2)) * 180 / (np.pi)  # Roll en grados
        Yaw_ini = 0
        # from S-Sensor to G-Global
        Rot_z = np.matrix([[np.cos(Yaw_ini * 3.1416 / 180), -np.sin(Yaw_ini * 3.1416 / 180), 0], [
                          np.sin(Yaw_ini * 3.1416 / 180), np.cos(Yaw_ini * 3.1416 / 180), 0], [0, 0, 1]])
        Rot_y = np.matrix([[np.cos(Pitch_ini * 3.1416 / 180), 0, np.sin(Pitch_ini * 3.1416 / 180)], [
                          0, 1, 0], [-np.sin(Pitch_ini * 3.1416 / 180), 0, np.cos(Pitch_ini * 3.1416 / 180)]])
        Rot_x = np.matrix([[1, 0, 0], [0, np.cos(Roll_ini * 3.1416 / 180), -np.sin(Roll_ini *
                                                                                   3.1416 / 180)], [0, np.sin(Roll_ini * 3.1416 / 180), np.cos(Roll_ini * 3.1416 / 180)]])
        Rot_GS = Rot_z[:, :] * Rot_y[:, :] * Rot_x[:, :]
        for i in range(num_samples):
            # Update orientation matrix with Giroscope data
            SkewGyros = np.matrix([[0, -Gyr[i, 2], Gyr[i, 1]],
                                   [Gyr[i, 2], 0, -Gyr[i, 0]],
                                   [-Gyr[i, 1], Gyr[i, 0], 0]])
            if i > 0:
                # function "expm" which is a Pade (6,6) aproximation
                Rot_GS = np.dot(Rot_GS, expm(SkewGyros / self.fs))
            # Transofrm linear accelerations
            # Rot_GS from S system to G system
            Acc_G[i, :] = np.dot(Rot_GS, Acc[i, :].T)
        Acc_G[:, 2] = Acc_G[:, 2] - gravedad

        # 2) Integration to linear speed
        Speed_G[0, :] = np.matrix([0, 0, 0])
        for i in range(1, num_samples):
            Speed_G[i, :] = Speed_G[i - 1, :] + 0.5 * \
                (Acc_G[i, :] + Acc_G[i - 1, :]) / self.fs

        # 3) Linear speed correction
        # Mean speed error at stance
        Error_Speed_G = np.zeros((self.num_steps + 2, 3))
        Step_event_ext = np.zeros(self.num_steps + 2)
        Step_event_ext[0] = 0
        for i in range(num_samples):
            # If Still==True (absolute rest), correct speed to zero
            if Still[i] > 0:
                self.Speed_G_corr[i, :] = [0, 0, 0]

        for i in range(self.num_steps):
            Step_event_ext[i + 1] = Step_event[i]
            Step_event_ext[self.num_steps + 1] = num_samples - 1
        for i in range(self.num_steps + 2):
            # Locate position where a step has been detected
            posAux = int(Step_event_ext[i])
            Error_Speed_G[i, :] = Speed_G[posAux, :]

        # Correct speed by interpolation (every sample)
        for k in range(1, self.num_steps + 1):
            for i in range(int(Step_event_ext[k - 1]), int(Step_event_ext[k]) - 1):
                samples_in_step = int(
                    Step_event_ext[k]) - int(Step_event_ext[k - 1])
                # If Still==True (absolute rest), correct speed to zero
                if Still[i] > 0:
                    self.Speed_G_corr[i, :] = [0, 0, 0]
                else:
                    self.Speed_G_corr[i, :] = Speed_G[i, :] - (Error_Speed_G[k, :] * (
                        i - Step_event_ext[k - 1]) + Error_Speed_G[k - 1, :] * (Step_event_ext[k] - i)) / samples_in_step

        # 4) deltasXYZ estimation
        # Speed integration to obtain linear displacement
        self.XYZ = np.zeros((num_samples + 1, 3))
        self.deltasXYZ = np.zeros((num_samples, 3))
        for i in range(1, num_samples):
            self.XYZ[i + 1] = self.XYZ[i] + 0.5 * \
                (self.Speed_G_corr[i, :] +
                 self.Speed_G_corr[i - 1, :]) / self.fs
            self.deltasXYZ[i, :] = 0.5 * \
                (self.Speed_G_corr[i, :] +
                 self.Speed_G_corr[i - 1, :]) / self.fs
