import numpy as np


class Gait_Parameters_Estimation:
    # e.g. How to use it in main script:
    # 1) Init object:       GPE = Gait_Parameters_Estimation(len(Step_events))
    #    With the positions of the steps detected previously, in the imu data array
    # 2) Pass IMU data, AHRS.angles and INS-ZUPT estimations:
    #    GPE.estimate(Phases,ZUPT.Speed_G_corr,ZUPT.deltasXYZ,ZUPT.XYZ,AHRS.angles[:,ori],timeStamps)
    #-----------------------------constructor----------------------
    def __init__(self, num_steps):
        self.PTO = np.zeros(num_steps)              # Toe Off Percentage (%)
        self.Pswing = np.zeros(num_steps)           # Swing Percentage (%)
        # Heel Strike Percentage (%)
        self.PHS = np.zeros(num_steps)
        self.PFF = np.zeros(num_steps)              # Foot Flat Percentage (%)
        self.tcycle = np.zeros(num_steps)           # Gate Cycle (s)
        self.TO = np.zeros(num_steps)               # Toe Off Angle (°)
        self.HS = np.zeros(num_steps)               # Heel Strike Angle (°)
        self.cadence = np.zeros(num_steps)          # Cadence  (steps/min)
        self.speed = np.zeros(num_steps)            # Speed (m/s)
        self.StrideLengths = np.zeros(num_steps)    # Stride Length (m)
        self.Path3Dlength = np.zeros(num_steps)     # Path 3D Length (m)
        self.Path2Dlength = np.zeros(num_steps)     # Path 2D Length (m)
        self.clearance = np.zeros(num_steps)        # Clearance (m)
        num_statistics = 29
        self.statistics = np.zeros(num_statistics)  # Statistics of parameters
        self.limit = 0.4
    #%================================Update======================================

    def estimate(self, Phases, Speed, deltasXYZ, XYZ, Pitch, Pitch0, timeStamps):

        # Phases=[num_steps,5]          Registered detection points of each gait phase (endFFprev,TO,HS,newFF,endFF)
        # Speed=[num_samples,3]         Speed corrected by ZUPT
        # deltasXYZ=[num_samples,3]     Deltas estimation by INS ZUPT
        # XYZ=[num_samples,3]           Deltas estimation by INS ZUPT
        # Pitch=[num_samples]           Pitch angle
        # timeStamps=[num_samples]      IMU time stamp data

        num_steps = Phases.shape[0]
        Speed_ZUPT = np.sqrt(Speed[:, 0]**2 + Speed[:, 1]**2 + Speed[:, 2]**2)
        deltas3d = np.sqrt(deltasXYZ[:, 0]**2 +
                           deltasXYZ[:, 1]**2 + deltasXYZ[:, 2]**2)
        deltas2d = np.sqrt(deltasXYZ[:, 0]**2 + deltasXYZ[:, 1]**2)
        for i in range(num_steps):
            self.tcycle[i] = timeStamps[int(
                Phases[i, 4])] - timeStamps[int(Phases[i, 0])]
            self.PTO[i] = (timeStamps[int(Phases[i, 1])] -
                           timeStamps[int(Phases[i, 0])]) * 100 / self.tcycle[i]
            self.Pswing[i] = (timeStamps[int(Phases[i, 2])] -
                              timeStamps[int(Phases[i, 1])]) * 100 / self.tcycle[i]
            self.PHS[i] = (timeStamps[int(Phases[i, 3])] -
                           timeStamps[int(Phases[i, 2])]) * 100 / self.tcycle[i]
            self.PFF[i] = (timeStamps[int(Phases[i, 4])] -
                           timeStamps[int(Phases[i, 3])]) * 100 / self.tcycle[i]
            self.TO[i] = -abs(Pitch0[i] - Pitch[int(Phases[i, 1])])
            self.HS[i] = abs(Pitch[int(Phases[i, 2])] - Pitch0[i])
            self.cadence[i] = 60 / (self.tcycle[i])
            self.speed[i] = np.mean(
                Speed_ZUPT[int(Phases[i, 0]):int(Phases[i, 4])])
            self.StrideLengths[i] = np.sqrt((XYZ[int(Phases[i, 4]), 0] - XYZ[int(
                Phases[i, 0]), 0])**2 + (XYZ[int(Phases[i, 4]), 1] - XYZ[int(Phases[i, 0]), 1])**2)
            self.Path3Dlength[i] = np.sum(
                deltas3d[int(Phases[i, 0]):int(Phases[i, 4])])
            self.Path2Dlength[i] = np.sum(
                deltas2d[int(Phases[i, 0]):int(Phases[i, 4])])
            maxi = XYZ[int(Phases[i, 0]), 2]
            mini = XYZ[int(Phases[i, 0]), 2]
            for j in range(int(Phases[i, 0]), int(Phases[i, 4])):
                if XYZ[j, 2] >= maxi:
                    maxi = XYZ[j, 2]
                if XYZ[j, 2] <= mini:
                    mini = XYZ[j, 2]
            self.clearance[i] = maxi - mini

        StrideLengths = self.trunc_array(self.StrideLengths)
        self.statistics[0] = "{0:.3f}".format(float(np.sum(StrideLengths)))
        tcycle = self.trunc_array(self.tcycle)
        self.statistics[1] = "{0:.3f}".format(float(np.sum(tcycle)))
        self.statistics[2] = "{0:.0f}".format(float(int(num_steps)))

        self.statistics[3] = "{0:.3f}".format(float(np.mean(tcycle)))
        self.statistics[4] = "{0:.3f}".format(float(np.std(tcycle)))

        PTO = self.trunc_array(self.PTO)
        self.statistics[5] = "{0:.2f}".format(float(np.mean(PTO)))
        self.statistics[6] = "{0:.2f}".format(float(np.std(PTO)))

        Pswing = self.trunc_array(self.Pswing)
        self.statistics[7] = "{0:.2f}".format(float(np.mean(Pswing)))
        self.statistics[8] = "{0:.2f}".format(float(np.std(Pswing)))

        PHS = self.trunc_array(self.PHS)
        self.statistics[9] = "{0:.2f}".format(float(np.mean(PHS)))
        self.statistics[10] = "{0:.2f}".format(float(np.std(PHS)))

        PFF = self.trunc_array(self.PFF)
        self.statistics[11] = "{0:.2f}".format(float(np.mean(PFF)))
        self.statistics[12] = "{0:.2f}".format(float(np.std(PFF)))

        TO = self.trunc_array(self.TO)
        self.statistics[13] = "{0:.3f}".format(float(np.mean(TO)))
        self.statistics[14] = "{0:.3f}".format(float(np.std(TO)))

        HS = self.trunc_array(self.HS)
        self.statistics[15] = "{0:.3f}".format(float(np.mean(HS)))
        self.statistics[16] = "{0:.3f}".format(float(np.std(HS)))

        cadence = self.trunc_array(self.cadence)
        self.statistics[17] = "{0:.3f}".format(float(np.mean(cadence)))
        self.statistics[18] = "{0:.3f}".format(float(np.std(cadence)))

        speed = self.trunc_array(self.speed)
        self.statistics[19] = "{0:.3f}".format(float(np.mean(speed)))
        self.statistics[20] = "{0:.3f}".format(float(np.std(speed)))

        self.statistics[21] = "{0:.3f}".format(float(np.mean(StrideLengths)))
        self.statistics[22] = "{0:.3f}".format(float(np.std(StrideLengths)))

        Path3Dlength = self.trunc_array(self.Path3Dlength)
        self.statistics[23] = "{0:.3f}".format(float(np.mean(Path3Dlength)))
        self.statistics[24] = "{0:.3f}".format(float(np.std(Path3Dlength)))

        Path2Dlength = self.trunc_array(self.Path2Dlength)
        self.statistics[25] = "{0:.3f}".format(float(np.mean(Path2Dlength)))
        self.statistics[26] = "{0:.3f}".format(float(np.std(Path2Dlength)))

        clearance = self.trunc_array(self.clearance)
        self.statistics[27] = "{0:.3f}".format(float(np.mean(clearance)))
        self.statistics[28] = "{0:.3f}".format(float(np.std(clearance)))

        # Change NaN values to mean of the vector, just for visualization of the parameters
        # not for obtaining the statistics, in statisticts we remove NaN values
        np.nan_to_num(self.tcycle, copy=False,
                      nan=self.statistics[3], posinf=self.statistics[3], neginf=self.statistics[3])
        np.nan_to_num(self.PTO, copy=False,
                      nan=self.statistics[5], posinf=self.statistics[5], neginf=self.statistics[5])
        np.nan_to_num(self.Pswing, copy=False,
                      nan=self.statistics[7], posinf=self.statistics[7], neginf=self.statistics[7])
        np.nan_to_num(self.PHS, copy=False,
                      nan=self.statistics[9], posinf=self.statistics[9], neginf=self.statistics[9])
        np.nan_to_num(self.PFF, copy=False,
                      nan=self.statistics[11], posinf=self.statistics[11], neginf=self.statistics[11])
        np.nan_to_num(self.TO, copy=False,
                      nan=self.statistics[13], posinf=self.statistics[13], neginf=self.statistics[13])
        np.nan_to_num(self.HS, copy=False,
                      nan=self.statistics[15], posinf=self.statistics[15], neginf=self.statistics[15])
        np.nan_to_num(self.cadence, copy=False,
                      nan=self.statistics[17], posinf=self.statistics[17], neginf=self.statistics[17])
        np.nan_to_num(self.speed, copy=False,
                      nan=self.statistics[19], posinf=self.statistics[19], neginf=self.statistics[19])
        np.nan_to_num(self.StrideLengths, copy=False,
                      nan=self.statistics[21], posinf=self.statistics[21], neginf=self.statistics[21])
        np.nan_to_num(self.Path3Dlength, copy=False,
                      nan=self.statistics[23], posinf=self.statistics[23], neginf=self.statistics[23])
        np.nan_to_num(self.Path2Dlength, copy=False,
                      nan=self.statistics[25], posinf=self.statistics[25], neginf=self.statistics[25])
        np.nan_to_num(self.clearance, copy=False,
                      nan=self.statistics[27], posinf=self.statistics[27], neginf=self.statistics[27])

    def trunc_array(self, arr):
        arr = arr[~np.isnan(arr)]
        median_arr = np.median(arr)
        limit_upper = median_arr * (1 + self.limit)
        limit_lower = median_arr * (self.limit)
        if median_arr > 0:
            idx = np.where((arr <= limit_upper) & (arr >= limit_lower))
        else:
            idx = np.where((arr >= limit_upper) & (arr <= limit_lower))
        arr = arr[idx]
        return arr
