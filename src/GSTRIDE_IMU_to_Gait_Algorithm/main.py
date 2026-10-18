import numpy as np
import matplotlib.pyplot as plt
import sys
import os

dir = '../Test_recordings_raw/'
import csv
f1 = open('./Parameters.csv', 'w')
f2 = open('./Statistics.csv', 'w')
writer1 = csv.writer(f1)
writer2 = csv.writer(f2)
ori = 0
parameter = ["Toe Off Percentage (%)", "Swing Percentage (%)", "Heel Strike Percentage (%)", "Foot Flat Percentage (%)",
             "Gate Cycle (s)", "Toe Off Angle (°)", "Heel Strike Angle (°)", "Cadence (steps/min)", "Speed (m/s)",
             "Stride Length (m)", "Path 3D Length (m)", "Path 2D Length (m)", "Clearance (m)"]
#------------------------------------------------------------------------------------------
# Configuration parameters
#------------------------------------------------------------------------------------------
plotStepDetection = True              # Plot Step Detections True/False
plotTrayectory = True                   # Plot Trayectory True/False
plotHistogram = True                    # Plot Histogram True/False
PARAM = 9                               # Parameter to plot in Histogram
# 0 : Toe Off Percentage (%)
# 1 : Swing Percentage (%)
# 2 : Heel Strike Percentage (%)
# 3 : Foot Flat Percentage (%)
# 4 : Gate Cycle (s)
# 5 : Toe Off Angle (°)
# 6 : Heel Strike Angle (°)
# 7 : Cadence (steps/min)
# 8 : Speed (m/s)
# 9 : Stride Length (m)
# 10 : Path 3D Length (m)
# 11 : Path 2D Length (m)
# 12 : Clearance (m)

with os.scandir(dir) as files:
    for file in files:
        if file.name != '.DS_Store':
            aux = file.name.split("_")
            ID = aux[0]
            Filename = dir + file.name
            Nbeacon = aux[1].split(".")[0]
            if Nbeacon == '0185':
                fs = 128
            else:
                fs = 104
            #------------------------------------------------------------------------------------------
            # read Arduino Logfile and calibrate
            #------------------------------------------------------------------------------------------
            from readLogFileArduinoCalib import readLogFileArduinoCalib
            [Acc, Gyr, timeStamps, fs] = readLogFileArduinoCalib(
                Filename, Nbeacon, fs)
            #------------------------------------------------------------------------------------------
            # AHRS Mahony for Quaternions and Pitch
            #------------------------------------------------------------------------------------------
            from MahonyAHRS import MahonyAHRS
            # 1) Init object:
            AHRS = MahonyAHRS(fs, len(Acc))
            # 2) Use in a loop:
            for i in range(len(Acc)):
                AHRS.UpdateIMU(Gyr[i, :], Acc[i, :])
            #------------------------------------------------------------------------------------------
            # Step detection by Guillermo
            #------------------------------------------------------------------------------------------
            from StepDetection_MultiCorrector_Pitch import StepDetection_MultiCorrector_Pitch
            # 1) Pass Angles,Giroscope data, IMU orientation for pitch angle 0/1/2 (x/y/z), IMU sample rate)
            # 1) Return: Sitll phases, Gyroscope Magnitude, positions of still new phases, positions of still end phases, positions of foot flat new phase, positions of foot flat end phases,
            #            Heel Strike angle value, Heel Strike position, Toe Off angle, Toe Off position, Object with all the positions of each detected phase, Pitch angle
            [Still, M_Gyr, newStill, endStill, FF, newFF, Step_events, endFF, HS, HS_i, TO, TO_i,
                Phases, Pitch0] = StepDetection_MultiCorrector_Pitch(AHRS.angles, Gyr, ori, fs)
            #------------------------------------------------------------------------------------------
            # INS-ZUPT by Antonio Ramon Jimenez Ruiz
            #------------------------------------------------------------------------------------------
            from INS_ZUPT import INS_ZUPT
            # 1) Init object: with the positions of the steps detected previously, in the imu data array
            ZUPT = INS_ZUPT(fs, len(Step_events))
            # 2) Pass all IMU data
            ZUPT.UPDATE_ZUPT(Acc, Gyr, Step_events, Still)
            #------------------------------------------------------------------------------------------
            # Parameters estimation
            #------------------------------------------------------------------------------------------
            from Gait_Parameters_Estimation import Gait_Parameters_Estimation
            # 1) Init object: with the positions of the steps detected previously, in the imu data array
            GPE = Gait_Parameters_Estimation(len(Step_events))
            # 2) Pass IMU data, Step-detection data, AHRS.angles and INS-ZUPT estimations
            GPE.estimate(Phases, ZUPT.Speed_G_corr, ZUPT.deltasXYZ,
                         ZUPT.XYZ, AHRS.angles[:, ori], Pitch0, timeStamps)
            #------------------------------------------------------------------------------------------
            # Plots
            #------------------------------------------------------------------------------------------
            GaitParams = [GPE.PTO, GPE.Pswing, GPE.PHS, GPE.PFF, GPE.tcycle, GPE.TO, GPE.HS, GPE.cadence,
                          GPE.speed, GPE.StrideLengths, GPE.Path3Dlength, GPE.Path2Dlength, GPE.Path2Dlength]
            # Plot Step Detection
            if plotStepDetection == True:
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.plot(AHRS.angles[:, ori], '-',
                         linewidth=1, color='k', label='Pitch')
                ax2.plot(M_Gyr, '-', linewidth=1, color='b', label="M_Gyr")
                ax1.plot(Still, '-', linewidth=1, color='y')
                ax1.plot(newStill, np.ones(len(newStill)),
                         'o', linewidth=1, color='y')
                ax1.plot(endStill, np.ones(len(endStill)),
                         'o', linewidth=1, color='y')
                ax1.plot(FF, '-', linewidth=1, color='m', label='Foot Flat')
                ax1.plot(newFF, np.ones(len(newFF)), 'o', linewidth=1,
                         color='m', label='Toe Strike')
                ax1.plot(endFF, np.ones(len(endFF)), 'o', linewidth=1,
                         color='m', label='Heel Off')
                ax1.plot(HS_i, HS, 'o', linewidth=1,
                         color='g', label='Heel Strike')
                ax1.plot(TO_i, TO, 'o', linewidth=1,
                         color='r', label='Toe Off')
                ax2.set_ylim([-115, 15])
                ax2.set_ylabel('Angular velocity (rad/s)', rotation=270)
                pos1 = ax2.get_position() # get the original position 
                ax2.yaxis.set_label_coords(1.115, 0.8)
                ax1.set_ylim([-115, 15])
                ax1.set_ylabel('Pitch angle ($^\circ$)')
                ax1.legend(loc='lower right')
                ax1.set_xlabel('Samples (num. sample)')
                plt.show()
            # Plot Trayectory
            if plotTrayectory == True:
                plt.plot(ZUPT.XYZ[:, 0], ZUPT.XYZ[:, 1])
                plt.xlabel('Displacement in Y (m)')
                plt.ylabel('Displacement in X (m)')
                plt.show()
            # Plot Parameter Histogram
            from matplotlib import colors
            from matplotlib.ticker import PercentFormatter
            if plotHistogram == True:
                # Number of bins
                q25, q75 = np.percentile(GaitParams[PARAM], [25, 75])
                bin_width = 2 * (q75 - q25) * \
                    len(GaitParams[PARAM]) ** (-1 / 3)
                bins = round((GaitParams[PARAM].max() -
                              GaitParams[PARAM].min()) / bin_width)
                print("Freedman–Diaconis number of bins:", bins)
                # Creating histogram
                fig, axs = plt.subplots(
                    1, 1, figsize=(10, 7), tight_layout=True)
                # Remove axes splines
                for s in ['top', 'bottom', 'left', 'right']:
                    axs.spines[s].set_visible(False)
                # Remove x, y ticks
                    axs.xaxis.set_ticks_position('none')
                    axs.yaxis.set_ticks_position('none')
                # Add padding between axes and labels
                axs.xaxis.set_tick_params(pad=5)
                axs.yaxis.set_tick_params(pad=10)
                # Add x, y gridlines
                axs.grid(visible=True, color='grey', linestyle='-.',
                         linewidth=0.5, alpha=0.6)
                # Creating histogram
                N, bins, patches = axs.hist(GaitParams[PARAM], bins=bins)
                # Setting color
                fracs = ((N**(1 / 5)) / N.max())
                norm = colors.Normalize(fracs.min(), fracs.max())
                for thisfrac, thispatch in zip(fracs, patches):
                    color = plt.cm.viridis(norm(thisfrac))
                    thispatch.set_facecolor(color)
                plt.xlabel(parameter[PARAM])
                plt.ylabel('Steps (num. of steps)')
                plt.show()
            #------------------------------------------------------------------------------------------
            # Save Data to CSV
            #------------------------------------------------------------------------------------------
            for i in range(len(Step_events)):
                dataID = str(ID)
                dataStep = str(i + 1)
                datacycle = "{0:.3f}".format(float(GPE.tcycle[i]))
                dataPswing = "{0:.3f}".format(float(GPE.Pswing[i]))
                dataPTO = "{0:.3f}".format(float(GPE.PTO[i]))
                dataPHS = "{0:.3f}".format(float(GPE.PHS[i]))
                dataPFF = "{0:.3f}".format(float(GPE.PFF[i]))
                dataTO = "{0:.3f}".format(float(GPE.TO[i]))
                dataHS = "{0:.3f}".format(float(GPE.HS[i]))
                datacadence = "{0:.3f}".format(float(GPE.cadence[i]))
                dataspeed = "{0:.3f}".format(float(GPE.speed[i]))
                dataStride = "{0:.3f}".format(float(GPE.StrideLengths[i]))
                dataclearance = "{0:.3f}".format(float(GPE.clearance[i]))

                data1 = [ID, dataStep, datacycle, dataPswing, dataPTO, dataPHS, dataPFF,
                         dataTO, dataHS, datacadence, dataspeed, dataStride, dataclearance]
                writer1.writerow(data1)

            data2 = [ID, str(GPE.statistics[0]), str(GPE.statistics[1]), str(GPE.statistics[2]), str(GPE.statistics[3]), str(GPE.statistics[4]),
                     str(GPE.statistics[5]), str(GPE.statistics[6]), str(
                GPE.statistics[7]), str(GPE.statistics[8]), str(GPE.statistics[9]),
                str(GPE.statistics[10]), str(GPE.statistics[11]), str(
                GPE.statistics[12]), str(GPE.statistics[13]), str(GPE.statistics[14]),
                str(GPE.statistics[15]), str(GPE.statistics[16]), str(
                GPE.statistics[17]), str(GPE.statistics[18]), str(GPE.statistics[19]),
                str(GPE.statistics[20]), str(GPE.statistics[21]), str(
                GPE.statistics[22]), str(GPE.statistics[23]), str(GPE.statistics[24]),
                str(GPE.statistics[25]), str(GPE.statistics[26]), str(GPE.statistics[27]), str(GPE.statistics[28])]

            writer2.writerow(data2)
f1.close()
f2.close()
