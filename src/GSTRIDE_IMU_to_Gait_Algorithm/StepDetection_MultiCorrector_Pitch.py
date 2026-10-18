import numpy as np


def StepDetection_MultiCorrector_Pitch(angles, Gyr, orientation, fs):
    num_samples = len(Gyr)
    #-----------------------------------------------------------------------------------------------------------------
    # Adjustable parameters
    #-----------------------------------------------------------------------------------------------------------------
    Th1 = 0.5
    Th2 = 2
    window = int(np.floor(fs / 20))
    window_still = int(np.floor(1.5 * fs))
    #-----------------------------------------------------------------------------------------------------------------
    # 1: Still
    #-----------------------------------------------------------------------------------------------------------------
    M_Gyr = np.sqrt(Gyr[:, 0]**2 + Gyr[:, 1]**2 + Gyr[:, 2]**2)
    Still = np.zeros(num_samples)
    newStill = np.array([])
    endStill = np.array([])

    for i in range(window_still, num_samples - 1):
        if len(np.argwhere(M_Gyr[int(i - window_still):i] < 0.2)) == window_still:
            Still[i] = 1
        else:
            Still[i] = 0
        if Still[i - 1] == 0 and Still[i] == 1:
            newStill = np.concatenate((newStill, i), axis=None)
        if Still[i - 1] == 1 and Still[i] == 0:
            endStill = np.concatenate((endStill, i), axis=None)
    #-----------------------------------------------------------------------------------------------------------------
    # 2: Movement Detection
    #-----------------------------------------------------------------------------------------------------------------
    M_Gyr2 = M_Gyr**2
    move = np.zeros(num_samples)
    newMove = np.array([])
    endMove = np.array([])
    for i in range(window, num_samples - window - 1):
        if Still[i] == 0:
            if len(np.argwhere(M_Gyr2[i:i + window] > Th2)) == window or len(np.argwhere(M_Gyr2[i - window:i] > Th2)) == window:
                move[i] = Th2
        if move[i - 1] == 0 and move[i] == Th2:
            newMove = np.concatenate((newMove, i), axis=None)
        if move[i - 1] == Th2 and move[i] == 0:
            endMove = np.concatenate((endMove, i), axis=None)

    distMove = endMove - newMove
    midMove = np.floor((newMove + endMove) / 2)
    #-----------------------------------------------------------------------------------------------------------------
    # 3: Movement Correction
    #-----------------------------------------------------------------------------------------------------------------
    median_dist = np.floor(np.median(distMove))
    l = len(distMove)
    for i in range(l):
        if distMove[i] <= median_dist / 3:
            move[int(newMove[i]):int(endMove[i])] = 0
            newMove[i] = np.nan
            endMove[i] = np.nan
            midMove[i] = np.nan
            distMove[i] = np.nan
    newMove = newMove[~np.isnan(newMove)]
    endMove = endMove[~np.isnan(endMove)]
    midMove = midMove[~np.isnan(midMove)]
    distMove = distMove[~np.isnan(distMove)]

    distIntraMove = np.zeros(len(midMove) - 1)
    l = len(distIntraMove) - 1
    for i in range(l):
        distIntraMove[i] = midMove[i + 1] - midMove[i]
    sumDistIntraMove = 0
    cont = 0
    l = len(distIntraMove)
    for i in range(l):
        if distIntraMove[i] < window_still:
            sumDistIntraMove = sumDistIntraMove + distIntraMove[i]
            cont = cont + 1
    medDistIntraMove = sumDistIntraMove / cont
    i = 1
    l = len(midMove)
    while i < l:
        if (midMove[i] - midMove[i - 1]) < (0.6 * medDistIntraMove):
            if distMove[i] > distMove[i - 1]:
                move[int(newMove[i - 1]):int(endMove[i - 1])] = 0
                newMove[i - 1] = np.nan
                endMove[i - 1] = np.nan
                midMove[i - 1] = np.nan
                distMove[i - 1] = np.nan
            else:
                move[int(newMove[i]):int(endMove[i])] = 0
                newMove[i] = np.nan
                endMove[i] = np.nan
                midMove[i] = np.nan
                distMove[i] = np.nan
            newMove = newMove[~np.isnan(newMove)]
            endMove = endMove[~np.isnan(endMove)]
            midMove = midMove[~np.isnan(midMove)]
            distMove = distMove[~np.isnan(distMove)]
            i = i - 1
            l = l - 1
        i = i + 1
    #-----------------------------------------------------------------------------------------------------------------
    # 4: Foot Flat (FF) Detection
    #-----------------------------------------------------------------------------------------------------------------
    FF = np.zeros(num_samples)
    newFF = np.array([])
    endFF = np.array([])
    for i in range(window, num_samples - window):
        if Still[i] == 0:
            if (np.mean(M_Gyr2[i:i + window]) < Th1) == True or (np.mean(M_Gyr2[i - window:i]) < Th1):
                FF[i] = Th1
        if FF[i - 1] == 0 and FF[i] > 0:
            newFF = np.concatenate((newFF, i), axis=None)
        if FF[i - 1] > 0 and FF[i] == 0:
            endFF = np.concatenate((endFF, i - 1), axis=None)
        if FF[i - 1] > 0 and FF[i] > 0 and i == num_samples - window - 1:
            endFF = np.concatenate((endFF, i - 1), axis=None)
    distFF = endFF - newFF
    midFF = np.floor((newFF + endFF) / 2)
    #-----------------------------------------------------------------------------------------------------------------
    # 5: Foot Flat (FF) Corrections
    #-----------------------------------------------------------------------------------------------------------------
    l = len(newFF)
    for i in range(l):
        # If non previous activity has been detected, a step/FF can not be possible
        cont_activity = 0
        for j in range(int(newFF[i] - window_still), int(newFF[i])):
            if j > 0 and move[j]:
                cont_activity = cont_activity + 1
        if cont_activity == 0:
            FF[int(newFF[i]) - 1:int(endFF[i]) + 1] = 0
            newFF[i] = np.nan
            endFF[i] = np.nan
            midFF[i] = np.nan
            distFF[i] = np.nan

    newFF = newFF[~np.isnan(newFF)]
    endFF = endFF[~np.isnan(endFF)]
    midFF = midFF[~np.isnan(midFF)]
    distFF = distFF[~np.isnan(distFF)]

    # Foot Flats too short are discriminated (0.3*median_FootFlat)
    median_dist = np.floor(np.median(distFF))
    l = len(distFF)
    for i in range(l):
        if distFF[i] <= np.floor(0.3 * median_dist):
            FF[int(newFF[i]) - 1:int(endFF[i]) + 1] = 0
            newFF[i] = np.nan
            endFF[i] = np.nan
            midFF[i] = np.nan
            distFF[i] = np.nan
    newFF = newFF[~np.isnan(newFF)]
    endFF = endFF[~np.isnan(endFF)]
    midFF = midFF[~np.isnan(midFF)]
    distFF = distFF[~np.isnan(distFF)]

    l = len(newFF)
    i = 1
    # Between to activities there can not be more than one Foot Flat
    while i < l:
        countextraFF = 0
        for j in range(int(endFF[i - 1]), int(newFF[i])):
            if move[j] > 0:
                countextraFF = countextraFF + 1
        if countextraFF == 0:
            if distFF[i] < distFF[i - 1]:
                FF[int(newFF[i]) - 1:int(endFF[i]) + 1] = 0
                newFF[i] = np.nan
                endFF[i] = np.nan
                midFF[i] = np.nan
                distFF[i] = np.nan
            else:
                FF[int(newFF[i - 1]) - 1:int(endFF[i - 1]) + 1] = 0
                newFF[i - 1] = np.nan
                endFF[i - 1] = np.nan
                midFF[i - 1] = np.nan
                distFF[i - 1] = np.nan
            newFF = newFF[~np.isnan(newFF)]
            endFF = endFF[~np.isnan(endFF)]
            midFF = midFF[~np.isnan(midFF)]
            distFF = distFF[~np.isnan(distFF)]
            l = l - 1
            i = i - 1
        i = i + 1

    #-----------------------------------------------------------------------------------------------------------------
    # 6: Pitch for Toe off (TO) &  Heel Strike (HS)
    #-----------------------------------------------------------------------------------------------------------------
    Pitch = angles[:, orientation]
    l = len(newFF)
    k = 0
    TO = np.zeros(l)
    TO_i = np.zeros(l)
    HS = np.zeros(l)
    HS_i = np.zeros(l)
    Phases = np.zeros((l, 5))
    Pitch0 = np.zeros(l)
    Pitch0_i = np.zeros(l)

    for i in range(l):
        if i == 0 or (newStill[k] >= endFF[i - 1] and newStill[k] < newFF[i]):
            TO[i] = Pitch[int(endStill[k])]
            TO_i[i] = endStill[k]
            # Toe off (TO) &  Heel Strike (HS) after a pause period
            for j in range(int(endStill[k]), int(newFF[i])):
                if Pitch[j] < Pitch[j - 1] and Pitch[j] < Pitch[j + 1] and Pitch[j] < TO[i]:
                    TO[i] = Pitch[j]
                    TO_i[i] = j
            HS[i] = TO[i]
            HS_i[i] = TO_i[i]
            for j in range(int(TO_i[i]), int(newFF[i])):
                if Pitch[j] >= Pitch[j - 1] and Pitch[j] > HS[i]:
                    HS[i] = Pitch[j]
                    HS_i[i] = j

            Pitch0[i] = HS[i]
            Pitch0_i[i] = HS_i[i]
            for j in range(int(HS_i[i]), int(newFF[i])):
                if Pitch[j] <= Pitch0[i]:
                    Pitch0[i] = Pitch[j]
                    Pitch0_i[i] = j

            Phases[i, 0] = endStill[k]
            Phases[i, 1] = TO_i[i]
            Phases[i, 2] = HS_i[i]
            Phases[i, 3] = newFF[i]
            Phases[i, 4] = endFF[i]

            while k < len(newStill) - 1 and newStill[k] < newFF[i]:
                k = k + 1

        else:
            # Toe off (TO) &  Heel Strike (HS) in continious walk
            TO[i] = Pitch[int(endFF[i - 1])]
            TO_i[i] = endFF[i - 1]
            for j in range(int(endFF[i - 1]), int(newFF[i])):
                if Pitch[j] < Pitch[j - 1] and Pitch[j] < Pitch[j + 1] and Pitch[j] < TO[i]:
                    TO[i] = Pitch[j]
                    TO_i[i] = j
            HS[i] = TO[i]
            HS_i[i] = TO_i[i]
            for j in range(int(TO_i[i]), int(newFF[i])):
                if Pitch[j] >= Pitch[j - 1] and Pitch[j] > HS[i]:
                    HS[i] = Pitch[j]
                    HS_i[i] = j

            Pitch0[i] = HS[i]
            Pitch0_i[i] = HS_i[i]
            for j in range(int(HS_i[i]), int(newFF[i])):
                if Pitch[j] <= Pitch0[i]:
                    Pitch0[i] = Pitch[j]
                    Pitch0_i[i] = j

            Phases[i, 0] = endFF[i - 1]
            Phases[i, 1] = TO_i[i]
            Phases[i, 2] = HS_i[i]
            Phases[i, 3] = newFF[i]
            Phases[i, 4] = endFF[i]

    return Still, M_Gyr, newStill, endStill, FF, newFF, midFF, endFF, HS, HS_i, TO, TO_i, Phases, Pitch0
