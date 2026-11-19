import numpy as np

class MahonyAHRS:
    # MAYHONYAHRS Madgwick's implementation of Mayhony's AHRS algorithm

    # For more information see:
    # http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms

    # Date          Author          Notes
    # 28/09/2011    SOH Madgwick    Initial release

    # Updated Antonio R. Jimenez (May/2021) to include history and log properties

    # e.g. How to use it in main script:
    # 1) Init object:     AHRS1 = MahonyAHRS('SamplePeriod', 1/sampling_rate, 'Kp', 0.5);
    # 2) Use in a loop:   AHRS1.UpdateIMU([Gyr[1:3],Acc[1:3]])
    #-----------------------------constructor----------------------
    def __init__(self,fs,num_samples):
        self.SamplePeriod=1/fs
        self.kp=0.5 # algorithm proportional gain
        self.ki=0   # algorithm integral gain
        self.Quaternion=np.array([1,0,0,0]) # output quaternion describing the Earth relative to the sensor
        self.eInt=np.array([0,0,0])
        # Angles log data
        self.angles=np.zeros((num_samples,3))
        # Quaternions log data
        self.quaternions=np.zeros((num_samples,4))
        self.quaternions[0,:]=np.array([self.Quaternion])
        # Counter
        self.i=0
    #%================================Update======================================
    def UpdateIMU(self,Gyr,Acc):

        q = self.Quaternion
        # Normalise accelerometer measurement
        if np.linalg.norm(Acc) == 0:
            return
        Acc = Acc / np.linalg.norm(Acc)

        # Estimated direction of gravity and magnetic flux
        v=np.array([2*(q[1]*q[3] - q[0]*q[2]) , 2*(q[0]*q[1] + q[2]*q[3]) , q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2])

        # Error is sum of cross product between estimated direction and measured direction of field
        e = self.cross(Acc,v)

        if self.ki > 0:
            self.eInt = self.eInt + e*self.SamplePeriod
        else:
            self.eInt = np.array([0,0,0])

        # Apply feedback terms
        Gyr = Gyr + self.kp*e + self.ki*self.eInt

        # Compute rate of change of quaternion
        qDot = 0.5 *self.quaternProd(q,np.array([0,Gyr[0],Gyr[1],Gyr[2]]))

        # Integrate to yield quaternion
        q = q + qDot * self.SamplePeriod
        self.Quaternion = q / np.linalg.norm(q)

        # Safe EULER and QUATERNIONS data
        euler=self.quatern2euler(self.quaternConj(self.Quaternion)) * (180/np.pi)
        self.angles[self.i,:]=euler
        self.quaternions[self.i,:]=self.Quaternion

        self.i=self.i+1


    def quatern2euler(self,q):
        R=np.zeros((3,3))

        R[0,0] = 2*q[0]**2 - 1 + 2*q[1]**2
        R[1,0] = 2*(q[1]*q[2] - q[0]*q[3])
        R[2,0] = 2*(q[1]*q[3] + q[0]*q[2])
        R[2,1] = 2*(q[2]*q[3] - q[0]*q[1])
        R[2,2] = 2*q[0]**2 - 1 + 2*q[3]**2

        phi = np.arctan2(R[2,1],R[2,2])
        theta = -np.arctan(R[2,0] / np.sqrt(1-R[2,0]**2))
        psi = np.arctan2(R[1,0],R[0,0])

        euler=np.array([[phi,theta,psi]])
        return euler

    def quaternProd(self,a,b):

        ab1 = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
        ab2 = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
        ab3 = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
        ab4 = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]

        ab=np.array([ab1,ab2,ab3,ab4])

        return ab

    def quaternConj(self,q):

        qConj=[q[0],-q[1],-q[2],-q[3]]
        return qConj

    def cross(self,a,b):

        ab1=a[1]*b[2]-b[1]*a[2]
        ab2=-(a[0]*b[2]-b[0]*a[2])
        ab3=a[0]*b[1]-b[0]*a[1]

        ab=np.array([ab1,ab2,ab3])

        return ab


