import numpy as np

class Cell:

    def __init__(self, SoC, temp, randomization):

        self.SoC = SoC #0.2 + np.random.rand() *0.05 #50
        # Battery parameters
        self.Q_init = 3.4                                     
        self.Qc = 3.4 - ((3.4 * 0.05) / 2) #2Ah Capacity     
        if randomization:
            self.Qc += np.random.rand() * (3.4 * 0.05)     
        self.I_t = (1-self.SoC)*self.Qc*0.967118      # ((1-SOC/100)*Batt.Q*(1/Batt.lambda)*3600)  Batt.lambda = 0.967118
        self.I_t_Init = 0.0001
        self.I_sat = 0.0001
        self.Rint = 0.0305 - ((0.0305 * 0.05) / 2)   # Internal resistance    
        if randomization:
            self.Rint += np.random.rand() * (0.0305 * 0.05)                  
       
        # voltage status
        self.V_c = 3.7      # Nominal constant voltage                       
        self.V_cha = 0.0001      # charging voltage
        self.V_dis = 0.0001      # discharging voltage
        self.V_NL = 0.0001       # no load voltage
        self.V_exp = 0.0001      # exponential voltage
        self.V_Rint = 0.0001     # Internal resistance voltage
        self.V_sum = 0.0001      # Voltage sum
        self.V_batt = 0.0001     # Battery voltage

        # current status
        self.I_batt = 0.0001
        self.I_batt_old = 0.0001
        self.I_batt_LPF_old = 0.0001
        self.I_batt_LPF = 0.0001

        # Battery Temperature
        self.T_batt = temp
        self.T_batt_old = self.T_batt
        self.T_fun = 0.0001      # Temperature function
        self.T_cel = self.T_batt - 273.15      # Celcius degree
        self.S_on = 0       

        # Battery loss
        self.Loss = 0
        self.Cap = 2
        if randomization:
            self.Cap = self.Cap * np.random.uniform(0.95, 1.05)
        self.V_O = 10
        self.Rdisc = 5
        if randomization:
            self.Rdisc = self.Rdisc * np.random.uniform(0.95, 1.05)
        self.sem_P_fwT1 = 0.0001
        self.sem_P_fwT2 = 0.0001
        self.sem_P_fwD1 = 0.0001
        self.sem_P_fwD2 = 0.0001
        self.bat_P_c = 0.0001
        self.bat_T_c = 25
        self.sem_T_JT1a = 0.0001
        self.sem_T_JT1b = 0.0001
        self.sem_T_IT1  = 0.0001
        self.sem_T_JT2a = 0.0001
        self.sem_T_JT2b = 0.0001
        self.sem_T_IT2  = 0.0001
        self.sem_D_JD1a = 0.0001
        self.sem_D_JD1b = 0.0001
        self.sem_D_ID1  = 0.0001
        self.sem_D_JD2a = 0.0001
        self.sem_D_JD2b = 0.0001
        self.sem_D_ID2  = 0.0001