import math
import random
import numpy as np
from random import shuffle

from batterymodel.cell import Cell

class Battery:
    
    def __init__(self, num_cell, Time_passed_by_step, randomization = True, scenario = 0, charging = False):
        self.num_cell = num_cell

        self.charging = charging

        self.SoC_min = 0.1
        self.SoC_max = 0.95

        self.C_inv = 0.5
        
        self.T_s = Time_passed_by_step  #2000000 / 1000 #20        #default 20 #Abtastrate vom Sinus
        self.esr =  0.1                 # ESR in Capacitor or Battery
        self.PI =  3.1415926535
        self.Rdisc = 20                # Discharging resistor
        
        #Battery parameter
        self.Qc = 3.4                 # Battery capacity (Ah)                                                   
        self.K = 0.0119             # Polarization constant, in V/Ah, or polariztion resistance, in Ohms
        self.Vc = 3.7              # Constant Voltage (V)                                                           
        self.Rint = 0.0305          # Constant internal resistance (Ohm)                                             
        self.A = 0.2711             # Exponential zone amplitude (V)
        self.B = 152.130            # Exponential zon time constant inverse (Ah)^(-1)
        self.Alpha = 1 + 3.98086e-6 # Arrhenius rate constant for the polarization resistance
        self.Beta = 1 + 0.3255e-9   # Arrhenius rate constant for the internal resistance 
        self.dE_dT = 0.00025        # Reversible voltage temperature coefficient. LiFePO4 -0.00008 / LiCoO2 -0.00025
        
        self.R_th = 25                              #0.06             # Thermal resistance K/W  **** Edited by              
        
        self.C_th = 45.585                                                               

        self.T_tau = 1/(self.R_th*self.C_th) 
        
        self.Lambda = 0.967118      # Batt Lambda

        #Storage temperatures
        self.bat_tau1_c = 1     
        self.bat_R_th_c = 20
        self.bat_A1_c = math.exp(-self.T_s/self.bat_tau1_c)
        self.bat_B1_c = 1-self.bat_A1_c
        
        #Semiconductor temperatures
        self.sem_tau1=2.4837
        self.sem_tau2=0.0911   
        self.sem_R1=0.1532
        self.sem_R2=0.6521

        self.sem_A1=math.exp(-self.T_s/self.sem_tau1)
        self.sem_B1=1-self.sem_A1

        self.sem_A2=math.exp(-self.T_s/self.sem_tau2)
        self.sem_B2=1-self.sem_A2
        self.disc_tau_inv = (1/self.Rdisc)*self.C_inv

        self.Twc = 0.0025 #5 # lowpass filter bandwidth


        #Semiconductor losses     
        self.U_CE = 2.0
        self.U_F = 2.0     
        
        self.N_on_old = 0          #Previous turned on self.cell
        
        self.t = self.T_s
        
        if randomization:

            self.SoC_init =  0.2 + (np.random.rand() * 0.4) if self.charging else 0.4 + (np.random.rand() * 0.5)
            self.SoC_range = np.random.rand() * 0.035
            self.SoC_init_cells = self.create_initial_cell_values(self.SoC_init - self.SoC_range, self.SoC_init + self.SoC_range,self.SoC_min,self.SoC_max)

            self.T_amb = 288.15 + (np.random.rand() * 20)         # Ambient temperature 

            self.temp_init = self.T_amb + 1.5
            # if no progress: then take exactly T_amb as initial value for each cell
            self.temp_range = np.random.rand() / 2
            self.temp_init_cells = self.create_initial_cell_values(self.temp_init - self.temp_range, self.temp_init + self.temp_range,self.temp_init - self.temp_range, self.temp_init + self.temp_range)

            #transfer the following two lines into cell.py
            self.C_th += (np.random.rand() - 0.5) * (self.C_th * 0.025)
            self.R_th += (np.random.rand() - 0.5) * (self.R_th * 0.025)
        else:
            #scenario is a value in [0,4]

            #set SoC values
            if charging:
                if scenario % 2 == 0:
                    starting_level = 0.2
                else:
                    starting_level = 0.4
                
                if scenario < 2:
                    self.SoC_init_cells = [x + starting_level for x in [0.01, 0, 0.002, 0.003, 0.002, 0.006, 0.005, 0.001, 0.009, 0.007, 0.004, 0.01, 0.006, 0.009, 0.002, 0.005]][:self.num_cell]
                elif scenario < 4:
                    self.SoC_init_cells = [x + starting_level for x in [0.025, 0, 0.012, 0.001, 0.019, 0.006, 0.021, 0.001, 0.006, 0.014, 0.002, 0.005, 0.023, 0.017, 0.013, 0.008]][:self.num_cell]
                else:
                    self.SoC_init_cells = [x + starting_level for x in [0, 0.05, 0.017, 0.044, 0.033, 0.024, 0.01, 0.044, 0.029, 0.001, 0.036, 0.008, 0.037, 0.034, 0.006, 0.039]][:self.num_cell]
            else:
                if scenario % 2 == 0:
                    starting_level = 0.9
                else:
                    starting_level = 0.6
                
                if scenario < 2:
                    self.SoC_init_cells = [starting_level - x for x in [0.01, 0, 0.002, 0.003, 0.002, 0.006, 0.005, 0.001, 0.009, 0.007, 0.004, 0.01, 0.006, 0.009, 0.002, 0.005]][:self.num_cell]
                elif scenario < 4:
                    self.SoC_init_cells = [starting_level - x for x in [0.025, 0, 0.012, 0.001, 0.019, 0.006, 0.021, 0.001, 0.006, 0.014, 0.002, 0.005, 0.023, 0.017, 0.013, 0.008]][:self.num_cell]
                else:
                    self.SoC_init_cells = [starting_level - x for x in [0, 0.05, 0.017, 0.044, 0.033, 0.024, 0.01, 0.044, 0.029, 0.001, 0.036, 0.008, 0.037, 0.034, 0.006, 0.039]][:self.num_cell]

            #set temperature values
            init_temp_values = [289, 305, 294, 297, 300]
            if scenario % 2 == 0:
                self.T_amb = init_temp_values[scenario]
                self.temp_init_cells = [self.T_amb + x for x in [1.15, 2, 1.583, 1.796, 1.599, 1.29, 1.396, 1.547, 1.957, 1.215, 1.656, 1.838, 1.332, 1.369, 1.811, 1.234]][:self.num_cell]
            else:
                self.T_amb = init_temp_values[scenario]
                self.temp_init_cells = [self.T_amb + x for x in [1.3, 2, 1.418, 1.572, 1.314, 1.15, 1.856, 1.44, 1.799, 1.331, 1.603, 1.679, 1.39, 1.339, 1.416, 1.577]][:self.num_cell]

        self.cell = [Cell(self.SoC_init_cells[x],self.temp_init_cells[x],True) for x in range(self.num_cell)]

    def step(self, cell, BMS_current, I_BCS):
        
        for c in self.cell:
            c.S_on = 0
        self.cell[cell].S_on = 1
        
        # Input current
        N_rez = 1/self.num_cell
        V_DC_sum = 20.0           #summed battery voltages (all self.cell inserted)
        V_Cref = N_rez * V_DC_sum #reference battery voltage

        for i in range(len(self.cell)):

            # Input battery current    Discharge: i_batt > 0, Charge: i_batt < 0. It can be sinusoidal current
            self.cell[i].I_batt = I_BCS + BMS_current * self.cell[i].S_on

            self.cell[i].I_batt_LPF = ((2 - self.Twc) * self.cell[i].I_batt_LPF + self.Twc * (self.cell[i].I_batt + self.cell[i].I_batt_old)) / (2 + self.Twc)
            self.cell[i].I_batt_old = self.cell[i].I_batt

            # Current charge (self.I_t) by input current
            if self.cell[i].I_batt_LPF > 0: 
                if self.cell[i].I_batt_LPF_old < 0:
                    self.cell[i].I_t = 0.000278 * (self.cell[i].I_t_Init + self.cell[i].I_batt)     # 1/3600
                else:
                    self.cell[i].I_t += 0.000278 * self.cell[i].I_batt
                sign_cha = 0
                sign_dis = 1
            else:
                self.cell[i].I_t += 0.000278 * self.cell[i].I_batt
                sign_cha = 1
                sign_dis = 0
        
            # extracted capacity limit
            if self.cell[i].I_t >= (self.cell[i].Qc * 0.9999):
                self.cell[i].I_sat = self.cell[i].Qc * 0.9999
            elif self.cell[i].I_t <= 0:
                self.cell[i].I_sat = 0
            else:
                self.cell[i].I_sat = self.cell[i].I_t

            self.cell[i].SoC = 1 - self.cell[i].I_sat/(self.cell[i].Qc * self.Lambda)
            
            self.cell[i].I_t_Init = self.cell[i].I_sat * 3600
            self.cell[i].I_batt_LPF_old = self.cell[i].I_batt_LPF

            self.cell[i].T_fun = math.exp(self.Alpha * ((1/self.cell[i].T_batt) - (1/(self.T_amb)))) 

            # Battery voltage components
            self.cell[i].V_cha = -1 * self.cell[i].T_fun * sign_cha * self.K * self.cell[i].I_batt_LPF * self.cell[i].Qc / (self.cell[i].I_t + 0.1*self.cell[i].Qc)  # Charging voltage
            self.cell[i].V_dis = -1 * self.cell[i].T_fun * sign_dis * self.K * self.cell[i].I_batt_LPF * self.cell[i].Qc / (self.cell[i].Qc - self.cell[i].I_sat)    # Discharging voltage
            self.cell[i].V_NL = -1 * self.cell[i].T_fun * self.K * self.cell[i].Qc * self.cell[i].I_sat / (self.cell[i].Qc - self.cell[i].I_sat)             # No Load Condition
            self.cell[i].V_Rint = -1 * self.cell[i].Rint * self.cell[i].I_batt * (math.exp(-1 * self.Beta * (1/self.cell[i].T_batt)-(1/(self.T_amb))) - 1)                  # Internal resistance
            self.cell[i].V_exp = self.A * math.exp(-1 * self.B * self.cell[i].I_sat)

            self.cell[i].V_sum = self.cell[i].V_cha + self.cell[i].V_dis + self.cell[i].V_NL + self.cell[i].V_Rint + self.cell[i].V_exp
            self.cell[i].V_batt = self.cell[i].V_c + self.cell[i].V_sum + ((self.cell[i].T_batt - self.T_amb) * self.dE_dT) - self.cell[i].Rint * self.cell[i].I_batt
            
            # Power Loss Calculation //A change is needed in the params -- de_DT change, the loss is too much// 
            self.cell[i].Loss = abs(self.cell[i].I_batt) * (self.dE_dT * self.cell[i].T_batt + abs(self.cell[i].Rint * self.cell[i].I_batt - self.cell[i].V_sum))
            
            # Temperature Calculation
            self.cell[i].T_batt = ((2 - self.T_tau) * self.cell[i].T_batt + self.T_tau * ((self.cell[i].Loss * self.R_th + self.T_amb) + self.cell[i].T_batt_old)) / (2 + self.T_tau)
            self.cell[i].T_batt_old = self.cell[i].T_batt
            self.cell[i].T_cel = self.cell[i].T_batt - 273.15

        self.t = self.t + self.T_s

        return False

    def get_cell_state(self):
        params = []
        for cell in self.cell:
            params += [cell.SoC, 
            self.scale_temp(cell.T_batt),
            ]
        observation = params
        return observation

    def get_cells(self):      
        return self.cell

    #temperature values in Kelvin need to get normalized into [0,1]
    #expects values in Kelvin between 283 K (ca. 10°C) and 323 (ca. 50°C)
    def scale_temp(self, temperature_in_K):
        return (temperature_in_K - 283) / 40

    def create_initial_cell_values(self,min_value, max_value,min_limit,max_limit):
        range_overall = max_value - min_value
        range_per_value = range_overall / (self.num_cell - 2)
        cell_values = []
        if min_value < min_limit:
            cell_values.append(min_limit)
        else:
            cell_values.append(min_value)
        for i in range(self.num_cell - 2):
            new_value = min_value + i * range_per_value + np.random.rand() * range_per_value
            if new_value < min_limit:
                new_value = min_limit
            elif new_value > max_limit:
                new_value = max_limit
            cell_values.append(new_value)
        if max_value > max_limit:
            cell_values.append(max_limit)
        else:
            cell_values.append(max_value)
        shuffle(cell_values)
        return cell_values

    def get_SoC_values(self):
        values = []
        for cell in self.cell:
            values.append(cell.SoC)
        return values

    def get_temp_values(self):
        values = []
        for cell in self.cell:
            values.append(cell.T_batt)
        return values