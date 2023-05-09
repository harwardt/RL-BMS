import copy
from batterymodel.battery import Battery
from gym import Env
import numpy as np
from gym import spaces

class Env(Env):

    def __init__(self, time_passed_by_step = 1, randomization = True, scenario = 0, charging = True, num_cells = 4, length_reference_run = 2600):
        self.randomization = randomization
        self.scenario = scenario

        self.discharging_current_for_evaluation = [3.95, 4.11, 3.49, 1.82, 2.72, 3.21, 4.73, 5.8, 3.12, 1.8]
        self.discharging_current_for_evaluation_counter = 1

        self.num_cells = num_cells
        self.time_passed_by_step = time_passed_by_step # unit: seconds
        self.length_reference_run = length_reference_run

        self.charging = charging
        if self.charging:
            self.I_BCS = -3
        elif self.randomization:
            self.I_BCS = np.random.rand() * 6
        else:
            self.I_BCS = self.discharging_current_for_evaluation[0]

        self.m_env = Battery(self.num_cells, self.time_passed_by_step, self.randomization, self.scenario, self.charging)
        self.done = False
        self.observation_space = spaces.Box(low=0, high=1,shape=([self.num_cells * 2]), dtype=np.float32)
        #first action: which cell to choose (needs to be discretized)
        #second action: BMS current (needs to be scaled by 2)
        self.action_space = spaces.Box(low=-1,high=1,shape=([2]),dtype=np.float32)
        self.counter = 0
        self.init_diffs()

        self.list_SoC_reward = []
        self.list_temp_reward =[]
        self.list_speed_reward = []
        
    def init_diffs(self):
        SoCs = []
        Temps = []
        for c in self.m_env.get_cells():
            SoCs.append(c.SoC)
            Temps.append(c.T_cel / 100)

        self.best_diff_SoC = self.score_for_closeness_of_values(SoCs,self.num_cells)
        self.best_diff_Temp = self.score_for_closeness_of_values(Temps,self.num_cells)

    def reset(self):
        self.m_env = Battery(self.num_cells, self.time_passed_by_step, self.randomization, self.scenario, self.charging)
        self.counter = 0
        self.done = False
        observation = np.array(self.m_env.get_cell_state(), dtype=np.float32)
        self.list_SoC_reward = []
        self.list_temp_reward =[]
        self.list_speed_reward = []
        self.init_diffs()
        if self.charging:
            self.I_BCS = -3
        elif self.randomization:
            self.I_BCS = np.random.rand() * 6
        else:
            self.I_BCS = self.discharging_current_for_evaluation[0]
        self.discharging_current_for_evaluation_counter = 1
        return observation

    def calc_reward(self):
        SoCs = []
        Temps = []
        for c in self.m_env.get_cells():
            SoCs.append(c.SoC)
            Temps.append(c.T_cel / 100)
        diff_SoC = self.score_for_closeness_of_values(SoCs,self.num_cells)
        diff_Temp = self.score_for_closeness_of_values(Temps,self.num_cells)

        SoC_balancing_reward = self.calculate_reward(diff_SoC,self.best_diff_SoC,0.0075,0.65)
        temp_balancing_reward = self.calculate_reward(diff_Temp,self.best_diff_Temp,0.0001,0.65)
        speed_reward = -1

        return SoC_balancing_reward, temp_balancing_reward, speed_reward

    
    def step(self, action):
        self.counter += 1

        self.I_BCS = self.calc_I_BCS()
        
        cell_from_action = self.calc_cell(action)

        BMS_current = self.calc_I_BMS(action)

        done = self.m_env.step(cell_from_action,BMS_current,self.I_BCS)

        observation = np.array(self.m_env.get_cell_state(), dtype=np.float32)

        SoC_balancing_reward, temp_balancing_reward, speed_reward = self.calc_reward()

        if self.counter < self.length_reference_run:
            factor_speed_reward = 0.0
        elif self.counter < 2 * self.length_reference_run:
            factor_speed_reward = (self.counter - self.length_reference_run) * (0.4 / self.length_reference_run)
        else:
            factor_speed_reward = 0.4

        reward = (0.8 * SoC_balancing_reward + 0.2 * temp_balancing_reward) * (1 - factor_speed_reward) + factor_speed_reward * speed_reward
        if BMS_current == 0:
            reward -= 0.3
        elif BMS_current < 0.0001 and BMS_current > -0.0001:
            reward -= 0.1
        elif BMS_current < 0.1 and BMS_current > -0.1:
            reward -= 0.1

        self.list_SoC_reward.append(SoC_balancing_reward)
        self.list_temp_reward.append(temp_balancing_reward)
        self.list_speed_reward.append(speed_reward)

        if reward < -1:
            reward = -1 

        if np.array(self.m_env.get_SoC_values()).max() > 0.95 or np.array(self.m_env.get_SoC_values()).min() < 0.1:
            self.done = True

        return observation, reward, self.done, {}
    
    def calc_I_BCS(self):
        if not self.charging and self.counter % (300 / self.time_passed_by_step) == 0:
            if self.randomization:
                I_BCS = np.random.rand() * 6
            else:
                I_BCS = self.discharging_current_for_evaluation[self.discharging_current_for_evaluation_counter]
                self.discharging_current_for_evaluation_counter = (self.discharging_current_for_evaluation_counter + 1) % 10

        if self.charging:
            factor_I = 1
            if np.array(self.m_env.get_SoC_values()).mean() > 0.3 and np.array(self.m_env.get_SoC_values()).mean() <= 0.45:
                factor_I = 0.85
            elif np.array(self.m_env.get_SoC_values()).mean() > 0.45 and np.array(self.m_env.get_SoC_values()).mean() <= 0.6:
                factor_I = 0.75
            elif np.array(self.m_env.get_SoC_values()).mean() > 0.6 and np.array(self.m_env.get_SoC_values()).mean() <= 0.75:
                factor_I = 0.6
            elif np.array(self.m_env.get_SoC_values()).mean() > 0.75:
                factor_I = 0.4
            I_BCS = -3 * factor_I

        return I_BCS

    def calc_cell(self,action):
        cell_float = action[0]
        if cell_float >= 1.0:
            return int(self.num_cells - 1)
        elif cell_float < -1.0:
            return int(0)
        else:
            return int(((cell_float + 1) / 2) * self.num_cells)
        
    def calc_I_BMS(self, action):
        if self.charging:
            BMS_current = ((action[1] + 1) * 3) - 2
        else:
            BMS_current = (((action[1] + 1) / 2) * 7) - 4
            if BMS_current + self.I_BCS > 6:
                BMS_current = 6 - self.I_BCS
            elif BMS_current + self.I_BCS < -1:
                BMS_current = - 1 - self.I_BCS
        return BMS_current

    def calculate_reward(self,diff,th,plv,nlv):
        if th < plv:
            th = plv
        if diff < plv:
            return 1 - (0.1 * (diff/plv))
        elif diff == plv:
            return 0.9
        elif diff < th:
            return ((th - diff) / (th - plv)) * 0.9
        elif diff == th:
            return 0
        elif diff < 0.75 - nlv:
            return (diff - th) * (- 0.8) * (1 / (0.75 - nlv - th))
        elif diff == 0.75 - nlv:
            return - 0.8
        elif diff < 0.75:
            return -0.8 - (diff - (0.75 - nlv)) * (0.2 / (nlv))
        else:
            return -1
        
    def score_for_closeness_of_values(self,values, num_values):
        mean = 0
        for v in values:
            mean += v
        mean = mean / num_values
        mean_diff_from_mean = 0
        for v in values:
            mean_diff_from_mean += abs(mean - v)
        mean_diff_from_mean /= num_values

        min = np.array(values).min()
        max = np.array(values).max()
        abs_diff_min_max = max - min

        return 0.5 * mean_diff_from_mean + 0.5 * abs_diff_min_max