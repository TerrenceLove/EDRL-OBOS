import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from greedy import  evaluate
import time

class PickingStation:
    def __init__(self, id, location, process_capacity):
        self.id = id
        self.location = location
        self.process_capacity = process_capacity  
        self.idle_time = 0 


class Order:
    def __init__(self, id, product_type, quantity):
        self.id = id
        self.product_type = product_type  
        self.quantity = quantity  
        self.time = 0
        self.picked = 0  
        self.assigned_station = None  
        self.assigned_shelves = []  
        self.assigned_robots = []  


class Shelf:
    def __init__(self, id, location, product_type):
        self.id = id
        self.location = location
        self.product_type = product_type
        self.time = 0
        self.used = False
        self.robot = None


class Robot:
    def __init__(self, id, location):
        self.id = id
        self.location = location
        self.used = False
        self.time = 0
        self.shelf = None
        self.idle_time = 0 


class ShelfGenerator:

    def __init__(self, shelf_nums=504):
        self.groups = 7 
        self.rows = 9  
        self.cols = 7  
        self.shelf_nums = shelf_nums

    def generate_shelf(self):
        Shelves = []
        
        rows = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 20, 22, 23, 25, 26]
        cols = [11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42,
                43, 44]  
        shelves = []
        for row in rows:
            for col in cols:
                shelf = (col, row)
                shelves.append(shelf)

       
        grouped_shelves1 = [shelves[i:i + 28] for i in range(0, len(shelves) - 27, 28)]
        grouped_shelves2 = [grouped_shelves1[i][j:j + 4] + grouped_shelves1[i + 1][j:j + 4] for i in range(0, 17, 2) for
                            j in range(0, 28, 4)]

        for type, possitions in enumerate(grouped_shelves2):
            for possition in possitions:
                Shelves.append(Shelf(possitions.index(possition) + 1 + type * 8, possition, type + 1))

        return Shelves  


class OrderGenerator:
    def __init__(self, order_nums):
        self.shelves_type = 63
        self.order_nums = order_nums
        np.random.seed(0)  

    def generate_order(self):
        orders = []
        for i in range(self.order_nums):
            product_type = []
            quantity = []
            total_quantity = 0

         
            while len(product_type) == 0:
                num_items = np.random.poisson(5)
                num_items = max(min(num_items, 10), 1)  

                while total_quantity < num_items:
                    current_product_type = np.random.choice(range(1, self.shelves_type + 1))
                    if current_product_type not in product_type:
                        current_quantity = 0
                        while current_quantity == 0:  
                            current_quantity = np.random.poisson(2)
                        current_quantity = min(current_quantity, num_items - total_quantity)
                        product_type.append(current_product_type)
                        quantity.append(current_quantity)
                        total_quantity += current_quantity

            orders.append(Order(i + 1, product_type, quantity))
        return orders


class RobotGenerator:
    def __init__(self, robot_nums):
        self.robot_nums = robot_nums
        random.seed(0) 

    def generate_robot(self):
        robots = []
        positions = [(x, y) for x in range(11, 45) for y in range(1, 28)]  
        for i in range(self.robot_nums):
            position = random.choice(positions) 
            positions.remove(position)  
            robots.append(Robot(i + 1, position))

        return robots


class StationGenerator:
    def __init__(self, station_nums, process_capicity):
        self.station_nums = station_nums
        self.process_capicity = process_capicity

    def generate_station(self):
        stations = []
        list1 = [6, 11, 16, 21]
        for i in range(self.station_nums):
            y = list1.pop(0)
            stations.append(PickingStation(i + 1, (0, y), self.process_capicity))
        return stations


class Warehouse:

    def __init__(self, Orders, Shelves, Stations, Robots, warehouse_size=(28, 45)):
        self.warehouse_size = warehouse_size  
        self.workstations = Stations.generate_station()  
        self.robots = Robots.generate_robot()  
        self.orders = Orders.generate_order()  
        self.shelves = Shelves.generate_shelf()  
        self.state = self.initial_state()
        self.states_bf = [None] * len(self.orders) 
        self.states_af = [None] * len(self.orders)  
        self.reward = 0
        self.evaluate_list = evaluate(order_num)

    def flatten_and_convert_instances(self):
        flat_list = []
        for station in self.workstations:
            flat_list.extend([station.id, *station.location, station.process_capacity, station.idle_time])
        for shelf in self.shelves:
            flat_list.extend([shelf.id, *shelf.location, shelf.product_type, shelf.time, int(shelf.used),
                              0 if shelf.robot is None else shelf.robot.id])
        for robot in self.robots:
            flat_list.extend(
                [robot.id, *robot.location, int(robot.used), robot.time, 0 if robot.shelf is None else robot.shelf.id,
                 robot.idle_time])
        return flat_list




    def initial_state(self):
        state = []
        for order in self.orders:
            state_unit = [order.id, 0,0,0,0,0] 
            for i in range(2*len(order.product_type)):
                state_unit.extend([0]*7) 
            state_unit.append(0) #cost
            state_unit.append(order.id)
            state.extend([state_unit])
        return state 

    def manhatan(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        return abs(x1 - x2) + abs(y1 - y2)

    def reset(self):
        self.reward = 0
        self.state = self.initial_state()
        for order in self.orders:
            order.time = 0
            order.picked = 0  
            order.assigned_station = None  
            order.assigned_shelves = []  
            order.assigned_robots = []


        for shelf in self.shelves:
                if shelf.used:
                    shelf.used = False
                    shelf.time = 0
                    shelf.robot = None

        for robot in self.robots:
            if robot.used:
                robot.used = False
                robot.time = 0
                robot.shelf = None
                robot.idle_time = 0
        for station in self.workstations:
            station.process_capacity = 0
            station.idle_time = 0

        self.states_bf = [None] * len(self.orders)  

        return self.state

    def get_current_state(self):
      

        return {
            'orders': [order.__dict__.copy() for order in self.orders],
            'robots': [robot.__dict__.copy() for robot in self.robots],
            'shelves': [shelf.__dict__.copy() for shelf in self.shelves],
            'workstations': [station.__dict__.copy() for station in self.workstations],
        }
    def get_current_state1(self):
    

        return {
            'robots': [robot.__dict__.copy() for robot in self.robots],
            'shelves': [shelf.__dict__.copy() for shelf in self.shelves],
            'workstations': [station.__dict__.copy() for station in self.workstations],
        }

    def set_current_state(self, sys_state): 
      
        for i in range(len(self.orders)):
            self.orders[i].__dict__.update(sys_state['orders'][i])
        for i in range(len(self.robots)):
            self.robots[i].__dict__.update(sys_state['robots'][i])
        for i in range(len(self.shelves)):
            self.shelves[i].__dict__.update(sys_state['shelves'][i])
        for i in range(len(self.workstations)):
            self.workstations[i].__dict__.update(sys_state['workstations'][i])

    def random_assign(self, order): 
        self.states_bf[order.id - 1] = self.get_current_state()  

        if not any(station.process_capacity < 3 for station in self.workstations):
           
            min_time_orders = sorted([order for order in self.orders if order.picked == 1],
                                     key=lambda order: order.time)[:-1]
            min_time_order = random.choice(min_time_orders) 
            min_time_order.picked = 2
            min_time_order.assigned_station.idle_time += (min_time_order.time - min_time_orders[0].time)
            assigned_station = min_time_order.assigned_station

            for used_order in min_time_orders:
                if used_order.time <= min_time_order.time: 
                    used_order.picked =2
                    used_order.assigned_station.process_capacity -= 1
                    for shelf in used_order.assigned_shelves:
                        shelf.time = 0
                        shelf.used = False
                        shelf.robot.time = 0
                        shelf.robot.used = False  
                        shelf.robot.location = shelf.location
            for other_order in self.orders: 
                if other_order.picked==1:
                    other_order.time -= min_time_order.time
        else:
            stations = [station for station in self.workstations if station.process_capacity < 3]
            assigned_station = random.choice(stations)
        order.assigned_station = assigned_station

      
        assigned_shelves = []
        for product_type in order.product_type:
            might_shelives = [] 
            for shelf in self.shelves:
                if shelf.product_type == product_type and not shelf.used:
                    might_shelives.append(shelf)
            if might_shelives == []:
                min_time_shelves = sorted([shelf for shelf in self.shelves if shelf.product_type == product_type],
                                          key=lambda shelf: shelf.time)[:1]  
                for shelf in self.shelves:  
                    if shelf not in min_time_shelves and shelf.used == True:
                        for min_time_shelf in min_time_shelves:
                            shelf.time -= min_time_shelf.time
                            shelf.robot.time -= min_time_shelf.robot.time
            else:
                assigned_shelf = random.choice(might_shelives)
                assigned_shelf.used = True  
                assigned_shelf.time = self.manhatan(assigned_shelf.location, order.assigned_station.location)
                assigned_shelves.append(assigned_shelf)
        order.assigned_shelves = assigned_shelves

       
        assigned_robots = []
        for shelf in assigned_shelves:
       
            if not any(robot.used == False for robot in self.robots):
                min_time_robots = sorted(self.robots, key=lambda robot: robot.time)[:-1] 
                min_time_robot = random.choice(min_time_robots)
                for robot in min_time_robots:
                    if robot.time <= min_time_robot.time:
                        robot.time = 0
                        robot.location = robot.shelf.location
                        robot.shelf.time = 0
                        robot.shelf.used = False
                        robot.used = False



                for robot in self.robots:  
                    if  robot.used == True:
                        robot.time -= min_time_robot.time
                        robot.shelf.time -= min_time_robot.shelf.time

                assigned_robot = min_time_robot
                assigned_robot.idle_time += (min_time_robots[0].time - min_time_robot.time)

            else:
                might_robots = []
                for robot in self.robots:
                    if not robot.used:
                        might_robots.append(robot)
                assigned_robot = random.choice(might_robots)

            assigned_robot.shelf = shelf  
            shelf.robot = assigned_robot  
            assigned_robot.time = self.manhatan(assigned_robot.location,
                                                assigned_robot.shelf.location) + assigned_robot.shelf.time * 2
            assigned_robot.used = True
            assigned_robots.append(assigned_robot)

        order.assigned_robots = assigned_robots
        d1 = 0  
        d2 = 0  
        for shelf in assigned_shelves:
            d11 = self.manhatan(order.assigned_station.location, shelf.location)
            d1 += d11
            d22 = self.manhatan(shelf.location, shelf.robot.location)
            d2 += d22

        order.time = d1 + d2 + np.sum(np.array(order.quantity))

        cost = 18.75 / 4.68 * order.time / 1000 + 1.75 / 4.68 * sum([robot.time for robot in order.assigned_robots]) / 1000
        return order, self.states_bf[order.id - 1], cost


    def neighbors(self, order, sys):
     
        current_sys = sys
        self.set_current_state(current_sys)

        
        new_order, new_sys,new_time = self.random_assign(order)
        new_cost = 18.75/4.68 * order.time /1000 + 1.75/4.68 *sum([robot.time for robot in order.assigned_robots])/1000

        return new_order, new_sys, new_cost

    def backfire_algorithm(self, order):
        T = 10000.0  
        T_min = 0.00001  
        alpha = 0.9  
        current_order, current_sys, current_cost = self.greedy_algorithm3(order,0) 
        while T > T_min:
            new_order, new_sys, new_cost = self.neighbors(current_order, current_sys) 
            delta_E = new_cost - current_cost  
            if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / T): 
                current_cost = new_cost
                current_order = new_order
                current_sys = new_sys
            T = T * alpha  
        return current_cost



    def greedy_algorithm3(self, order, number):
        self.states_bf[order.id - 1] = self.get_current_state() 
        if not any(station.process_capacity < 3 for station in self.workstations):
           
            min_time_orders = sorted([order1 for order1 in self.orders if order1.picked == 1],
                                     key=lambda order1: order1.time)

            min_time_order = min_time_orders[number]  

            for used_order in min_time_orders:
                if used_order.time <= min_time_order.time:
                    used_order.picked = 2
                    used_order.assigned_station.process_capacity -= 1
                    for shelf in order.assigned_shelves:
                        shelf.used = False
                        shelf.robot.used = False  
                        shelf.robot.location = shelf.location
            for other_order in self.orders: 
                if other_order.picked == 1:
                    other_order.time -= min_time_order.time

        
        min_total_distance = float('inf')
        assigned_station = None
        assigned_shelves = []
        for station in self.workstations:
            if station.process_capacity < 3:
              
                total_distance = 0
                current_assigned_shelves = []
                for product_type in order.product_type:
                    min_distance = float('inf')
                    assigned_shelf = None
                    for shelf in self.shelves:
                        if shelf.product_type == product_type and not shelf.used:
                            distance = self.manhatan(shelf.location, station.location)
                            if distance < min_distance:
                                min_distance = distance
                                assigned_shelf = shelf
                    total_distance += min_distance
                    current_assigned_shelves.append(assigned_shelf)

              
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    assigned_shelves = current_assigned_shelves
                    assigned_station = station

        total_distance = min_total_distance
        
        for shelf in assigned_shelves:
            shelf.used = True
            shelf.time = self.manhatan(shelf.location, assigned_station.location)
       
        assigned_robots = []

        total_robot_distance = 0
        for shelf in assigned_shelves:
          
            robot_idle_time = 0
            if not any(robot.used == False for robot in self.robots):
                min_time_robots = sorted(self.robots, key=lambda robot: robot.time)[:12]
                min_robots_time = float('inf')
                assigned_min_time_robot = None
                for robot in min_time_robots:  
                    robot_time = self.manhatan(robot.shelf.location, shelf.location)
                    for other_robot in min_time_robots:
                        if other_robot.time < robot.time and other_robot != robot:
                            robot_idle_time += (
                                        robot_time - self.manhatan(min_time_robots[0].shelf.location, shelf.location))
                    if robot_idle_time + robot_time < min_robots_time:
                        min_robots_time = robot_idle_time + robot_time
                        assigned_min_time_robot = robot
                min_time_robot = assigned_min_time_robot

                for robot in min_time_robots:
                    if robot.time <= min_time_robot.time:  
                        robot.location = robot.shelf.location
                        robot.shelf.time = 0
                        robot.shelf.used = False
                        robot.used = False
                        robot.idle_time = min_time_robot.time - robot.time
                        robot.time = 0
                    else:
                        robot.time -= min_time_robot.time
                        robot.shelf.time -= min_time_robot.shelf.time

            min_robot_distance = float('inf')
            assigned_robot = None
            for robot in self.robots:
                if not robot.used:
                    distance = self.manhatan(robot.location, shelf.location)
                    if distance < min_robot_distance:
                        min_robot_distance = distance
                        assigned_robot = robot

            assigned_robots.append(assigned_robot)
            assigned_robot.used = True  
            assigned_robot.time = min_robot_distance + shelf.time * 2  
            assigned_robot.shelf = shelf
            shelf.robot = assigned_robot
            total_robot_distance += min_robot_distance  
        total_distance += total_robot_distance  

        order.time = total_distance + np.sum(np.array(order.quantity)) 
        order.picked = 1
        order.assigned_station = assigned_station
        order.assigned_shelves = assigned_shelves
        order.assigned_robots = assigned_robots
        assigned_station.process_capacity += 1  

        cost = 18.75 / 4.68 * order.time / 1000 + 1.75 / 4.68 * sum(
            [robot.time for robot in order.assigned_robots]) / 1000

        return order, self.states_bf, cost



    def step(self,action,order):
        done =False

        a,b,cost = self.greedy_algorithm3(order,action)


        reward = (self.evaluate_list[order.id-1] - order.time)
        #print(order.time,order.id,order.product_type,order.assigned_station.id,[shelf.id for shelf in order.assigned_shelves])
        
        state_unit = [order.id, order.assigned_station.id,*order.assigned_station.location,order.assigned_station.process_capacity,order.assigned_station.idle_time ]
        for shelf in order.assigned_shelves:
            state_unit.extend([shelf.id,*shelf.location,shelf.product_type ,shelf.time ,shelf.used ,shelf.robot.id])
        for robot in order.assigned_robots:
            state_unit.extend([robot.id,*robot.location ,robot.time ,robot.used ,robot.idle_time,robot.shelf.id])
        state_unit.append(18.75 / 4.68 * order.time / 1000 + 1.75 / 4.68 * sum(
            [robot.time for robot in order.assigned_robots]) / 1000)
        state_unit.append(order.id)

        self.state[order.id-1] = state_unit  #更新state
        next_state = self.convert_state(self.state)
        if order.id == 100:

            done = True

        return next_state,reward,done,cost


    def convert_state(self,state):#让state变成一维
        normal_state = []
        for order in state:
            normal_state.extend(order)

        return normal_state






# 定义参数1

LEARNING_RATE = 0.001  
GAMMA= 0.99  
MEMORY_SIZE = 50000  
BATCH_SIZE = 128 
TAU = 0.009 
EPS_START = 0.99  
EPS_END = 0.09  
EPS_DECAY = 0.9995  
SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)





# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.dropout = nn.Dropout(p=0.6)  # Dropout layer
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self,env,state_size):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPS_START
        self.device = torch.device("cpu")
        self.q_net = QNetwork( state_size,12).to(self.device)  # Input state dimension is 5, output action dimension is 4
        self.target_net = QNetwork(state_size, 12).to(self.device)
        self.update_counter = 0
        self.epoch = 0
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
        self.env = env
        self.writer = SummaryWriter(f'runs/实验二{order_num}DQN')

    def update_target_net(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(12)

        else:
            self.q_net.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state)[0]
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        # Sample a batch from the experience replay buffer
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(int)).to(self.device)

        self.q_net.eval()
        self.target_net.eval()
        
        current_q2 = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        next_q2 = self.target_net(next_states)[:].max(1)[0].detach()
       
        expected_q2 = rewards + (GAMMA * next_q2 * (1 - dones))

        self.q_net.train()
        loss2 =  F.smooth_l1_loss(current_q2, expected_q2)
        self.optimizer.zero_grad()
        loss2.backward()
        self.optimizer.step()

        self.writer.add_scalar('Loss', loss2.item(), self.epoch)

        self.epoch += 1

        self.update_counter += 1
        if self.update_counter %50==0:
            self.update_target_net()
            self.update_counter = 0  

        # Update exploration probability
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)




def train_dqn(order_num,state_size):
    start = time.time()
    env = Warehouse(OrderGenerator(order_num), ShelfGenerator(), StationGenerator(4, 0), RobotGenerator(12))
    agent = DQN(env,state_size)
    data = pd.DataFrame(columns=['total_cost'])
    for i in range(100):
        reward_total = 0
        uncover_state = env.reset()
        state = env.convert_state(uncover_state)
        total_order_time = 0
        total_cost =0
        for order in env.orders:
            action = agent.choose_action(state)
            next_state, reward, done,cost = env.step(action,order)
            reward_total += reward
            total_order_time += order.time
            state = next_state
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            if done:
                cost +=(1.75 / 4.68 * sum([robot.time for robot in env.robots]) / 1000+18.75/4.68 * sum([station.idle_time for station in env.workstations] )/1000)
            total_cost +=cost
        total_cost *= 7.5
        total_cost = '{:.2f}'.format(total_cost)
        data.loc[i+1] = [total_cost]
       
        agent.writer.add_scalar('reward', reward_total, i)
      

        print(f"第{i + 1}回合,cost:{total_cost}")
    end = time.time()
    data.loc["time"] = [end-start]
    data.to_csv(f'实验二{order_num}DQN.csv')

for order_num,state_size in [(50,2066)]:
    train_dqn(order_num,state_size)





