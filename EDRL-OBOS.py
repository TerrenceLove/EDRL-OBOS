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
        self.process_capacity = process_capacity  # 订单处理数量 初始为0
        self.idle_time = 0  # 拣货台闲置时间


class Order:
    def __init__(self, id, product_type, quantity):
        self.id = id
        self.product_type = product_type  # 列表形式比如需要2,3,12类的货物则为[2,3,12]
        self.quantity = quantity  # 与货物类型相对于，如上例有3类，数量分别为1,2,3，这里则为[1,2,3]
        self.time = 0
        self.picked = 0  # 0表示还未分配；1表示正在拣选；2表示已拣选
        self.assigned_station = None  # 分配的拣货台
        self.assigned_shelves = []  # 代表分配的货架 如[2,64,15]
        self.assigned_robots = []  # 代表分配的机器人,如[1,3,6] 每个货架对应一个机器人


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
        self.idle_time = 0 #空闲时间


class ShelfGenerator:
    # 货架位置生成
    # 货架编号从1开始
    def __init__(self, shelf_nums=504):
        self.groups = 7  # 7*9个类
        self.rows = 9  # 货架区域的列数
        self.cols = 7  # 货架区域的行数
        self.shelf_nums = shelf_nums

    def generate_shelf(self):
        Shelves = []
        # 生成货架的坐标，每一行有28个货架,共有18行
        rows = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 20, 22, 23, 25, 26]  # 纵坐标 列
        cols = [11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42,
                43, 44]  # 横坐标 行
        shelves = []
        for row in rows:
            for col in cols:
                shelf = (col, row)
                shelves.append(shelf)

        # 将所有坐标分组到小列表中
        grouped_shelves1 = [shelves[i:i + 28] for i in range(0, len(shelves) - 27, 28)]
        grouped_shelves2 = [grouped_shelves1[i][j:j + 4] + grouped_shelves1[i + 1][j:j + 4] for i in range(0, 17, 2) for
                            j in range(0, 28, 4)]

        for type, possitions in enumerate(grouped_shelves2):
            for possition in possitions:
                Shelves.append(Shelf(possitions.index(possition) + 1 + type * 8, possition, type + 1))

        return Shelves  # 调用方法，Shelves[3].location 3号货架的位置


class OrderGenerator:
    def __init__(self, order_nums):
        self.shelves_type = 63
        self.order_nums = order_nums
        np.random.seed(0)  # 设置随机数生成器的种子

    def generate_order(self):
        orders = []
        for i in range(self.order_nums):
            product_type = []
            quantity = []
            total_quantity = 0

            # 确保每个订单至少有一件商品
            while len(product_type) == 0:
                num_items = np.random.poisson(5)
                num_items = max(min(num_items, 10), 1)  # 限制货物数量在1到10之间

                while total_quantity < num_items:
                    current_product_type = np.random.choice(range(1, self.shelves_type + 1))
                    if current_product_type not in product_type:
                        current_quantity = 0
                        while current_quantity == 0:  # 确保每种商品至少有1个
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
        random.seed(0)  # 设置随机数生成器的种子

    def generate_robot(self):
        robots = []
        positions = [(x, y) for x in range(11, 45) for y in range(1, 28)]  # 创建所有可能的位置列表
        for i in range(self.robot_nums):
            position = random.choice(positions)  # 随机选择一个位置
            positions.remove(position)  # 从列表中移除已选的位置
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
    # 暂定的仓库环境为横向向右为0-44；纵向向上为0-27
    # 货架区域为(11-44,1-26)
    # 所有拣货台的单个货物处理时间相同
    # 假设机器人每走一步的时间花费为1
    # 假设每个货架上的货物都是无限的
    # 所有订单、机器人、货架、拣选台的编号都是从1开始
    # 货物种类的编号是从0开始的
    def __init__(self, Orders, Shelves, Stations, Robots, warehouse_size=(28, 45)):
        self.warehouse_size = warehouse_size  # 仓库的大小为横*纵
        self.workstations = Stations.generate_station()  # 生成拣货站
        self.robots = Robots.generate_robot()  # 生成机器人
        self.orders = Orders.generate_order()  # 生成订单
        self.shelves = Shelves.generate_shelf()  # 生成货架
        self.state = self.initial_state()
        self.states_bf = [None] * len(self.orders)  # 保存每个订单分配前系统状态
        self.states_af = [None] * len(self.orders)  # 保存每个订单分配后系统状态
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
            state_unit = [order.id, 0,0,0,0,0] #拣货台的4个属性
            for i in range(2*len(order.product_type)):
                state_unit.extend([0]*7) #机器人和货架的6个属性 坐标占2个位置
            state_unit.append(0) #cost
            state_unit.append(order.id)
            state.extend([state_unit])
        return state #此时的state还不是一维的而是[[1, 0, 0, 0, 0, 0, 0, 0, ,0,1], [2, 0, 0, 0, 0, 0,,0 2], [3, 0, 0, 0, 0, 0, 0, 0, 0,3]]这种形式

    def manhatan(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        return abs(x1 - x2) + abs(y1 - y2)

    def reset(self):
        self.reward = 0
        self.state = self.initial_state()
        for order in self.orders:
            order.time = 0
            order.picked = 0  # 0表示还未分配；1表示正在拣选；2表示已拣选
            order.assigned_station = None  # 分配的拣货台
            order.assigned_shelves = []  # 代表分配的货架 如[2,64,15]
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

        self.states_bf = [None] * len(self.orders)  # 保存每个订单分配前系统状态

        return self.state

    def get_current_state(self):
        # 获取当前的状态

        return {
            'orders': [order.__dict__.copy() for order in self.orders],
            'robots': [robot.__dict__.copy() for robot in self.robots],
            'shelves': [shelf.__dict__.copy() for shelf in self.shelves],
            'workstations': [station.__dict__.copy() for station in self.workstations],
        }
    def get_current_state1(self):
        # 获取当前的状态

        return {
            'robots': [robot.__dict__.copy() for robot in self.robots],
            'shelves': [shelf.__dict__.copy() for shelf in self.shelves],
            'workstations': [station.__dict__.copy() for station in self.workstations],
        }

    def set_current_state(self, sys_state):  # 设置系统状态
        # 设置当前的状态
        for i in range(len(self.orders)):
            self.orders[i].__dict__.update(sys_state['orders'][i])
        for i in range(len(self.robots)):
            self.robots[i].__dict__.update(sys_state['robots'][i])
        for i in range(len(self.shelves)):
            self.shelves[i].__dict__.update(sys_state['shelves'][i])
        for i in range(len(self.workstations)):
            self.workstations[i].__dict__.update(sys_state['workstations'][i])

    def random_assign(self, order):  # 对每一个订单，生成一个随机的方案
        self.states_bf[order.id - 1] = self.get_current_state()  # 储存每个订单分配前的系统状态
        # 分配拣货台

        if not any(station.process_capacity < 3 for station in self.workstations):
            # 找到 time 属性最小的订单，将其 picked 属性设置为12
            # 目前定义的释放一个订单，但是可以更改为释放好几个订单进行选择
            min_time_orders = sorted([order for order in self.orders if order.picked == 1],
                                     key=lambda order: order.time)[:-1]
            min_time_order = random.choice(min_time_orders)  # 从正在拣货的订单当中随机选取一个释放,并不是最小时间的顶单
            min_time_order.picked = 2
            min_time_order.assigned_station.idle_time += (min_time_order.time - min_time_orders[0].time)
            assigned_station = min_time_order.assigned_station

            for used_order in min_time_orders:
                if used_order.time <= min_time_order.time: #释放时间小于等于已选择订单的订单
                    used_order.picked =2
                    used_order.assigned_station.process_capacity -= 1
                    for shelf in used_order.assigned_shelves:
                        shelf.time = 0
                        shelf.used = False
                        shelf.robot.time = 0
                        shelf.robot.used = False  # 释放还在使用的货架和机器人
                        shelf.robot.location = shelf.location
            for other_order in self.orders: #更新其他的订单的时间
                if other_order.picked==1:
                    other_order.time -= min_time_order.time
        else:
            stations = [station for station in self.workstations if station.process_capacity < 3]
            assigned_station = random.choice(stations)
        order.assigned_station = assigned_station

        # 分配货架
        assigned_shelves = []
        for product_type in order.product_type:
            might_shelives = []  # 可以选的货架
            for shelf in self.shelves:
                if shelf.product_type == product_type and not shelf.used:
                    might_shelives.append(shelf)
            if might_shelives == []:
                min_time_shelves = sorted([shelf for shelf in self.shelves if shelf.product_type == product_type],
                                          key=lambda shelf: shelf.time)[:1]  # 只是释放一个货架
                for shelf in self.shelves:  # 更新其余货架的属性
                    if shelf not in min_time_shelves and shelf.used == True:
                        for min_time_shelf in min_time_shelves:
                            shelf.time -= min_time_shelf.time
                            shelf.robot.time -= min_time_shelf.robot.time
            else:
                assigned_shelf = random.choice(might_shelives)
                assigned_shelf.used = True  # 更新货架属性
                assigned_shelf.time = self.manhatan(assigned_shelf.location, order.assigned_station.location)
                assigned_shelves.append(assigned_shelf)
        order.assigned_shelves = assigned_shelves

        # 分配机器人
        assigned_robots = []
        for shelf in assigned_shelves:
            # 检查是否有可用机器人
            if not any(robot.used == False for robot in self.robots):
                min_time_robots = sorted(self.robots, key=lambda robot: robot.time)[:-1]  # 只是释放一个机器人
                min_time_robot = random.choice(min_time_robots)
                for robot in min_time_robots:
                    if robot.time <= min_time_robot.time:
                        robot.time = 0
                        robot.location = robot.shelf.location
                        robot.shelf.time = 0
                        robot.shelf.used = False
                        robot.used = False



                for robot in self.robots:  # 更新其余机器人的属性
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

            assigned_robot.shelf = shelf  # 机器人匹配的货架
            shelf.robot = assigned_robot  # 货架分配的机器人
            assigned_robot.time = self.manhatan(assigned_robot.location,
                                                assigned_robot.shelf.location) + assigned_robot.shelf.time * 2
            assigned_robot.used = True
            assigned_robots.append(assigned_robot)

        order.assigned_robots = assigned_robots
        d1 = 0  # 货架到拣货站的总距离
        d2 = 0  # 货架到机器人的总距离
        for shelf in assigned_shelves:
            d11 = self.manhatan(order.assigned_station.location, shelf.location)
            d1 += d11
            d22 = self.manhatan(shelf.location, shelf.robot.location)
            d2 += d22

        order.time = d1 + d2 + np.sum(np.array(order.quantity))

        cost = 18.75 / 4.68 * order.time / 1000 + 1.75 / 4.68 * sum([robot.time for robot in order.assigned_robots]) / 1000
        return order, self.states_bf[order.id - 1], cost


    def neighbors(self, order, sys):
        # 重新分配订单
        # 先恢复到分配前的状态
        current_sys = sys
        self.set_current_state(current_sys)

        # 选择重新分配的方案：1.重新分配拣货台，2.重新分配货架，3.重新分配机器人
        new_order, new_sys,new_time = self.random_assign(order)
        new_cost = 18.75/4.68 * order.time /1000 + 1.75/4.68 *sum([robot.time for robot in order.assigned_robots])/1000

        return new_order, new_sys, new_cost

    def backfire_algorithm(self, order):
        T = 10000.0  # 初始温度
        T_min = 0.00001  # 最小温度
        alpha = 0.9  # 降温系数
        current_order, current_sys, current_cost = self.greedy_algorithm3(order,0)  # 初始状态
        while T > T_min:
            new_order, new_sys, new_cost = self.neighbors(current_order, current_sys)  # 生成邻居状态
            delta_E = new_cost - current_cost  # 计算能量差
            if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / T):  # Metropolis准则
                current_cost = new_cost
                current_order = new_order
                current_sys = new_sys
            T = T * alpha  # 降温
        return current_cost



    def greedy_algorithm3(self, order, number):  # 最短拣货台处理原则 + 随机机器人原则
        self.states_bf[order.id - 1] = self.get_current_state()  # 储存每个订单分配前的状态
        if not any(station.process_capacity < 3 for station in self.workstations):
            # 找到 time 属性最小的订单，将其 picked 属性设置为12
            # 目前定义的释放一个订单，但是可以更改为释放好几个订单进行选择
            min_time_orders = sorted([order1 for order1 in self.orders if order1.picked == 1],
                                     key=lambda order1: order1.time)

            min_time_order = min_time_orders[number]  # 从正在拣货的订单当中随机选取一个释放

            for used_order in min_time_orders:
                if used_order.time <= min_time_order.time:  # 释放时间小于等于已选择订单的订单
                    used_order.picked = 2
                    used_order.assigned_station.process_capacity -= 1
                    for shelf in order.assigned_shelves:
                        shelf.used = False
                        shelf.robot.used = False  # 释放还在使用的货架和机器人
                        shelf.robot.location = shelf.location
            for other_order in self.orders:  # 更新其他的订单的时间
                if other_order.picked == 1:
                    other_order.time -= min_time_order.time

        # 如果有可用的拣货台，就正常执行给订单分配拣货台和货架，再将货架分配给机器人
        min_total_distance = float('inf')
        assigned_station = None
        assigned_shelves = []
        for station in self.workstations:
            if station.process_capacity < 3:
                # 分配货架
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

                # 分配拣货台
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    assigned_shelves = current_assigned_shelves
                    assigned_station = station

        total_distance = min_total_distance
        # 更新分配的货架的属性
        for shelf in assigned_shelves:
            shelf.used = True
            shelf.time = self.manhatan(shelf.location, assigned_station.location)
        # 分配货架给机器人
        assigned_robots = []

        total_robot_distance = 0
        for shelf in assigned_shelves:
            # 检查是否有可用机器人
            robot_idle_time = 0
            if not any(robot.used == False for robot in self.robots):
                min_time_robots = sorted(self.robots, key=lambda robot: robot.time)[:12]
                min_robots_time = float('inf')
                assigned_min_time_robot = None
                for robot in min_time_robots:  # 选择到货架距离与产生空闲时间最短的机器人
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
                    if robot.time <= min_time_robot.time:  # 释放机器人
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
            assigned_robot.used = True  # 更新机器人的使用状态
            assigned_robot.time = min_robot_distance + shelf.time * 2  # 更新机器人的time
            assigned_robot.shelf = shelf
            shelf.robot = assigned_robot
            total_robot_distance += min_robot_distance  # 所有机器人到货架的距离之和
        total_distance += total_robot_distance  # 加入机器人到货架的曼哈顿距离

        order.time = total_distance + np.sum(np.array(order.quantity))  # 设置order.time为总距离+每个订单物品数量
        order.picked = 1
        order.assigned_station = assigned_station
        order.assigned_shelves = assigned_shelves
        order.assigned_robots = assigned_robots
        assigned_station.process_capacity += 1  # 更新station的process_capacity

        cost = 18.75 / 4.68 * order.time / 1000 + 1.75 / 4.68 * sum(
            [robot.time for robot in order.assigned_robots]) / 1000

        return order, self.states_bf, cost



    def step(self,action,order):
        done =False

        a,b,cost = self.greedy_algorithm3(order,action)


        reward = (self.evaluate_list[order.id-1] - order.time)
        #print(order.time,order.id,order.product_type,order.assigned_station.id,[shelf.id for shelf in order.assigned_shelves])
        #处理state，替换state中的这个订单的属性
        #更新每个订单的分配
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

LEARNING_RATE = 0.001  # 学习率
GAMMA= 0.99  # 折扣因子
MEMORY_SIZE = 50000  # 经验回放池的大小
BATCH_SIZE = 128 # 每次训练的批次大小
TAU = 0.009  # 目标网络的更新速率
EPS_START = 0.99  # 开始的探索概率
EPS_END = 0.09  # 最小的探索概率
EPS_DECAY = 0.9995  # 探索概率衰减率
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
        # 计算当前的Q值
        current_q2 = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 计算下一步的Q值
        next_q2 = self.target_net(next_states)[:].max(1)[0].detach()
        # 计算下一步的期望
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
            self.update_counter = 0  # 重置更新计数器

        # Update exploration probability
        self.epsilon = max(EPS_END, EPS_DECAY * self.epsilon)

#动作设定成随机


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
        #print(f"第{i+1}回合,total_reward:{reward_total}")
        agent.writer.add_scalar('reward', reward_total, i)
        #agent.writer.add_scalar('total_time', total_order_time, i)
        #print(f"第{i + 1}回合,total_time:{total_order_time}")

        print(f"第{i + 1}回合,cost:{total_cost}")
    end = time.time()
    data.loc["time"] = [end-start]
    data.to_csv(f'实验二{order_num}DQN.csv')

for order_num,state_size in [(50,2066)]:
    train_dqn(order_num,state_size)


#time对应的单位是m
#在Optimization models for scheduling operations in robotic mobile fulﬁllment systems当中假设拣货员的工作成本为18.75欧元/小时即；机器人的成本为1.75欧元/公里；机器人速度为1.3m/s即 4.68 公里/小时
""""
贪心算法：
1.495.58
退火算法：
466.34
473.16
472.13
473.17
469.56
470.37
472.07
472.52
470.76
472.06
473.06
467.62
471.66
469.50
目前DQN最好的结果：
第1回合,cost:392.62
第2回合,cost:387.63
第3回合,cost:396.09
第4回合,cost:390.80
"""
"""

解决经验回放的问题

"""

#解决gr的随机性，固定下来

#TODO: 将每个订单分配的拣货台、货架、机器人信息写入到状态当中


