import numpy as np
import random
import math
import copy



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
        self.idle_time = 0 #空闲时间


class ShelfGenerator:
    # 货架位置生成
    # 货架编号从1开始
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

            # 确保每个订单至少有一件商品
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
        self.Orders = Orders
        self.Shelves = Shelves
        self.Stations = Stations
        self.Robots = Robots
        self.warehouse_size = warehouse_size  
        self.workstations = self.Stations.generate_station() 
        self.robots = self.Robots.generate_robot()  
        self.orders = self.Orders.generate_order()  
        self.shelves = self.Shelves.generate_shelf()  
        self.state = self.initial_state()
        self.states_before_bf = [None] * len(self.orders)  
        self.states_before_gr = [None] * len(self.orders)  
        self.reward = 0
        # store initial states
        self.initial_workstations = copy.deepcopy(self.workstations)
        self.initial_robots = copy.deepcopy(self.robots)
        self.initial_orders = copy.deepcopy(self.orders)
        self.initial_shelves = copy.deepcopy(self.shelves)



    def initial_state(self):
        state = []
        for order in self.orders:
            state_unit = [order.id, 0]
            for i in range(2*len(order.product_type)):
                state_unit.append(0)
            state_unit.append(0)  #cost
            state_unit.append(order.id)
            state.extend([state_unit])
        return state 

    def manhatan(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        return abs(x1 - x2) + abs(y1 - y2)

    def reset(self):

        self.reward = 0
        self.workstations = copy.deepcopy(self.initial_workstations)
        self.robots = copy.deepcopy(self.initial_robots)
        self.orders = copy.deepcopy(self.initial_orders)
        self.shelves = copy.deepcopy(self.initial_shelves)
        self.state = self.initial_state()
        self.states_before_bf = [None] * len(self.orders)

        return self.state

    def get_current_state(self):
        

        return {
            'orders': [order.__dict__.copy() for order in self.orders],
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
        self.states_before_bf[order.id - 1] = self.get_current_state()  #
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
        return order, self.states_before_bf, cost


    def neighbors(self, orders, sys):
       
        current_sys = sys
        self.set_current_state(current_sys[orders[0].id-1])
        
        new_cost = 0
        for order in orders:
            current_order, current_sys, cost = self.random_assign(order)  
            new_cost += cost

        return orders, new_cost

    def backfire_algorithm(self, orders):
        T = 10000.0  
        T_min = 0.00001  
        alpha = 0.9  
        current_cost = 0
        for order in orders:
            current_order, current_sys, cost = self.greedy_algorithm3(order)  
            current_cost += cost
        while T > T_min:
            new_order,  new_cost = self.neighbors(orders, self.states_before_bf)  #
            delta_E = new_cost - current_cost  #
            if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / T):  
                current_cost = new_cost

            T = T * alpha  # 降温
        return current_cost


    def greedy_algorithm3(self, order):  # 最短拣货台处理原则 + 随机机器人原则
        self.states_before_bf[order.id - 1] = self.get_current_state() 
        if not any(station.process_capacity < 3 for station in self.workstations):
            
            min_time_orders = sorted([order for order in self.orders if order.picked == 1],
                                     key=lambda order: order.time)[:1]


            min_time_order = random.choice(min_time_orders)  

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
       
        for shelf in assigned_shelves:
            shelf.used = True
            shelf.time = self.manhatan(shelf.location, assigned_station.location)
      
        assigned_robots = []

        total_robot_distance = 0
        for shelf in assigned_shelves:
           
            robot_idle_time = 0
            if not any(robot.used == False for robot in self.robots):
                min_time_robots = sorted(self.robots, key=lambda robot: robot.time)[:12]
                min_time_robot = min_time_robots[0]
                min_time_robot.idle_time += 0

                for robot in min_time_robots:
                    if robot.time <= min_time_robot.time:  # 释放机器人
                        robot.time = 0
                        robot.location = robot.shelf.location
                        robot.shelf.time = 0
                        robot.shelf.used = False
                        robot.used = False

                for robot in self.robots:  # 更新其余机器人的属性
                    if robot.used == True:
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
            assigned_robot.idle_time += robot_idle_time
            assigned_robots.append(assigned_robot)
            assigned_robot.used = True 
            assigned_robot.time = min_robot_distance + shelf.time * 2  
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

        return order, self.states_before_bf, cost




def evaluate(order_num):
    env = Warehouse(OrderGenerator(order_num), ShelfGenerator(), StationGenerator(4, 0), RobotGenerator(12))
    cost = 0
    evaluate_list =[]
    for order in env.orders:
        a, b, c = env.greedy_algorithm3(order)
        cost += c
        evaluate_list.append(a.time)
    cost += 18.75 / 4.68 * sum([robot.idle_time for robot in env.robots]) / 1000 + 18.75 / 4.68 * sum(
        [station.idle_time for station in env.workstations]) / 1000
    return  evaluate_list





