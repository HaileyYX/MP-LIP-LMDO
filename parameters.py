import math

from classes import *


class Parameter:
    def __init__(self):
        self.prod_type_num = -1  # 产品类型的数量
        self.max_demands = dict()  # 给定仓库编号、顾客编号、产品类型编号、库存水平，求出能服务的最大平均需求
        self.product_set = set()  # 产品集合
        self.product_type_set = list()  # 按产品类型区分的产品集合
        self.product_type_list = list()  # 按产品类型区分的产品列表
        self.num_type_product_dict = dict()  # (num, type)对应的产品字典
        self.product_list = list()  # 产品列表
        self.warehouse_set = set()  # 仓库集合
        self.warehouse_list = list()  # 仓库列表
        self.express_set = set()  # 快运点集合
        self.express_list = list()  # 快运点列表s
        self.per_cost = list()  # 各类product的单位距离运输成本
        self.vir_per_cost = list() 
        self.dist_cp = None 
        self.dist_ew = None 
        self.dist_wp = None  
        self.v_ew = 0  
        self.v_wp = list()  
        self.big_M = 1000000
        self.io_to = None

        self.bks = None  # 当前最优解
        self.base_s = None  # base解
        self.s = None  # 当前解

        self.allocate_best_records = []
        self.locate_best_records = []


class Solution:
    def __init__(self):
        self.sequence = list()  
        self.cost = math.inf


class SolutionStructure:

    def __init__(self, s: Solution, pr: Parameter):
        location = list()
        stocks = list()
        allocation_pw = dict()
        allocation_pe = dict()
        for wh in s.sequence:
            assert isinstance(wh, WareHouse)
            location.append(wh.num)
            stocks.append(wh.stock.copy())
            for prod in wh.cover_to:
                assert isinstance(prod, Product)
                allocation_pw[(prod.num, prod.type)] = wh.num
                if prod.assign_to_ex is not None:
                    allocation_pe[(prod.num, prod.type)] = prod.assign_to_ex.num

        self.cost = s.cost  
        self.location = location  
        self.allocation_pw = allocation_pw 
        self.allocation_pe = allocation_pe
        self.stocks = stocks 
        self.pr = pr

    def restore_solution(self):
        for wh in self.pr.warehouse_list:
            wh.clear()
        for ex in self.pr.express_list:
            ex.clear()
        for prod in self.pr.product_list:
            prod.clear()
        s = Solution()
        for i in range(len(self.location)):
            wh_num = self.location[i]
            wh = self.pr.warehouse_list[wh_num]
            stock = self.stocks[i].copy()
            assert isinstance(wh, WareHouse)
            wh.stock = stock
            s.sequence.append(wh)
        for num, p_type in self.allocation_pw:
            prod = self.pr.num_type_product_dict[num, p_type]
            assert isinstance(prod, Product)
            wh_num = self.allocation_pw[num, p_type]
            wh = self.pr.warehouse_list[wh_num]
            assert isinstance(wh, WareHouse)
            wh.covering(prod)
            prod.assigning_wh(wh)
            if (num, p_type) in self.allocation_pe:
                ex_num = self.allocation_pe[num, p_type]
                ex = self.pr.express_list[ex_num]
                assert isinstance(ex, Express)
                ex.covering(prod)
                prod.assigning_ex(ex)
        s.cost = self.cost
        return s


class SystemPara:

    def __init__(self, location_t_start=100000, allocation_t_start=100000, location_rho=0.98, allocation_rho=0.99,
                 location_t_end=10000, allocation_t_end=10000, r=0.55, bonus1=60, bonus2=30, bonus3=20, from_amount=15,
                 to_amount=10):
        self.location_t_start = location_t_start  # 初始温度
        self.allocation_t_start = allocation_t_start
        self.location_rho = location_rho  # 冷却速率
        self.allocation_rho = allocation_rho
        self.location_t_end = location_t_end  # 终止温度
        self.allocation_t_end = allocation_t_end
        self.r = r  # 分数控制权重
        self.bonus1 = bonus1  # 最高分数奖励
        self.bonus2 = bonus2  # 次高分数奖励
        self.bonus3 = bonus3  # 最低分数奖励
        self.from_amount = from_amount 
        self.to_amount = to_amount 
        self.location_op_scores = [0, 0]  
        self.location_op_weights = [0, 0]  
        self.location_op_weights_to1 = [0.5, 0.5] 
        self.location_op_counts = [0, 0] 
        self.allocation_op_scores = [0, 0] 
        self.allocation_op_weights = [0.5, 0.5] 
        self.allocation_op_weights_to1 = [0.5, 0.5]  
        self.allocation_op_counts = [0, 0] 

    def computing_location_weights_to1(self):
        sum_weight = sum(self.location_op_weights)
        if sum_weight == 0:
            return
        for i in range(len(self.location_op_weights)):
            self.location_op_weights_to1[i] = self.location_op_weights[i] / sum_weight

    def computing_allocation_weights_to1(self):
        sum_weight = sum(self.allocation_op_weights)
        for i in range(len(self.allocation_op_weights)):
            self.allocation_op_weights_to1[i] = self.allocation_op_weights[i] / sum_weight

    def updating_location_score(self, update_level, num):
        if update_level == 0:
            self.location_op_scores[num] += self.bonus1
        elif update_level == 1:
            self.location_op_scores[num] += self.bonus2
        else:
            self.location_op_scores[num] += self.bonus3
        self.location_op_counts[num] += 1

    def updating_allocation_score(self, update_level, num):
        if update_level == 0:
            self.allocation_op_scores[num] += self.bonus1
        elif update_level == 1:
            self.allocation_op_scores[num] += self.bonus2
        else:
            self.allocation_op_scores[num] += self.bonus3
        self.allocation_op_counts[num] += 1

    def updating_location_weight(self):
        for i in range(len(self.location_op_weights)):
            if self.location_op_counts[i] != 0:
                self.location_op_weights[i] = (1 - self.r) * self.location_op_weights[i] + self.r * \
                                              self.location_op_weights[i] / self.location_op_counts[i]
                self.location_op_counts[i] = 0

    def updating_allocation_weight(self):
        for i in range(len(self.allocation_op_weights)):
            if self.allocation_op_counts[i] != 0:
                self.allocation_op_weights[i] = (1 - self.r) * self.allocation_op_weights[i] + self.r * \
                                                self.allocation_op_weights[i] / self.allocation_op_counts[i]
                self.allocation_op_counts[i] = 0
