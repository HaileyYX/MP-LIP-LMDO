import math

import numpy as np


class Product:
    def __init__(self, num: int, xy: tuple, i_type, d, s_level, time_limit):
        self.xy = xy  # 坐标
        self.num = num  # 对应客户的编号
        self.type = i_type  # 产品类型
        self.d = d  # 平均需求量
        self.s_level = s_level  # 服务水平
        self.assign_to_wh = None  # 分配仓库对象
        self.assign_to_ex = None  # 分配快运点对象
        self.time_limit = time_limit  # 时限要求
        self.min_dist_wh = None  # 缓存距离最近的warehouse
        self.able_to_cover = set()  # 缓存可以分配的wh集合

    def assigning_wh(self, wh):
        """
        分配仓库
        :param wh:
        :return:
        """
        self.assign_to_wh = wh

    def assigning_ex(self, ex):
        """
        选择快运点
        :param ex:
        :return:
        """
        self.assign_to_ex = ex

    def clear(self):
        self.assign_to_wh = None
        self.assign_to_ex = None


class WareHouse:
    def __init__(self, num, xy, f_cost, hold_cost, lead_time):
        self.num = num  # 编号
        self.xy = xy  # 坐标
        self.f_cost = f_cost  # 开放成本
        self.hold_cost = np.asarray(hold_cost)  # 对各产品的单位年持有成本
        self.lead_time = np.asarray(lead_time)  # 对各产品的提前期
        # TODO:self.min_arrive_ex = None  # 到达仓库时间最短的ex（响应时间+运输时间）
        self.cover_to = set()  # 覆盖的产品对象
        self.stock = np.zeros(len(hold_cost))  # 各产品的库存水平
        self.max_s_level = np.zeros(len(hold_cost))  # 根据覆盖对象需满足的最大服务水平
        self.must_open = False  # 是否必须要开放(供perturb1使用)
        self.type_travel_cost_part1 = np.zeros(len(hold_cost))  # 缓存当前仓库各产品类型的正常运输成本
        self.type_travel_cost_part2 = np.zeros(len(hold_cost))  # 缓存当前仓库各产品类型的中心库运输成本

    def covering(self, prod):
        """
        覆盖产品prod
        :param prod:
        :return:
        """
        self.cover_to.add(prod)
        self.max_s_level[prod.type] = np.max((self.max_s_level[prod.type], prod.s_level))

    def pop(self, prod):
        # todo:在使用优先队列储存客户后，可以进行优化
        self.cover_to.remove(prod)
        self.max_s_level[prod.type] = 0
        for now_prod in self.cover_to:
            if now_prod.type == prod.type:
                self.max_s_level[prod.type] = max(self.max_s_level[prod.type], now_prod.s_level)

    def clear(self):
        self.cover_to.clear()
        self.stock *= 0
        self.max_s_level *= 0
        self.type_travel_cost_part1 *= 0
        self.type_travel_cost_part2 *= 0

    def __str__(self):
        return "仓库编号{0},覆盖的产品的编号类型对{1}".format(self.num, [(prod.num, prod.type) for prod in self.cover_to])


class Express:
    def __init__(self, num, xy, res_time, dist_limit=math.inf):
        self.num = num  # 编号
        self.xy = xy  # 坐标
        self.res_time = res_time  # 响应时间
        self.dist_limit = dist_limit  # 分配距离限制
        self.cover_prod = set()  # 服务的产品

    def covering(self, prod):
        """
        服务产品
        :param prod:
        :return:
        """
        self.cover_prod.add(prod)

    def pop(self, prod):
        self.cover_prod.remove(prod)

    def clear(self):
        self.cover_prod.clear()
