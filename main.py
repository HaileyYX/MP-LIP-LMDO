import _io
import random
import time

from classes import *
from data_loading import data_loading
from parameters import Parameter, Solution
from function_tools import *
import matplotlib.pyplot as plt


def test_solution():
    pr = Parameter()
    data_loading(pr)
    pr.io_to = open("log1", "w")
    pr.s = Solution()
    location_list = [0,1,2]
    allocation_list = [2, 2, 2, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]
    travel_type_list = [4, 4, 4, 3, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 4, 4, 2, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]
    stocks_dict = {2: [8, 8, 7, 4, 4]}
    all_type = 3
    for i in location_list:
        pr.s.sequence.append(pr.warehouse_list[i])
    for i in range(len(allocation_list)):
        prod = pr.product_list[i]
        wh = pr.warehouse_list[allocation_list[i]]
        if travel_type_list[i] == -1:
            ex = None
        else:
            ex = pr.express_list[travel_type_list[i]]
        assert isinstance(prod, Product)
        assert isinstance(wh, WareHouse)
        prod.assigning_wh(wh)
        wh.covering(prod)
        prod.assigning_ex(ex)
        if ex is not None:
            ex.covering(prod)
    for wh in pr.s.sequence:
        assert isinstance(wh, WareHouse)
        demand_sum = np.zeros(pr.prod_type_num)
        for prod in wh.cover_to:
            demand_sum[prod.type] += prod.d
        demand_sum *= wh.lead_time
        for i in range(len(demand_sum)):
            if wh.max_s_level[i] == 0:
                wh.stock[i] += 0
            else:
                wh.stock[i] += get_min_s(wh.max_s_level[i], demand_sum[i], pr.max_demands)
    """for prod in pr.product_set:
        assert isinstance(prod, Product)
        v2 = pr.v_wp[prod.type]
        min_time = math.inf
        min_ex = None
        wh = prod.assign_to_wh
        for ex in pr.express_set:
            need_time = ex.res_time + pr.dist_ew[ex.num, wh.num] / pr.v_ew + pr.dist_wp[wh.num, prod.num] / v2
            if min_time > need_time:
                min_time = need_time
                min_ex = ex
        if min_time <= prod.time_limit:
            prod.assigning_ex(min_ex)
            min_ex.covering(prod)"""
    pr.s.cost = get_cost(pr.s.sequence, pr)
    pr.bks = SolutionStructure(pr.s, pr)
    pr.base_s = SolutionStructure(pr.s, pr)

    print(pr.s.cost)
    structure = SolutionStructure(pr.s, pr)
    solution_output(structure)
    # log_out_complete(pr.io_to, pr)
    remove_num_set = {(2, 2), (3, 2), (4, 0), (5, 0), (6, 1), (8, 0), (1, 1), (8, 1), (2, 0), (9, 0)}
    remove_set = set()
    for prod in pr.product_list:
        if (prod.num, prod.type) in remove_num_set:
            remove_set.add(prod)
    # log_out_remove(pr.io_to, remove_set)
    implement_destroy(remove_set, pr)
    pr.io_to.close()


if __name__ == '__main__':
    # random.seed(1)
    start = time.time()
    pr = Parameter()
    data_loading(pr)
    # pr.io_to = open("log", "w")
    # test_solution()
    greedy_heuristic(pr)
    location_iter(pr)
    end = time.time()
    print(pr.bks.cost)
    solution_output(pr.bks)
    s = pr.bks.restore_solution()
    print("开放成本 = ", get_open_cost(s.sequence))
    print("持有成本 = ", get_hold_cost(s.sequence))
    print("行驶成本 = ", get_travel_cost(s.sequence, pr))
    print("算法耗时：", round(end - start, 3), 's')
    # assert isinstance(pr.io_to, _io.TextIOWrapper)
    # pr.io_to.close()
    # print("开放的网点为：", pr.bks.location)

### 求解全部直运的运输方案：注释掉if段，将现存方案的location allocation 运输方案输入，在使用函数test_solution()
# test_solution()
