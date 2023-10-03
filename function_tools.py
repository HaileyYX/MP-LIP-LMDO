import _io
import math
import random
import heapq
import numpy as np

from parameters import *
from classes import *


def check_relationship(pr: Parameter):

    for prod in pr.product_list:
        assert isinstance(prod, Product)
        wh = prod.assign_to_wh
        ex = prod.assign_to_ex
        assert isinstance(wh, WareHouse)
        assert prod in wh.cover_to
        if ex is not None:
            assert isinstance(ex, Express)
            assert prod in ex.cover_prod
    count = 0
    for wh in pr.s.sequence:
        assert isinstance(wh, WareHouse)
        count += len(wh.cover_to)
    assert count == len(pr.product_list)
    for wh in pr.warehouse_list:
        if wh not in pr.s.sequence:
            assert len(wh.cover_to) == 0


def get_cost(s: list, pr: Parameter, implement=False):

    cost1 = get_open_cost(s)
    cost2 = get_hold_cost(s)
    cost3 = get_travel_cost(s, pr, implement)
    # print("open_cost = {0}, hold_cost = {1}, travel_cost = {2}".format(cost1, cost2, cost3))
    return cost1 + cost2 + cost3


def get_open_cost(s: list):

    cost = 0
    for wh in s:
        if len(wh.cover_to) != 0:
            cost += wh.f_cost
    return cost


def get_travel_cost(s: list, pr: Parameter, implement=False):
    cost = 0
    for wh in s:
        assert isinstance(wh, WareHouse)
        for prod in wh.cover_to:
            ex = prod.assign_to_ex
            if ex is None:
                cost_part1 = pr.dist_wp[wh.num, prod.num] * pr.vir_per_cost[prod.type]
            else:
                cost_part1 = (pr.dist_ew[ex.num, wh.num] + pr.dist_wp[wh.num, prod.num]) * pr.per_cost[prod.type]
            cost_part1 *= prod.d
            cost += cost_part1
            if implement:
                wh.type_travel_cost_part1[prod.type] += cost_part1
    return cost


def get_hold_cost(s: list):
  
    cost = 0
    for wh in s:
        assert isinstance(wh, WareHouse)
        cost += np.sum(wh.hold_cost * wh.stock)
    return cost


def get_min_s(s_level, demand_sum, max_d_table):


    def binary_search(l, r, aim, s_demand_list):
        if r == l:
            if s_demand_list[l][1] >= aim:
                return s_demand_list[l][0]
            else:
                raise Exception('最大库存依然无法满足需求')
        if r - 1 == l:
            if s_demand_list[l][1] >= aim:
                return s_demand_list[l][0]
            elif s_demand_list[r][1] >= aim:
                return s_demand_list[r][0]
            else:
                raise Exception('最大库存依然无法满足需求')
        c = (l + r) // 2
        c_val = s_demand_list[c][1]
        if c_val >= aim:
            return binary_search(l, c, aim, s_demand_list)
        return binary_search(c + 1, r, aim, s_demand_list)

    s_demand_list = max_d_table[s_level]
    return binary_search(0, len(s_demand_list) - 1, demand_sum, s_demand_list)


def get_minus_travel(aim_list: list, pr: Parameter, implement=False):

    direct_cost = 0
    for prod in aim_list:
        assert isinstance(prod, Product)
        wh = prod.assign_to_wh
        assert isinstance(wh, WareHouse)
        ex = prod.assign_to_ex
        if ex is not None:
            assert isinstance(ex, Express)
            cost1_part1 = pr.dist_ew[ex.num, wh.num] * pr.per_cost[prod.type]
            cost1_part2 = pr.dist_wp[wh.num, prod.num] * pr.per_cost[prod.type]
        else:
            cost1_part1 = 0
            cost1_part2 = pr.dist_wp[wh.num, prod.num] * pr.vir_per_cost[prod.type]
        cost1 = (cost1_part1 + cost1_part2) * prod.d
        single_direct_cost = cost1
        direct_cost += single_direct_cost
    return direct_cost


def get_minus_hold(aim_list: list, pr: Parameter, implement=False):
    cost = 0
    aim_set = set(aim_list)
    change_wh_set = {(prod.assign_to_wh, prod.type) for prod in aim_list}
    for wh, ttype in change_wh_set:
        notfull_demand = 0
        max_s_level = 0
        assert isinstance(wh, WareHouse)
        for prod in wh.cover_to:
            assert isinstance(prod, Product)
            if prod.type != ttype:
                continue
            if prod not in aim_set:
                max_s_level = max(max_s_level, prod.s_level)
                notfull_demand += prod.d
        notfull_demand *= wh.lead_time[ttype]
        if max_s_level == 0:
            new_stock = 0
        else:
            try:
                new_stock = get_min_s(max_s_level, notfull_demand, pr.max_demands)
            except Exception:
                new_stock = pr.big_M
        cost += (wh.stock[ttype] - new_stock) * wh.hold_cost[ttype]
        if implement:
            wh.stock[ttype] = new_stock
    return cost


def get_minus_open(aim_list: list, pr: Parameter):
    cost = 0
    remove_prod_dict = dict()
    for prod in aim_list:
        remove_prod_dict[prod.assign_to_wh] = remove_prod_dict.get(prod.assign_to_wh, 0) + 1
    for wh, val in remove_prod_dict.items():
        assert isinstance(wh, WareHouse)
        if len(wh.cover_to) == val:
            cost += wh.f_cost
    return cost


def get_minus_cost(aim_list: list, pr: Parameter, implement=False):
    cost1 = get_minus_travel(aim_list, pr, implement)
    cost3 = get_minus_hold(aim_list, pr, implement)
    cost4 = get_minus_open(aim_list, pr)
    return cost1 + cost3 + cost4


def get_add_travel(aim_list: list, action_list, pr: Parameter, implement=False):

    assign_to_ex_list = []
    direct_cost = 0  # 直接运输成本
    for i in range(len(aim_list)):
        prod = aim_list[i]
        wh = action_list[i]
        assert isinstance(prod, Product)
        assert isinstance(wh, WareHouse)
        # 确定运输模式
        time_wp = pr.dist_wp[wh.num, prod.num] / pr.v_wp[prod.type]
        if time_wp > prod.time_limit:
            return pr.big_M
        best_ex = None
        for ex in pr.express_list:
            assert isinstance(ex, Express)
            time_ew = pr.dist_ew[ex.num, wh.num] / pr.v_ew + ex.res_time
            if time_ew + time_wp <= prod.time_limit:
                if best_ex is None:
                    best_ex = ex
                elif pr.dist_ew[ex.num, wh.num] < pr.dist_ew[best_ex.num, wh.num]:
                    best_ex = ex
        assign_to_ex_list.append(best_ex)
        if best_ex is None:
            cost1 = 0
            cost2 = pr.dist_wp[wh.num, prod.num] * pr.vir_per_cost[prod.type] * prod.d
        else:
            cost1 = pr.dist_ew[best_ex.num, wh.num] * pr.per_cost[prod.type] * prod.d
            cost2 = pr.dist_wp[wh.num, prod.num] * pr.per_cost[prod.type] * prod.d
        cost_part1 = cost1 + cost2
        single_direct_cost = cost_part1
        direct_cost += single_direct_cost
    return direct_cost


def get_add_hold(aim_list: list, action_list: list, pr: Parameter, implement=False):
    wh_type_levels = dict()
    wh_type_add_d = dict()
    for i in range(len(aim_list)):
        prod = aim_list[i]
        wh = action_list[i]
        assert isinstance(prod, Product)
        assert isinstance(wh, WareHouse)
        wh_type_levels[wh, prod.type] = max(wh_type_levels.get((wh, prod.type), 0), wh.max_s_level[prod.type],
                                            prod.s_level)
        wh_type_add_d[wh, prod.type] = wh_type_add_d.get((wh, prod.type), 0) + prod.d
    cost = 0
    for (wh, p_type), s_level in wh_type_levels.items():
        add_d = wh_type_add_d[wh, p_type]
        ori_d = 0
        for prod in wh.cover_to:
            if prod.type == p_type:
                ori_d += prod.d
        lead_d = (ori_d + add_d) * wh.lead_time[p_type]
        if s_level == 0:
            new_stock = 0
        else:
            try:
                new_stock = get_min_s(s_level, lead_d, pr.max_demands)
            except Exception:
                new_stock = pr.big_M
        cost += (new_stock - wh.stock[p_type]) * wh.hold_cost[p_type]
        if implement:
            wh.stock[p_type] = new_stock
    return cost


def get_add_open(aim_list: list, action_list: list, pr: Parameter):
    cost = 0
    is_compute = set()
    for wh in action_list:
        assert isinstance(wh, WareHouse)
        if len(wh.cover_to) == 0 and wh.num not in is_compute:
            cost += wh.f_cost
            is_compute.add(wh.num)
    return cost


def get_add_cost(aim_list: list, action_list: list, pr: Parameter, implement=False):
    cost1 = get_add_travel(aim_list, action_list, pr, implement)
    cost2 = get_add_hold(aim_list, action_list, pr, implement)
    cost3 = get_add_open(aim_list, action_list, pr)
    """if implement:
        print("增加的运输成本为：", cost1)
        print("增加的持有成本为：", cost2)
        print("增加的开放成本为：", cost3)"""
    return cost1 + cost2 + cost3


def greedy_heuristic(pr: Parameter):
    """for prod in pr.product_set:
        assert isinstance(prod, Product)
        wh = prod.min_dist_wh
        assert isinstance(wh, WareHouse)
        prod.assigning_wh(wh)
        wh.covering(prod)"""

    for prod_list in pr.product_type_list:
        num_one_type = len(prod_list)
        max_cover = int(math.ceil(num_one_type / len(pr.warehouse_list)))
        wh_cover_num_list = [0 for _ in range(len(pr.warehouse_list))]
        for prod in prod_list:
            assert isinstance(prod, Product)
            dist = np.copy(pr.dist_wp[:, prod.num])
            ite = 0
            while True:
                aim_wh_num = np.argmin(dist)
                if wh_cover_num_list[aim_wh_num] < max_cover:
                    wh_cover_num_list[aim_wh_num] += 1
                    wh = pr.warehouse_list[aim_wh_num]
                    assert isinstance(wh, WareHouse)
                    prod.assigning_wh(wh)
                    wh.covering(prod)
                    break
                else:
                    dist[aim_wh_num] += sum(dist)
                    ite += 1
                if ite >= len(pr.warehouse_list):
                    Exception("平均分配方案也不行")

    pr.s = Solution()
    for wh in pr.warehouse_list:
        if len(wh.cover_to) != 0:
            pr.s.sequence.append(wh)
    for wh in pr.s.sequence:
        assert isinstance(wh, WareHouse)
        demand_sum = np.zeros(pr.prod_type_num)
        for prod in wh.cover_to:
            demand_sum[prod.type] += prod.d
        demand_sum *= wh.lead_time
        for i in range(len(demand_sum)):
            if wh.max_s_level[i] == 0:
                wh.stock[i] = 0
            else:
                try:
                    wh.stock[i] = get_min_s(wh.max_s_level[i], demand_sum[i], pr.max_demands)
                except Exception:
                    wh.stock[i] = 100000000
    for prod in pr.product_set:
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
            min_ex.covering(prod)
    pr.s.cost = get_cost(pr.s.sequence, pr, True)
    pr.bks = SolutionStructure(pr.s, pr)
    pr.base_s = SolutionStructure(pr.s, pr)
    # check:
    travel_cost = 0
    for wh in pr.s.sequence:
        travel_cost += (np.sum(wh.type_travel_cost_part1) + np.sum(wh.type_travel_cost_part2))
    if abs(travel_cost - get_travel_cost(pr.s.sequence, pr)) > 0.1:
        raise Exception("一开始缓存成本就错了！")


def destroy1_random(from_size, size: int, pr: Parameter):
    return set(random.sample(pr.product_list, size))


def destroy2_greedy_random(from_size: int, select_size: int, pr: Parameter):
    heap = []
    for prod in pr.product_list:
        assert isinstance(prod, Product)
        cost = get_minus_cost([prod], pr)
        block = Block(prod, cost)
        if len(heap) < from_size:
            heapq.heappush(heap, block)
        else:
            if block > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, block)
    return {block.item for block in random.sample(heap, select_size)}


def perturb1_wh_random_remove(select_size, pr: Parameter):
    assert select_size == 1
    for wh in pr.s.sequence:
        wh.must_open = False
    for prod in pr.product_list:
        assert isinstance(prod, Product)
        only_wh = None
        is_break = False
        for wh in pr.s.sequence:
            assert isinstance(wh, WareHouse)
            if pr.dist_wp[wh.num, prod.num] / pr.v_wp[prod.type] <= prod.time_limit:
                if only_wh is None:
                    only_wh = wh
                else:
                    is_break = True
                    break
        if not is_break:
            if only_wh is None:
                raise Exception("当前数据找不到可行解！")
            else:
                only_wh.must_open = True
    waiting_remove_list = list()
    for wh in pr.s.sequence:
        assert isinstance(wh, WareHouse)
        if not wh.must_open:
            waiting_remove_list.append(wh)
    remove_prod_set = set()
    if len(waiting_remove_list) < select_size:
        return
    else:
        remove_wh = random.sample(waiting_remove_list, select_size)
        for wh in remove_wh:
            for prod in wh.cover_to:
                remove_prod_set.add(prod)
    implement_destroy(remove_prod_set, pr)
    for wh in remove_wh:
        pr.s.sequence.remove(wh)
    repair1_regret_k(2, remove_prod_set, pr)


def perturb2_wh_random_exchange(select_size, pr: Parameter)
    assert select_size == 1
    able_to_change_list = pr.s.sequence[:] 
    close_wh_set = set() 
    open_wh_set = set(pr.s.sequence) 
    for wh in pr.warehouse_list:
        assert isinstance(wh, WareHouse)
        if wh not in open_wh_set:
            close_wh_set.add(wh)
    while len(able_to_change_list) > 0:
        remove_wh = random.sample(able_to_change_list, 1)[0]
        assert isinstance(remove_wh, WareHouse)
        open_wh_set.remove(remove_wh)
        able_to_change_list.remove(remove_wh)

        must_select_set = close_wh_set.copy()
        for prod in remove_wh.cover_to:
            assert isinstance(prod, Product)
            if len(prod.able_to_cover | open_wh_set) == 0:
                must_select_set = must_select_set | prod.able_to_cover
        open_wh_set.add(remove_wh)
        if len(must_select_set) != 0:
            add_wh = random.sample(list(must_select_set), 1)[0]
            assert isinstance(add_wh, WareHouse)
            add_wh.clear()
            remove_list = list(remove_wh.cover_to)
            implement_destroy(set(remove_list), pr)
            # 换location
            for i in range(len(pr.s.sequence)):
                if pr.s.sequence[i] == remove_wh:
                    pr.s.sequence[i] = add_wh
        
            pr.s.cost += get_add_cost(remove_list, [add_wh for _ in range(len(remove_list))], pr, True)
            for prod in remove_list:
                prod.assigning_wh(add_wh)
                add_wh.covering(prod)
                best_ex = None
                time_wp = pr.dist_wp[add_wh.num, prod.num] / pr.v_wp[prod.type]
                if time_wp > prod.time_limit:
                    pass
                else:
                    for ex in pr.express_list:
                        assert isinstance(ex, Express)
                        time_ew = pr.dist_ew[ex.num, add_wh.num] / pr.v_ew + ex.res_time
                        if time_ew + time_wp <= prod.time_limit:
                            if best_ex is None:
                                best_ex = ex
                            elif pr.dist_ew[ex.num, add_wh.num] < pr.dist_ew[best_ex.num, add_wh.num]:
                                best_ex = ex
                prod.assigning_ex(best_ex)
                best_ex.covering(prod)
            assert abs(get_cost(pr.s.sequence, pr) - pr.s.cost) < 0.1
            break


def perturb3_wh_random_add(select_size, pr: Parameter):
    assert select_size == 1
    close_wh_list = list()
    open_wh_set = set(pr.s.sequence)
    for wh in pr.warehouse_list:
        assert isinstance(wh, WareHouse)
        if wh not in open_wh_set:
            close_wh_list.append(wh)
    if len(close_wh_list) == 0:
        return
    select_wh = random.sample(close_wh_list, 1)[0]
    empty_prod = Product(0, (-1, -1), 0, 0, 0, 0)
    select_wh.covering(empty_prod)
    pr.s.sequence.append(select_wh)
    empty_prod.assigning_wh(select_wh)
    empty_prod.assigning_ex(None)
    pr.s.cost = get_cost(pr.s.sequence, pr)
    remove_set = destroy2_greedy_random(len(pr.product_set) // len(pr.s.sequence),
                                        len(pr.product_set) // len(pr.s.sequence), pr)
    implement_destroy(remove_set, pr)
    repair1_regret_k(2, remove_set, pr)
    select_wh.pop(empty_prod)
    pr.s.cost = get_cost(pr.s.sequence, pr)


def implement_destroy(remove_set: set, pr: Parameter):
    delta_cost = get_minus_cost(list(remove_set), pr, True)
    pr.s.cost -= delta_cost
    for prod in remove_set:
        assert isinstance(prod, Product)
        wh = prod.assign_to_wh
        ex = prod.assign_to_ex
        wh.pop(prod)
        if ex is not None:
            ex.pop(prod)
        prod.clear()
    real_cost = get_cost(pr.s.sequence, pr)
    # log_out_rest(pr.io_to, pr)
    # log_out_cost(pr.io_to, pr.s.cost, real_cost)
    if abs(real_cost - pr.s.cost) > 0.1:
        print("update cost = {0}".format(pr.s.cost))
        print("real cost = {0}".format(real_cost))
        assert isinstance(pr.io_to, _io.TextIOWrapper)
        pr.io_to.close()
        raise Exception("成本不匹配")


def repair_greedy(k, remove_set: set, pr: Parameter):
    remove_list = list(remove_set)
    random.shuffle(remove_list)
    for prod in remove_list:
        assert isinstance(prod, Product)
        aim_wh = None
        aim_cost = 1e+50
        for wh in pr.s.sequence:
            assert isinstance(wh, WareHouse)
            cost = get_add_cost([prod], [wh], pr)
            if cost < aim_cost:
                aim_cost = cost
                aim_wh = wh
        wh = aim_wh
        pr.s.cost += get_add_cost([prod], [wh], pr, True)
        prod.assigning_wh(wh)
        wh.covering(prod)
        best_ex = None
        time_wp = pr.dist_wp[wh.num, prod.num] / pr.v_wp[prod.type]
        if time_wp > prod.time_limit:
            pass
        else:
            for ex in pr.express_list:
                assert isinstance(ex, Express)
                time_ew = pr.dist_ew[ex.num, wh.num] / pr.v_ew + ex.res_time
                if time_ew + time_wp <= prod.time_limit:
                    if best_ex is None:
                        best_ex = ex
                    elif pr.dist_ew[ex.num, wh.num] < pr.dist_ew[best_ex.num, wh.num]:
                        best_ex = ex
        prod.assigning_ex(best_ex)
        if best_ex is not None:
            best_ex.covering(prod)


def repair1_regret_k(k: int, remove_set: set, pr: Parameter)

    def update_info(best_prod_wh1, max_regret1, prod1, wh_dict):

        sorted_blocks = [Block(wh1, cost1) for (wh1, cost1) in wh_dict.items()]
        sorted_blocks.sort()
        sorted_blocks = sorted_blocks[: min(len(sorted_blocks), k)]
        value = sum([(b.value - sorted_blocks[0].value) for b in sorted_blocks])
        if value > max_regret1:
            max_regret1 = value
            best_prod_wh1 = (prod1, sorted_blocks[0].item)
        return best_prod_wh1, max_regret1

    remove_list = list(remove_set)
    prod_wh_dict = dict()
    best_prod_wh = None
    max_regret = -1
    for prod in remove_set:
        assert isinstance(prod, Product)
        prod_wh_dict[prod] = dict()
        for wh in pr.s.sequence:
            assert isinstance(wh, WareHouse)
            cost = get_add_cost([prod], [wh], pr)
            prod_wh_dict[prod][wh] = cost
        best_prod_wh, max_regret = update_info(best_prod_wh, max_regret, prod, prod_wh_dict[prod])
    while True:
        prod, wh = best_prod_wh
        assert isinstance(prod, Product)
        assert isinstance(wh, WareHouse)
        pr.s.cost += get_add_cost([prod], [wh], pr, True)
        prod.assigning_wh(wh)
        wh.covering(prod)
        best_ex = None
        time_wp = pr.dist_wp[wh.num, prod.num] / pr.v_wp[prod.type]
        if time_wp > prod.time_limit:
            pass
        else:
            for ex in pr.express_list:
                assert isinstance(ex, Express)
                time_ew = pr.dist_ew[ex.num, wh.num] / pr.v_ew + ex.res_time
                if time_ew + time_wp <= prod.time_limit:
                    if best_ex is None:
                        best_ex = ex
                    elif pr.dist_ew[ex.num, wh.num] < pr.dist_ew[best_ex.num, wh.num]:
                        best_ex = ex
        prod.assigning_ex(best_ex)
        if best_ex is not None:
            best_ex.covering(prod)
        remove_set.remove(prod)
        best_prod_wh = None
        max_regret = -1
        if len(remove_set) == 0:
            break
        # 修复
        for prod in remove_set:
            assert isinstance(prod, Product)
            cost = get_add_cost([prod], [wh], pr)
            prod_wh_dict[prod][wh] = cost
            best_prod_wh, max_regret = update_info(best_prod_wh, max_regret, prod, prod_wh_dict[prod])
    # log_out_complete(pr.io_to, pr)
    # log_out_text(pr.io_to, "修复完成后的结果：")
    # log_out_cost(pr.io_to, pr.s.cost, get_cost(pr.s.sequence, pr))
    if abs(pr.s.cost - get_cost(pr.s.sequence, pr)) > 0.1:
        print("update cost = {0}".format(pr.s.cost))
        print("real cost = {0}".format(get_cost(pr.s.sequence, pr)))
        assert isinstance(pr.io_to, _io.TextIOWrapper)
        pr.io_to.close()
        raise Exception("成本不匹配！")


def allocation_iter(sp: SystemPara, pr: Parameter):
    temper = sp.allocation_t_start
    funcs = [destroy1_random, destroy2_greedy_random]
    improve_level = 2 
    while temper >= sp.allocation_t_end:
        prob = random.random()
        acc_prob = 0
        for i in range(len(funcs)):
            acc_prob += sp.allocation_op_weights_to1[i]
            if acc_prob > prob:
                # log_out_iter_split(pr.io_to)
                # log_out_complete(pr.io_to, pr)
                # log_out_cost(pr.io_to, pr.s.cost, get_cost(pr.s.sequence, pr))
                check(pr)
                last_s_struct = SolutionStructure(pr.s, pr)
                check(pr)
                remove_set = funcs[i](sp.from_amount, sp.to_amount, pr)
                check(pr)
                # log_out_remove(pr.io_to, remove_set)
                implement_destroy(remove_set, pr)
                if random.random() > 0.5:
                    repair_greedy(1, remove_set, pr)
                else:
                    repair1_regret_k(2, remove_set, pr)
                check_relationship(pr)
                check(pr)
                if pr.s.cost < pr.bks.cost:
                    improve_level = 0
                elif pr.s.cost < pr.base_s.cost and improve_level > 0:
                    improve_level = 1

                if pr.s.cost < pr.bks.cost:
                    level = 0
                    pr.bks = SolutionStructure(pr.s, pr)
                elif pr.s.cost < last_s_struct.cost:
                    level = 1
                else:
                    level = 2
                    thres_prob = math.exp((last_s_struct.cost - pr.s.cost) / temper)
                    sa_prob = random.random()
                    if sa_prob > thres_prob:
                        # 不接受当前较差的解
                        pr.s = last_s_struct.restore_solution()
                check(pr)
                sp.updating_allocation_score(level, i)
                check(pr)
                pr.allocate_best_records.append(pr.bks.cost)

                break
        temper *= sp.allocation_rho
    sp.updating_allocation_weight()
    sp.computing_allocation_weights_to1()
    pr.s.sequence = list(filter(lambda wh: len(wh.cover_to) > 0, pr.s.sequence))
    return improve_level


def location_iter(pr: Parameter):
    sp = SystemPara()
    temper = sp.location_t_start
    funcs = [perturb1_wh_random_remove, perturb3_wh_random_add]
    perturb_num = 0
    while temper >= sp.location_t_end:
        perturb_num += 1
        prob = random.random()
        acc_prob = 0
        assert isinstance(pr.base_s, SolutionStructure)
        check(pr)
        for i in range(len(funcs)):
            acc_prob += sp.location_op_weights_to1[i]
            check(pr)
            if acc_prob > prob:
                if pr.bks.cost > 100000000:
                    print("成本太高，增开仓库")
                    try:
                        check(pr)
                        funcs[2](1, pr)
                        check(pr)
                    except Exception:
                        Exception("错误的迭代标签" + str(22))
                else:
                # if i == 1:
                    print("执行扰动", i)

                    check(pr)
                    funcs[i](1, pr)
                    check(pr)

                # log_out_location(pr.io_to)
                check(pr)
                level = allocation_iter(sp, pr)
                check(pr)
                if level == 2:
                    check(pr)
                    thres_prob = math.exp((pr.base_s.cost - pr.s.cost) / temper)
                    sa_prob = random.random()
                    if sa_prob > thres_prob:
                        pr.s = pr.base_s.restore_solution()
                    else:
                        pr.base_s = SolutionStructure(pr.s, pr)

                sp.updating_location_score(level, i)
                sp.updating_location_weight()
                sp.computing_location_weights_to1()
                check(pr)
                break
        check(pr)
        pr.locate_best_records.append(pr.bks.cost)
        print("第{0}轮扰动后，当前最优解成本为：{1}".format(perturb_num, pr.bks.cost))
        check(pr)
        temper *= sp.location_rho


def solution_output(bks: SolutionStructure):
    lis = []
    for prod_num, p_type in bks.allocation_pw:
        wh_num = bks.allocation_pw[prod_num, p_type]
        ex_num = bks.allocation_pe.get((prod_num, p_type), '直运')
        lis.append("{0}分配的仓库是{1}，运输方式是{2}".format((prod_num, p_type), wh_num, ex_num))
        #lis.append("\n")
        #print("{0}分配的仓库是{1}，运输方式是{2}".format((prod_num, p_type), wh_num, ex_num))
    lis.sort()
    for line in lis:
        print(line)
    # 库存水平
    for i in range(len(bks.location)):
        print("仓库{0}的库存水平为{1}：".format(bks.location[i], bks.stocks[i]))


def log_out_complete(to: _io.TextIOWrapper, pr: Parameter):
    """
    打印完整的信息
    :param to:
    :param pr:
    :return:
    """
    for wh in pr.s.sequence:
        to.writelines(str(wh))
        to.writelines("\n")


def log_out_rest(to: _io.TextIOWrapper, pr: Parameter):
    log_out_complete(to, pr)


def log_out_remove(to: _io.TextIOWrapper, remove_set):
    to.writelines('移除的节点是：{0}'.format([(prod.num, prod.type) for prod in remove_set]))
    to.writelines("\n")


def log_out_iter_split(to: _io.TextIOWrapper):
    to.writelines("\n")
    to.writelines("new_iter")
    to.writelines("\n")


def log_out_cost(to: _io.TextIOWrapper, computing_cost, real_cost):
    to.writelines("计算的成本为：{0}，真实的成本为：{1}".format(computing_cost, real_cost))
    to.writelines("\n")


def log_out_location(to: _io.TextIOWrapper):
    to.writelines("\n")
    to.writelines("location执行完")
    to.writelines("\n")


def log_out_text(to: _io.TextIOWrapper, text):
    to.writelines(text)
    to.writelines('\n')


class Block:
    """
    自定义比较器的类
    """

    def __init__(self, item, value):
        self.item = item
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)


class TopK:
    """
    最大的topK
    """

    def __init__(self, k):
        self.k = k
        self.heap = []

    def add(self, other):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, other)
        else:
            if self.heap[0] < other:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, other)


def check(pr: Parameter):
    for prod in pr.product_list:
        if prod.assign_to_wh is None:
            Exception("12345")
