import numpy as np
import math
from parameters import Parameter
from classes import *
import pandas as pd


def get_b(miu, s):
    b = 0
    for r in range(s):
        b += (miu ** r) * math.e ** (-miu) / (math.factorial(r))
    return b


def binary_get(aim, func=get_b, s=0, lb=0., ub=1., tlr=0.01, precision=3):
    c = (lb + ub) / 2
    valc = func(c, s)
    if abs(valc - aim) < tlr:
        return round(c, precision)
    if aim < valc:
        return binary_get(aim, func, s, c, ub, tlr, precision)
    return binary_get(aim, func, s, lb, c, tlr, precision)


def get_dist_per(x1, y1, x2, y2):
    rady1 = math.radians(y1)
    rady2 = math.radians(y2)
    a = rady1 - rady2
    b = math.radians(x1) - math.radians(x2)
    s = 2 * math.asin(
        math.sqrt(math.sin(a / 2) ** 2 + math.cos(rady1) * math.cos(rady2) * math.sin(b / 2) ** 2)) * 6378.004
    return s


def get_dist(group1, group2):
    dist_mat = np.zeros([len(group1), len(group2)])
    for i in range(len(group1)):
        for j in range(len(group2)):
            # dist_mat[i,j] = ((group1[i][0]-group2[j][0])**2 + (group1[i][1]-group2[j][1])**2)**0.5
            dist_mat[i, j] = get_dist_per(group1[i][0], group1[i][1], group2[j][0], group2[j][1])
    return dist_mat.round(2)


def data_loading(pr: Parameter):
    df_data = pd.read_excel("Instance300.xlsx", sheet_name='data')
    df_data_t = pd.read_excel('Instance300.xlsx', sheet_name='data_t_jk')
    df_data_h = pd.read_excel('Instance300.xlsx', sheet_name='data_h_jk')
    df_data_info_cust = pd.read_excel('Instance300.xlsx', sheet_name='info_cust')
    with open('data1.txt', "r", encoding='utf-8') as fl:
        C_num = int(fl.readline().split()[0])
        J_num = int(fl.readline().split()[0])
        I_num = int(fl.readline().split()[0])
        M_num = int(fl.readline().split()[0])
        virtual_num = int(fl.readline().split()[0])
        K_num = int(fl.readline().split()[0])
        per_cost = list(df_data.iloc[:, 1].values)
        #per_cost = list(map(int, fl.readline().split()[0].split(",")))
        h = df_data_h.values[:, 1:].astype("int32")
        """temp = []
        for line in fl.readline().split()[0].split(";"):
            temp.append(list(map(int, line.split(","))))
        h = np.array(temp)"""
        t = df_data_t.values[:, 1:].astype("int32")
        """temp = []
        for line in fl.readline().split()[0].split(";"):
            temp.append(list(map(int, line.split(","))))
        t = np.array(temp)"""
        v_1 = int(fl.readline().split()[0])
        v_2 = list(df_data.iloc[:, 3].values.astype('int'))
        #v_2 = list(map(int, fl.readline().split()[0].split(",")))
        max_s = int(fl.readline().split()[0])
        virtual_per_cost = list(df_data.iloc[:, 2].values.astype('int'))
        #virtual_per_cost = list(map(int, fl.readline().split()[0].split(",")))

    with open('info_center.txt', "r", encoding='utf-8') as fl:
        c_coords = []
        for line in fl.readlines()[1:]:
            c_coords.append(tuple(map(float, line.split()[1:])))

    with open('info_depot.txt', "r", encoding='utf-8') as fl:
        j_coords = []
        f = []
        for line in fl.readlines()[1:]:
            j_coords.append(tuple(map(float, line.split()[1:3])))
            f.append(int(line.split()[-1]))

    with open('info_express.txt', "r", encoding='utf-8') as fl:
        m_coords = []
        t_res = []
        for line in fl.readlines()[1:]:
            m_coords.append(tuple(map(float, line.split()[1:3])))
            t_res.append(int(line.split()[-1]))
        for coord in j_coords:
            m_coords.append(coord)
            t_res.append(0)

    with open('info_cust.txt', "r", encoding='utf-8') as fl:
        i_coords = []
        D = []
        alpha = []
        T = []
        for line in fl.readlines()[1:]:
            temp = line.split()
            i_coords.append(tuple(map(float, temp[1:3])))
            #D.append(list(map(float, temp[3].split(","))))
            #alpha.append(list(map(float, temp[4].split(","))))
            #T.append(list(map(float, temp[5].split(","))))
        D = np.array(D)
        alpha = np.array(alpha)
        T = np.array(T)
        D = np.zeros((I_num, K_num))
        alpha = np.zeros((I_num, K_num))
        T = np.zeros((I_num, K_num))
        for _, cust_n, prod_n, d_n, alpha_n, t_n in df_data_info_cust.itertuples(index=False):
            D[cust_n,prod_n] = d_n
            alpha[cust_n,prod_n] = alpha_n
            T[cust_n,prod_n] = t_n

    C = {i for i in range(C_num)}
    I = {i for i in range(I_num)}
    J = {i for i in range(J_num)}
    M = {i for i in range(M_num)}
    K = {i for i in range(K_num)}
    dist_ci = get_dist(c_coords, i_coords)
    dist_jm = get_dist(j_coords, m_coords)
    dist_ji = get_dist(j_coords, i_coords)


    c_4 = np.zeros([len(I), len(M), len(J), len(K)])
    for i in range(len(c_4)):
        for m in range(len(c_4[0])):
            for j in range(len(c_4[0, 0])):
                for k in range(len(c_4[0, 0, 0])):
                    if m < M_num - virtual_num:
                        c_4[i, m, j, k] = (dist_jm[j, m] + dist_ji[j, i]) * per_cost[k]
                    else:
                        c_4[i, m, j, k] = (dist_jm[j, m] + dist_ji[j, i]) * virtual_per_cost[k]

    d_jm = dist_jm
    d_ji = dist_ji


    "h = np.random.randint(10,20,size=(len(J), len(K)))"

    "D = np.random.randint(0,2,size=(len(I),len(K)))"

    "t = np.random.randint(1,3, size=(len(J), len(K)))"

    """v_1 = 1 
    v_2 = [random.randint(10,20)*0.1 for _ in K]"""
    tao_jm = d_jm / v_1
    tao_jik = np.zeros((len(J), len(I), len(K)))
    for k in K:
        tao_jik[:, :, k] = (d_ji / v_2[k]).round(2)

    "alpha = np.random.random((len(I), len(K))).round(2)*0.5"

    "t_res = [random.randint(1,10)*0.1+1 for _ in M] "

    "T = np.ones((len(I), len(K))) * 80"

    "max_s = 5"
    L = dict()  
    for j in J:
        for k in K:
            L[j, k] = set(i + 1 for i in range(max_s))

    omega = T  

    miu_lb = 0
    miu_ub = float(np.sum(D)) * 10

    J_ik = dict()  # 能服务客户i的所有本地库备选点集合
    I_jk = dict()  # 本地库j能服务的所有客户集合
    for i in I:
        for j in J:
            for k in K:
                if (i, k) not in J_ik.keys():
                    J_ik[i, k] = set()
                if (j, k) not in I_jk.keys():
                    I_jk[j, k] = set()
                if tao_jik[j, i, k] < omega[i, k]:
                    J_ik[i, k].add(j)
                    I_jk[j, k].add(i)

    ijk_b = dict()
    bjk_i = dict()
    for i in I:
        for j in J:
            for k in K:
                ijk_b[i, j, k] = alpha[i, k]
    for i in I:
        for j in J:
            for k in K:
                b = ijk_b[i, j, k]
                if (b, j, k) not in bjk_i.keys():
                    bjk_i[b, j, k] = i

    miu = dict()
    for i in I:
        for j in J:
            for k in K:
                b = ijk_b[i, j, k]
                for s in L[j, k]:
                    miu[i, j, k, s] = binary_get(b, s=s, lb=miu_lb, ub=miu_ub)

    pr.prod_type_num = K_num
    for i in range(J_num):
        wh = WareHouse(i, j_coords[i], f[i], list(h[i]), list(t[i]))
        pr.warehouse_list.append(wh)
        pr.warehouse_set.add(wh)
    for _ in K:
        pr.product_type_set.append(set())
        pr.product_type_list.append(list())
    for i in range(I_num):
        temp_set = set()
        temp_list = list()
        for j in range(K_num):
            if D[i, j] == 0:
                continue
            prod = Product(i, i_coords[i], j, D[i, j], alpha[i, j], T[i, j])
            pr.product_list.append(prod)
            pr.product_set.add(prod)
            pr.product_type_list[prod.type].append(prod)
            pr.product_type_set[prod.type].add(prod)
            pr.num_type_product_dict[prod.num, prod.type] = prod
            temp_set.add(prod)
            temp_list.append(prod)
    for i in range(M_num - virtual_num):
        ex = Express(i, m_coords[i], t_res[i])
        pr.express_set.add(ex)
        pr.express_list.append(ex)
    pr.per_cost = list(per_cost)
    pr.vir_per_cost = list(virtual_per_cost)
    pr.dist_cp = dist_ci
    pr.dist_ew = dist_jm.T[: M_num - virtual_num]
    pr.dist_wp = dist_ji
    b_set_dict = dict()
    for (i, j, k, s), v in miu.items():
        b = ijk_b[i, j, k]
        if b not in pr.max_demands.keys():
            pr.max_demands[b] = list()
            b_set_dict[b] = set()
        if (s, v) not in b_set_dict[b]:
            b_set_dict[b].add((s, v))
            pr.max_demands[b].append((s, v))
    for v in pr.max_demands.values():
        v.sort()
    pr.v_ew = v_1
    pr.v_wp = v_2

    for prod in pr.product_set:
        prod.min_dist_wh = pr.warehouse_list[np.argmin(pr.dist_wp[:, prod.num])]

    for prod in pr.product_list:
        assert isinstance(prod, Product)
        for wh in pr.warehouse_list:
            assert isinstance(wh, WareHouse)
            if pr.dist_wp[wh.num, prod.num] / pr.v_wp[prod.type] <= prod.time_limit:
                prod.able_to_cover.add(wh)
    return pr
