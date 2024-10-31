import time

from pyvrp import Model
import matplotlib.pyplot as plt

from pyvrp.plotting import plot_coordinates
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution
from marpdan.problems import *
import torch
from tqdm import tqdm

PROBLEM_TYPE = 'nr_tj'
PROBLEM_SIZE = 100
DEG_OF_DYNAS = [0.3]  # 0.3 0.4 0.6
APPEAR_EARLY_RATIO = 0.5
TEST_INSTANCE_NUM = 10000


def load_pyth_data(filename, index):
    data = torch.load(filename)
    scale = 100
    x_dict = {}
    x_dict['locs'] = [tuple(row.tolist()) for row in data.nodes[index, :, :2] * scale]
    x_dict['dur_matrix'] = (torch.cdist(data.nodes[index, :, :2] * scale, data.nodes[index, :, :2] * scale,
                                        p=2) / data.veh_speed).tolist()
    x_dict['demand'] = data.nodes[index, :, 2].tolist()
    # 可加入距离矩阵
    x_dict['time_windows'] = [tuple(row.tolist()) for row in data.nodes[index, :, 3:5] * scale]
    x_dict['dyn_time'] = data.nodes[index, :, -1] * scale
    if type(data) == DVRPTW_QC_Dataset:
        x_dict['qc_time'] = data.nodes[index, :, -1] * scale
    if type(data) == DVRPTW_NR_Dataset or type(data) == DVRPTW_NR_TJ_Dataset:
        x_dict['nr_time'] = data.nodes[index, :, -1] * scale
    if type(data) == DVRPTW_QS_Dataset or type(data) == DVRPTW_QS_VB_Dataset:
        x_dict['qs_time'] = data.nodes[index, :, -1] * scale
        x_dict['changed_dem'] = data.changed_dem[index]
    if type(data) == DVRPTW_VB_Dataset or type(data) == DVRPTW_QS_VB_Dataset:
        x_dict['b_time'] = data.b_time[index] * scale
        x_dict['b_veh_idx'] = data.b_veh_idx[index]
    if type(data) == DVRPTW_TJ_Dataset or type(data) == DVRPTW_NR_TJ_Dataset:
        x_dict['distances_with_tj'] = data.distances_with_tj[index] * scale
        x_dict['distances_with_tj'].fill_diagonal_(0)
        x_dict['tj_time'] = data.tj_time[index] * scale
    x_dict['problem_type'] = type(data).__name__
    return x_dict


def per_solve(data, partial, t, start_times):
    COORDS = data['locs']
    DEMANDS = data['demand']
    TIME_WINDOWS = data['time_windows']
    m = Model()
    veh_num = int(len(data['locs']) / 5)
    if "b_time" in data.keys() and t >= data['b_time']:
        veh_num -= len(data['b_time'])
    partial_list = []
    depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1])
    partial_depots = {}
    cur_time = 0
    if partial:
        num = 0
        for idx, per_p in enumerate(partial):
            if len(per_p) == 0:
                continue
            num += 1
            cap = 1
            cur = start_times[idx]
            pre_node = 0
            for n in per_p:
                cur += data['dur_matrix'][pre_node][n]
                pre_node = n
                partial_list.append(n)
                cap -= DEMANDS[n]
            cur_time = max(cur, cur_time)
            partial_depots[per_p[-1]] = (cap, idx)
        if veh_num > num:
            m.add_vehicle_type(
                num_available=veh_num - num,
                capacity=1
            )
    else:
        m.add_vehicle_type(
            num_available=veh_num,
            capacity=1
        )
    partial_set = set(partial_list)
    clients = []
    for idx in range(1, len(COORDS)):
        if idx in partial_depots.keys():
            partial_depot = m.add_depot(x=COORDS[idx][0], y=COORDS[idx][1])
            if "b_time" in data.keys() and t >= data['b_time'] and data['b_veh_idx'] == partial_depots[idx][1]:
                pass
            else:
                m.add_vehicle_type(
                    num_available=1,
                    capacity=partial_depots[idx][0],
                    start_depot=partial_depot,
                    name=str(idx)
                )
            client = partial_depot
        elif idx in partial_set or ("nr_time" in data.keys() and (data['nr_time'][idx] - t) > 0) or (
                "qc_time" in data.keys() and data['qc_time'][idx] != 0 and data['qc_time'][idx] <= t):
            client = m.add_client(
                x=COORDS[idx][0],
                y=COORDS[idx][1],
                tw_early=max(0, TIME_WINDOWS[idx][0] - cur_time),
                tw_late=max(0, TIME_WINDOWS[idx][1] - cur_time),
                delivery=DEMANDS[idx],
                prize=0,
                required=False
            )
        else:
            client = m.add_client(
                x=COORDS[idx][0],
                y=COORDS[idx][1],
                tw_early=max(0, TIME_WINDOWS[idx][0] - cur_time),
                tw_late=max(0, TIME_WINDOWS[idx][1] - cur_time),
                delivery=DEMANDS[idx]
            )
        clients.append(client)
    locations = [depot] + clients
    for frm_idx, frm in enumerate(locations):
        for to_idx, to in enumerate(locations):
            if frm == None or to == None:
                continue
            distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
            dur = data['dur_matrix'][frm_idx][to_idx]
            if distance == 0:
                dur = 0
            m.add_edge(frm, to, distance=distance, duration=dur)
    res = m.solve(stop=MaxRuntime(1), display=False)
    a = res.best.distance_cost()
    ap = []
    for w in res.best.routes():
        ap.append(w.visits())
    f = res.best.prizes()
    traj, start_times = generate_traj(res.best.routes(), data['dur_matrix'])
    return res, traj, start_times


def solve_loop_nr(data):
    partial = None
    start_times = None
    t = 0
    ori_dyn_time = data['nr_time'].tolist()
    dyn_time = sorted(ori_dyn_time, reverse=True)
    while dyn_time and dyn_time[-1] == 0.0:
        dyn_time.pop()
    while dyn_time:
        res, traj, start_times = per_solve(data, partial, t, start_times)
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < dyn_time[-1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while dyn_time and t >= dyn_time[-1]:
                dyn_time.pop()
            break
        else:
            t = dyn_time.pop()
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def solve_loop_qc(data):
    partial = None
    start_times = None
    t = 0
    ori_dyn_time = data['qc_time'].tolist()
    dyn_time = sorted(ori_dyn_time, reverse=True)
    while dyn_time and dyn_time[-1] == 0.0:
        dyn_time.pop()
    while dyn_time:
        res, traj, start_times = per_solve(data, partial, t, start_times)
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < dyn_time[-1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while dyn_time and t >= dyn_time[-1]:
                dyn_time.pop()
            break
        else:
            t = dyn_time.pop()
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def solve_loop_qs(data):
    partial = None
    start_times = None
    hidden_changed = data['qs_time'] > 0
    changed = []
    t = 0
    for i in range(len(hidden_changed)):
        if hidden_changed[i]:
            changed.append((i, data['qs_time'][i]))
    changed = sorted(changed, key=lambda x: x[1], reverse=True)
    changed_dem = data['changed_dem']
    while changed:
        res, traj, start_times = per_solve(data, partial, t, start_times)
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < changed[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while changed and t >= changed[-1][1]:
                c = changed.pop()
                data['demand'][c[0]] = changed_dem[c[0]]
            break
        else:
            c = changed.pop()
            t = c[1]
            data['demand'][c[0]] = changed_dem[c[0]]
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def solve_loop_vb(data):
    b_time = data['b_time']
    partial = None
    start_times = None
    t = 0
    for _ in range(len(b_time)):
        res, traj, start_times = per_solve(data, partial, t, start_times)
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < b_time[-1] and to_n > 0:
                partial[veh_i].append(to_n)
        t = b_time
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def solve_loop_tj(data):
    tj_time = data['tj_time']
    distance_with_tj = data['distances_with_tj']
    tjs = [[i.item(), j.item(), tj_time[i, j].item()] for i, j in torch.nonzero(tj_time)]
    tjs = sorted(tjs, key=lambda x: x[2], reverse=True)
    start_times = None
    partial = None
    t = 0
    while tjs:
        res, traj, start_times = per_solve(data, partial, t, start_times)
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < tjs[-1][2]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while tjs and t >= tjs[-1][2]:
                tj = tjs.pop()
                data['dur_matrix'][tj[0]][tj[1]] = distance_with_tj[tj[0]][tj[1]]
            break
        else:
            tj = tjs.pop()
            t = tj[2]
            data['dur_matrix'][tj[0]][tj[1]] = distance_with_tj[tj[0]][tj[1]]
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def solve_loop_nr_tj(data):
    partial = None
    start_times = None
    t = 0
    tj_time = data['tj_time']
    distance_with_tj = data['distances_with_tj']

    ori_dyn_time = [x for x in data['nr_time'].tolist() if x != 0]
    aprs = [[0, i] for i in ori_dyn_time]
    tjts = [[1, tj_time[i, j].item()] for i, j in torch.nonzero(tj_time)]
    event_times = sorted(aprs + tjts, key=lambda x: x[-1], reverse=True)
    while event_times:
        res, traj, start_times = per_solve(data, partial, t, start_times)
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < event_times[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while event_times and t >= event_times[-1][1]:
                et = event_times.pop()
                if et[0] == 1:
                    rows, cols = torch.where(tj_time == et[1])
                    for row, col in zip(rows, cols):
                        data['dur_matrix'][row][col] = distance_with_tj[row][col]
            break
        else:
            et = event_times.pop()
            t = et[1]
            if et[0] == 1:
                rows, cols = torch.where(tj_time == et[1])
                for row, col in zip(rows, cols):
                    data['dur_matrix'][row][col] = distance_with_tj[row][col]
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def solve_loop_qs_vb(data):
    b_time = data['b_time']
    partial = None
    start_times = None
    t = 0
    hidden_changed = data['qs_time'] > 0
    changed_dem = data['changed_dem']
    changed_times = []
    for i in range(len(hidden_changed)):
        if hidden_changed[i]:
            changed_times.append((i, data['qs_time'][i]))  # 节点，时间
    event_times = sorted(changed_times + [(-1, b_time)], key=lambda x: x[1], reverse=True)
    while event_times:
        res, traj, start_times = per_solve(data, partial, t, start_times)
        if not res:
            return None
        partial = [[] for _ in range(res.best.num_routes())]
        for t, veh_i, to_n in traj:
            if t < event_times[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while event_times and t >= event_times[-1][1]:
                et = event_times.pop()
                if et[0] != -1:
                    data['demand'][et[0]] = changed_dem[et[0]]
            break
        else:
            et = event_times.pop()
            t = et[1]
            if et[0] != -1:
                data['demand'][et[0]] = changed_dem[et[0]]
    res, traj, start_times = per_solve(data, partial, t, start_times)
    return cal_partial_cost(partial, data['dur_matrix']) + res.best.distance_cost() / 100, cal_serv_rate(res, partial)


def generate_traj(routes, dist_m):
    traj = []
    start_times = []
    for idx, route in enumerate(routes):
        route_list = route.visits()
        route_list.insert(0, route.start_depot())
        route_list.append(route.end_depot())
        t = route.start_time()
        start_times.append(t)
        traj.append((t, idx, route.start_depot()))
        for i, node in enumerate(route_list):
            from_n = node
            to_n = route_list[i + 1] if i < len(route_list) - 1 else None
            if to_n:
                t += dist_m[from_n][to_n]
                traj.append((t, idx, to_n))
    traj.sort()
    return traj, start_times


def cal_partial_cost(partial, dist_m):
    cost = 0
    for per_p in partial:
        from_n = 0
        for to_n in per_p:
            cost += dist_m[from_n][to_n]
            from_n = to_n
    return cost / 100


def cal_serv_rate(res, partial):
    serv_set = set()
    for p in partial:
        for i in p:
            serv_set.add(i)
    for w in res.best.routes():
        for i in w.visits():
            serv_set.add(i)
    return len(serv_set) / PROBLEM_SIZE


if __name__ == "__main__":
    problem_type = PROBLEM_TYPE
    problem_size = PROBLEM_SIZE
    deg_of_dynas = DEG_OF_DYNAS  # 0.3 0.4 0.6
    appear_early_ratio = APPEAR_EARLY_RATIO
    data_paths = []
    function_map = {
        'nr': solve_loop_nr,
        'qc': solve_loop_qc,
        'qs': solve_loop_qs,
        'vb': solve_loop_vb,
        'tj': solve_loop_tj,
        'nr_tj': solve_loop_nr_tj,
        'qs_vb': solve_loop_qs_vb
    }
    for deg_of_dyna in deg_of_dynas:
        data_path = "./dyn_data/{}_n{}m{}_{}_{}_10000/norm_data.pyth".format(problem_type, problem_size,
                                                                             problem_size // 5,
                                                                             deg_of_dyna,
                                                                             appear_early_ratio)
        cost = 0
        step = 0
        rate = 0
        for i in tqdm(range(TEST_INSTANCE_NUM)):
            data = load_pyth_data(data_path, i)
            solve_dyn = function_map.get(problem_type, lambda: print("无效的问题类型"))
            c, r = solve_dyn(data)
            cost += c
            rate += r
            step += 1
            if step > 1:
                break
        print(f"{data_path} AVG COST: {cost / step} AVG SERV RATE: {rate / step}")
