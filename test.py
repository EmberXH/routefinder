import argparse
import os
import pickle
import time
import warnings

import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE

import progressbar

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


PROBLEM_TYPE = 'qs_vb'
PROBLEM_SIZE = 100
DEG_OF_DYNAS = [0.3, 0.4]  # 0.3 0.4 0.6
APPEAR_EARLY_RATIO = 0.5
def test(
        policy,
        td,
        env,
        num_augment=1,
        augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
        num_starts=None,
        device="cuda",
):
    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts
            if num_augment == 1:  # add
                n_start = 1

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)
            # Evaluate policy
            if 'problem_type' in td.keys():
                out = get_dyn_out(td, env, td['problem_type'])
            else:
                out, tj = policy(td, env, phase="test", num_starts=n_start, return_actions=True)

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            if out["reward"].dim() != 1:
                reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = (
                            100
                            * (-max_aug_reward - torch.abs(costs_bks))
                            / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})  # Dummy value

            return out


def get_dyn_out(td, env, problem_type):
    # from ._data_tj import DVRPTW_TJ_Dataset
    # from ._data_qs import DVRPTW_QS_Dataset
    # from ._data_qc import DVRPTW_QC_Dataset
    # from ._data_vb import DVRPTW_VB_Dataset
    # from ._data_nr import DVRPTW_NR_Dataset
    # from ._data_nr_tj import DVRPTW_NR_TJ_Dataset
    # from ._data_qs_vb import DVRPTW_QS_VB_Dataset
    function_map = {
        'DVRPTW_NR_Dataset': solve_loop_nr,
        'DVRPTW_QC_Dataset': solve_loop_qc,
        'DVRPTW_QS_Dataset': solve_loop_qs,
        'DVRPTW_VB_Dataset': solve_loop_vb,
        'DVRPTW_QS_VB_Dataset': solve_loop_qs_vb
    }
    f = function_map.get(problem_type, lambda: print("无效的问题类型"))
    return f(td, env)


def solve_loop_nr(td, env):
    partial = None
    ori_dyn_time = td['dyn_time'][0].tolist()
    dyn_time  = sorted(ori_dyn_time, reverse=True)
    while dyn_time and dyn_time[-1] == 0.0:
        dyn_time.pop()
    while dyn_time:
        out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
        partial = [[] for _ in range(out["veh_num"])]
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
        td['demand_linehaul'] = td['demand_linehaul'][:, 1:]
        td['demand_backhaul'] = td['demand_backhaul'][:, 1:]
        td = env.reset(td).to(device)
        dyn_cust_mask = torch.ones(1, len(ori_dyn_time), dtype=torch.bool)
        for idx, ot in enumerate(ori_dyn_time):
            if t < ot:
                dyn_cust_mask[0][idx] = False
        td['dyn_cust_mask'] = dyn_cust_mask
    out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
    return out

def solve_loop_qc(td, env):
    partial = None
    ori_dyn_time = td['dyn_time'][0].tolist()
    dyn_time  = sorted(ori_dyn_time, reverse=True)
    while dyn_time and dyn_time[-1] == 0.0:
        dyn_time.pop()
    while dyn_time:
        out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
        partial = [[] for _ in range(out["veh_num"])]
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
        td['demand_linehaul'] = td['demand_linehaul'][:, 1:]
        td['demand_backhaul'] = td['demand_backhaul'][:, 1:]
        td = env.reset(td).to(device)
        dyn_cust_mask = torch.ones(1, len(ori_dyn_time), dtype=torch.bool)
        for idx, ot in enumerate(ori_dyn_time):
            if t >= ot and ot > 0:
                dyn_cust_mask[0][idx] = False
        td['dyn_cust_mask'] = dyn_cust_mask
    out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
    return out


def solve_loop_qs(td, env):
    partial = None
    hidden_changed = td['dyn_time'][0] > 0
    changed = []
    for i in range(len(hidden_changed)):
        if hidden_changed[i]:
            changed.append((i, td['dyn_time'][0][i]))
    changed = sorted(changed, key=lambda x: x[1], reverse=True)
    changed_dem = td['changed_dem']
    while changed:
        out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
        if not out:
            return None
        partial = [[] for _ in range(out["veh_num"])]
        for t, veh_i, to_n in traj:
            if t < changed[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while changed and t >= changed[-1][1]:
                c = changed.pop()
                td['demand_linehaul'][0, c[0]] = changed_dem[0, c[0]]
            break
        else:
            c = changed.pop()
            td['demand_linehaul'][0, c[0]] = changed_dem[c[0]]
        td['demand_linehaul'] = td['demand_linehaul'][:, 1:]
        td['demand_backhaul'] = td['demand_backhaul'][:, 1:]
        td = env.reset(td).to(device)
    out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
    return out

def solve_loop_vb(td, env):
    b_time = td['b_time']
    partial = None
    for _ in range(len(b_time)):
        out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
        partial = [[] for _ in range(out["veh_num"])]
        for t, veh_i, to_n in traj:
            if t < b_time[0][-1] and to_n > 0:
                partial[veh_i].append(to_n)
        td['demand_linehaul'] = td['demand_linehaul'][:, 1:]
        td['demand_backhaul'] = td['demand_backhaul'][:, 1:]
        td = env.reset(td).to(device)
    td['b_happen'] = True
    out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
    return out

def solve_loop_qs_vb(td, env):
    partial = None
    hidden_changed = td['dyn_time'][0] > 0
    changed = []
    for i in range(len(hidden_changed)):
        if hidden_changed[i]:
            changed.append((i, td['dyn_time'][0][i]))
    changed.append((-1, td['b_time'].squeeze()))
    changed = sorted(changed, key=lambda x: x[1], reverse=True)
    changed_dem = td['changed_dem']
    while changed:
        out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
        if not out:
            return None
        partial = [[] for _ in range(out["veh_num"])]
        for t, veh_i, to_n in traj:
            if t < changed[-1][1]:
                if to_n > 0:
                    partial[veh_i].append(to_n)
                continue
            while changed and t >= changed[-1][1]:
                c = changed.pop()
                if c[0] == -1:
                    td['b_happen'] = torch.ones(1, 1, dtype=torch.bool)
                else:
                    td['demand_linehaul'][0, c[0]] = changed_dem[0, c[0]]
            break
        else:
            c = changed.pop()
            if c[0] == -1:
                td['b_happen'] = torch.ones(1, 1, dtype=torch.bool)
            else:
                td['demand_linehaul'][0, c[0]] = changed_dem[c[0]]
        td['demand_linehaul'] = td['demand_linehaul'][:, 1:]
        td['demand_backhaul'] = td['demand_backhaul'][:, 1:]
        td = env.reset(td).to(device)
    out, traj = policy(td, env, phase="test", num_starts=1, return_actions=True, partial=partial)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1)  # 原为1000
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint",
    )

    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.datasets == 'dyn':
        # 解决动态问题
        problem_type = PROBLEM_TYPE
        problem_size = PROBLEM_SIZE
        deg_of_dynas = DEG_OF_DYNAS  # 0.3 0.4 0.6
        appear_early_ratio = APPEAR_EARLY_RATIO
        data_paths = []
        for deg_of_dyna in deg_of_dynas:
            data_path = "./dyn_data/{}_n{}m{}_{}_{}_10000/norm_data.pyth".format(problem_type, problem_size,
                                                                                 problem_size // 5,
                                                                                 deg_of_dyna,
                                                                                 appear_early_ratio)
            data_paths.append(data_path)
    else:
        if opts.datasets is not None:
            data_paths = opts.datasets.split(",")
        else:
            # list recursively all npz files in data/
            data_paths = []
            for root, _, files in os.walk("data"):
                for file in files:
                    # print(file)
                    if "test" not in root:
                        continue
                    if file.endswith(".npz"):
                        if opts.remove_mixed_backhaul and "m" in root:
                            continue
                        # if name in 50 or 100, append
                        if str(opts.size) in file:
                            if file == "50.npz" or file == "100.npz":
                                data_paths.append(os.path.join(root, file))
            assert len(data_paths) > 0, "No datasets found. Check the data directory."
            data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
            print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

    # Load model
    print("Loading checkpoint from ", opts.checkpoint)
    if "mvmoe" in opts.checkpoint:
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint:
        BaseLitModule = MTPOMO
    elif "moe" in opts.checkpoint:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    model = BaseLitModule.load_from_checkpoint(
        opts.checkpoint, map_location="cpu", strict=False
    )

    env = MTVRPEnv()
    policy = model.policy.to(device).eval()  # Use mixed precision if supported

    results = {}
    for dataset in data_paths:

        print(f"Loading {dataset}")
        td_test = env.load_data(dataset)  # this also adds the bks cost
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        for batch in tqdm(dataloader):
            td_test = env.reset(batch).to(device)
            o = test(policy, td_test, env, device=device)
            res.append(o)
            if len(res) > 10:
                break
        out = {}
        if "max_aug_reward" not in res[0].keys():
            print(f"AVG COST: {-torch.stack([item['reward'] for item in res]).sum() / len(res)}")
            print(f"SERV RATE: {torch.stack([item['serv_rate'] for item in res]).sum() / len(res)}")
        else :
            out["max_aug_reward"] = torch.cat([o["max_aug_reward"] for o in res])
            out["gap_to_bks"] = torch.cat([o["gap_to_bks"] for o in res])

            inference_time = time.time() - start

            dataset_name = dataset.split("/")[-3].split(".")[0].upper()
            print(
                f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Gap: {out['gap_to_bks'].mean().item():.3f}% | Inference time: {inference_time:.3f} s"
            )

            if results.get(dataset_name, None) is None:
                results[dataset_name] = {}
            results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
            results[dataset_name]["gap"] = out["gap_to_bks"].mean().item()
            results[dataset_name]["inference_time"] = inference_time

        if opts.save_results:
            # Save results with checkpoint name under results/main/
            checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
            savedir = f"results/main/{opts.size}/"
            os.makedirs(savedir, exist_ok=True)
            pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))
