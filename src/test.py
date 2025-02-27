import torch
import numpy as np

import argparse
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm, trange
from pathlib import Path
import random


from utils import load_config_test, set_seeds
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="train"
    )

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument(
        "--load_model", type=str, default=None, help="path of pre-trained model"
    )
    parser.add_argument("--logging_path", type=str, default=None)

    parser.add_argument(
        "--config_root",
        type=str,
        default="config/",
        help="root path to config file",
    )
    parser.add_argument("--scene", type=str, default="eth", help="scene name")

    parser.add_argument(
        "--aug_scene", action="store_true", help="trajectron++ augmentation"
    )
    parser.add_argument(
        "--w_mse", type=float, default=0, help="loss weight of mse_loss"
    )

    parser.add_argument("--clusterGMM", action="store_true")
    parser.add_argument(
        "--cluster_method", type=str, default="kmeans", help="clustering method"
    )
    parser.add_argument("--cluster_n", type=int, help="n cluster centers")
    parser.add_argument(
        "--cluster_name", type=str, default="", help="clustering model name"
    )
    parser.add_argument(
        "--manual_weights", nargs='+', default=None, type=int)

    parser.add_argument("--var_init", type=float, default=0.7, help="init var")
    parser.add_argument("--learnVAR", action="store_true")

    return parser.parse_args()


def k_means(batch_x, ncluster=20, iter=10):
    B, N, D = batch_x.size()
    batch_c = torch.Tensor().cuda()
    for i in trange(B):
        x = batch_x[i]
        c = x[torch.randperm(N)[:ncluster]]
        for i in range(iter):
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            c[nanix] = x[torch.randperm(N)[:ndead]]

        batch_c = torch.cat((batch_c, c.unsqueeze(0)), dim=0)
    return batch_c


def run_inference(cfg, model, metrics, data_loader):
    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []
        obs_list = []

        for i, data_dict in enumerate(tqdm(data_loader, leave=False)):
            pred_list_i = []
            gt_list_i = []

            data_dict = {
                k: (
                    data_dict[k].cuda()
                    if isinstance(data_dict[k], torch.Tensor)
                    else data_dict[k]
                )
                for k in data_dict
            }

            dist_args = model.encoder(data_dict)

            if cfg.MGF.ENABLE:
                base_pos = model.get_base_pos(data_dict).clone()
            else:
                base_pos = (
                    model.get_base_pos(data_dict)[:, None]
                    .expand(-1, cfg.MGF.POST_CLUSTER, -1)
                    .clone()
                )  # (B, 20, 2)
            dist_args = dist_args[:, None].expand(-1, cfg.MGF.POST_CLUSTER, -1, -1)

            sampled_seq = model.flow.sample(
                base_pos, cond=dist_args, n_sample=cfg.MGF.POST_CLUSTER
            )

            dict_list = []
            for i in range(cfg.MGF.POST_CLUSTER):
                data_dict_i = deepcopy(data_dict)
                data_dict_i[("pred_st", 0)] = sampled_seq[:, i]
                if torch.sum(torch.isnan(data_dict_i[("pred_st", 0)])):
                    data_dict_i[("pred_st", 0)] = torch.where(
                        torch.isnan(data_dict_i[("pred_st", 0)]),
                        data_dict_i["obs_st"][:, 0, None, 2:4].expand(
                            data_dict_i[("pred_st", 0)].size()
                        ),
                        data_dict_i[("pred_st", 0)],
                    )
                dict_list.append(data_dict_i)

            dict_list = metrics.denormalize(dict_list)

            for data_dict in dict_list:
                obs_list_i = data_dict["obs"].cpu().numpy()
                pred_traj_i = data_dict[("pred", 0)].cpu().numpy()  # (B,12,2)
                pred_list_i.append(pred_traj_i)
                gt_list_i = data_dict["gt"].cpu().numpy()
            pred_list_i = np.array(pred_list_i).transpose(1, 0, 2, 3)

            pred_list.append(pred_list_i)
            gt_list.append(gt_list_i)
            obs_list.append(obs_list_i)

    pred_list = np.concatenate(pred_list, axis=0)
    gt_list = np.concatenate(gt_list, axis=0)
    obs_list = np.concatenate(obs_list, axis=0)

    pred_list_flatten = torch.Tensor(
        pred_list.reshape(pred_list.shape[0], cfg.MGF.POST_CLUSTER, -1)
    ).cuda()
    pred_list = (
        k_means(pred_list_flatten).cpu().numpy().reshape(pred_list.shape[0], 20, -1, 2)
    )

    return obs_list, gt_list, pred_list


def evaluate_metrics(args, gt_list, pred_list):
    l2_dis = np.sqrt(((pred_list - gt_list[:, np.newaxis, :, :]) ** 2).sum(-1))
    minade = l2_dis.mean(-1).min(-1)
    minfde = l2_dis[:, :, -1].min(-1)
    if args.scene == "eth":
        minade /= 0.6
        minfde /= 0.6
    elif args.scene == "sdd":
        minade *= 50
        minfde *= 50
    return minade.mean(), minfde.mean()


def test():
    args = parse_args()
    scene = args.scene
    args.load_model = f"./checkpoint/{scene}.ckpt"
    args.config_file = f"./config/{scene}.yml"
    cfg = load_config_test(args)

    args.clusterGMM = cfg.MGF.ENABLE
    args.cluster_n = cfg.MGF.CLUSTER_N
    args.var_init = cfg.MGF.VAR_INIT
    args.learn_var = cfg.MGF.VAR_LEARNABLE

    model = Build_Model(cfg, args)
    model.load(Path(args.load_model))
    data_loader = unified_loader(cfg, rand=False, split="test")
    metrics = Build_Metrics(cfg)

    for i_trial in range(3):
        set_seeds(random.randint(0,1000))
        _, gt_list, pred_list = run_inference(cfg, model, metrics, data_loader)
        minade, minfde = evaluate_metrics(args, gt_list, pred_list)
        print(f"{args.scene} test {i_trial}:\n {minade}/{minfde}")


if __name__ == "__main__":
    test()
