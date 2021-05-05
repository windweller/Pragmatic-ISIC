import os
import random
import torch
from os.path import join as pjoin
from tqdm import tqdm
import utils.arg_parser
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image

from rsa_eval import load_inc_rsa_model
from rsa import BirdDistractorDataset

def get_args(argstring=None, verbose=False):
    if argstring is None:
        argstring = "--model gve --dataset cub --eval ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth"
    args = utils.arg_parser.get_args(argstring)
    if verbose:
        # Print arguments
        utils.arg_parser.print_args(args)

    return args


def plot_images(target_img, dis_cell, sim_cell, issue_name, max_display=5):
    # 3 rows: target image (first row)
    # then 5 distractors, 5 similars

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, max_display, figsize=(20, 10))

    # Adjust the vertical spacing
    hspace = 0.2
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # plot the target image
    axes.flat[0].imshow(Image.open("./data/cub/images/" + target_img))
    axes.flat[0].set_xlabel("Target Image \n" + target_img.split("/")[0])
    axes.flat[0].set_xticks([])
    axes.flat[0].set_yticks([])
    for i in range(1, max_display):
        fig.delaxes(axes.flat[i])

    for i in range(min(len(sim_cell), max_display)):
        img_id = sim_cell[i]
        axes.flat[max_display + i].imshow(Image.open("./data/cub/images/" + img_id))
        axes.flat[max_display + i].set_xlabel("Similar Image \n" + img_id.split("/")[0])
        axes.flat[max_display + i].set_xticks([])
        axes.flat[max_display + i].set_yticks([])

    for i in range(min(len(dis_cell), max_display)):
        img_id = dis_cell[i]
        axes.flat[max_display * 2 + i].imshow(Image.open("./data/cub/images/" + img_id))
        axes.flat[max_display * 2 + i].set_xlabel("Dissimilar Image \n" + img_id.split("/")[0])
        axes.flat[max_display * 2 + i].set_xticks([])
        axes.flat[max_display * 2 + i].set_yticks([])

    fig.suptitle("Issue: {}".format(issue_name))
    # Show the plot
    plt.show()

class CUBPartitionDataset(object):
    def __init__(self, cell_select_strategy=None, argstring=None, return_labels=True):
        self.cell_select_strategy = cell_select_strategy

        self.args = args = get_args(argstring)
        self.device = torch.device('cuda:{}'.format(args.cuda_device) if
                                            torch.cuda.is_available() and not args.disable_cuda else 'cpu')

        self.image_folder = './data/cub/'

        # load issues here
        self.attr_issue_matrix = np.load(pjoin(self.image_folder, "strict_cap_issue_matrix.npz"))['data']
        self.mat_idx_to_imgid = json.load(open(pjoin(self.image_folder, "mat_idx_to_imgid.json")))
        self.imgid_to_mat_idx = json.load(open(pjoin(self.image_folder, "imgid_to_mat_idx.json")))
        self.issue_vocab = json.load(open(pjoin(self.image_folder, "issue_vocab.json")))


    def get_cells_by_partition(self, img_id, issue_id, max_cap_per_cell=60, cell_select_strategy=None,
                               no_similar=False):

        assert type(issue_id) == int and 0 <= issue_id <= len(self.issue_vocab)
        assert issue_id in self.get_valid_issues(img_id)[0], "question_id not valid!"

        sim_img_indices = self.attr_issue_matrix[:, issue_id].nonzero()[0].tolist()
        dis_img_indices = (self.attr_issue_matrix[:, issue_id] == 0).nonzero()[0].tolist()

        sampled_sim_img_indices = random.sample(sim_img_indices, max_cap_per_cell)
        sampled_dis_img_indices = random.sample(dis_img_indices, max_cap_per_cell)

        sim_img_ids = [self.mat_idx_to_imgid[i] for i in sampled_sim_img_indices]
        dis_img_ids = [self.mat_idx_to_imgid[i] for i in sampled_dis_img_indices]

        if no_similar:
            sim_img_ids = []

        return dis_img_ids, sim_img_ids, 0, sampled_dis_img_indices, sampled_dis_img_indices

    def get_valid_issues(self, img_id):
        issue_row = self.attr_issue_matrix[self.imgid_to_mat_idx[img_id]]
        valid_issue_ids = issue_row.nonzero()[0].tolist()

        labels = []
        for issue_id in valid_issue_ids:
            labels.append(self.issue_vocab[issue_id])

        return valid_issue_ids, labels

    def clone(self, b):
        a = np.empty_like(b)
        a[:] = b
        return a


def generate_caption_for_test(save_file_prefix, max_cap_per_cell=40, rationality=20, entropy_penalty_alpha=0.4,
                              no_retry=False, no_similar=False):

    open(save_file_prefix + "_gen_captions.json", 'w').close()
    open(save_file_prefix + "_sampled_partitions.json", 'w').close()

    cub_partition = CUBPartitionDataset()
    rsa_dataset = BirdDistractorDataset(randomized=True)
    rsa_model = load_inc_rsa_model(rsa_dataset)

    test_ids = []
    with open(pjoin(cub_partition.image_folder, 'test.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    img_id_to_caption = {}
    img_id_to_partition_idx = {}

    for imgid in tqdm(test_ids):

        img_issues, issue_names = cub_partition.get_valid_issues(imgid)
        img_id_to_caption[imgid] = {}
        img_id_to_partition_idx[imgid] = {}

        for issue_id, issue_name in zip(img_issues, issue_names):
            dis_cell2, sim_cell2, _, dis_indices, sim_indices = cub_partition.get_cells_by_partition(
                imgid, issue_id, max_cap_per_cell=max_cap_per_cell)

            if no_similar:
                sim_cell2 = []

            cap = rsa_model.greedy_pragmatic_speaker_free(
                    [imgid] + sim_cell2 + dis_cell2,
                    num_sim=len(sim_cell2), rationality=rationality,
                    speaker_prior=True, entropy_penalty_alpha=entropy_penalty_alpha)[0]

            img_id_to_caption[imgid][issue_id] = cap
            img_id_to_partition_idx[imgid][issue_id] = [dis_indices, sim_indices]

    json.dump(img_id_to_caption, open(save_file_prefix+"_gen_captions.json", 'w'))
    json.dump(img_id_to_partition_idx, open(save_file_prefix+"_sampled_partitions.json", 'w'))

def generate_literal_caption_for_test(save_file_prefix):
    cub_partition = CUBPartitionDataset()
    rsa_dataset = BirdDistractorDataset(randomized=True)
    rsa_model = load_inc_rsa_model(rsa_dataset)

    open(save_file_prefix + "_gen_captions.json", 'w').close()

    test_ids = []
    with open(pjoin(cub_partition.image_folder, 'test.txt')) as f:
        for line in f:
            test_ids.append(line.strip())

    img_id_to_caption = {}
    img_id_to_partition_idx = {}

    for imgid in tqdm(test_ids):
        img_id_to_caption[imgid] = {}
        img_id_to_partition_idx[imgid] = {}
        img_issues, issue_names = cub_partition.get_valid_issues(imgid)

        cap = rsa_model.semantic_speaker([imgid])[0]
        for issue_id, issue_name in zip(img_issues, issue_names):
            img_id_to_caption[imgid][issue_id] = cap

    json.dump(img_id_to_caption, open(save_file_prefix + "_gen_captions.json", 'w'))

if __name__ == '__main__':
    # both S1-Q, and S1-QH get "retry"
    parser = argparse.ArgumentParser()
    parser.add_argument('--rationality', type=float, default=10, help="raitionality")
    parser.add_argument('--entropy', type=float, default=0.4, help="raitionality")
    parser.add_argument('--max_cell_size', type=int, default=40, help="cell size")

    parser.add_argument('--exp_num', type=int, help="which evaluation experiment to run; this helps parallelization")
    parser.add_argument('--root_dir', type=str, default="./results/", help="format is ./results/{}, no slash aft+er")
    parser.add_argument('--file_prefix', type=str, default="{}", help="prefix hyperparameter for the run")
    parser.add_argument('--run_time', type=int, default=1, help="format is ./results/{}, no slash after")
    args = parser.parse_args()

    for time in range(args.run_time):

        os.makedirs(pjoin(args.root_dir, "random_run_{}".format(time), "test.txt"), exist_ok=True)

        save_dir = pjoin(args.root_dir, "random_run_{}".format(time))

        if args.exp_num == 0:
            generate_literal_caption_for_test(args.save_dir.format("S0"))

        if args.exp_num == 1:
            generate_caption_for_test(save_dir + "/S1", max_cap_per_cell=args.max_cell_size,
                                              rationality=args.rationality,
                                              entropy_penalty_alpha=0, no_similar=True)
        if args.exp_num == 2:
            generate_caption_for_test(save_dir + "/S1_Q", max_cap_per_cell=args.max_cell_size,
                                      rationality=args.rationality,
                                      entropy_penalty_alpha=0)

        if args.exp_num == 3:
            generate_caption_for_test(save_dir + "/S1_QH", max_cap_per_cell=args.max_cell_size,
                                      rationality=args.rationality,
                                      entropy_penalty_alpha=args.entropy)
