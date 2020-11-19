from collections import defaultdict

import random
import torch
import torch.nn.functional as F
from os.path import join as pjoin
from tqdm import tqdm

from models.model_loader import ModelLoader
from utils.data.data_prep import DataPreparation
from train.trainer_loader import TrainerLoader
import utils.arg_parser

from torch.distributions import Categorical

from rsa_utils import logsumexp
import numpy as np
import pickle

from scipy.stats import entropy

import matplotlib
import matplotlib.pyplot as plt

def get_args(argstring=None, verbose=True):
    if argstring is None:
        argstring = "--model gve --dataset cub --eval ./checkpoints/gve-cub-D2020-03-14-T18-04-04-G0-best-ckpt.pth"
    args = utils.arg_parser.get_args(argstring)
    if verbose:
        # Print arguments
        utils.arg_parser.print_args(args)

    return args


class BirdDistractorDataset(object):

    def __init__(self, cell_select_strategy=None, argstring=None, return_labels=True, randomized=True):
        """

        :param cell_select_strategy:
        :param argstring:
        :param return_labels:
        :param randomized: randomized means that we get a shuffled-once random attribute matrix. It's still deterministic.
        """

        self.cell_select_strategy = cell_select_strategy

        self.args = args = get_args(argstring)
        self.device = device = torch.device('cuda:{}'.format(args.cuda_device) if
                                            torch.cuda.is_available() and not args.disable_cuda else 'cpu')

        self.split_to_data = self.get_train_val_test_loader(args)

        self.image_folder = self.split_to_data['train'].image_path

        self.image_id_to_split = self.get_image_id_to_split()

        # load attributes here
        if not randomized:
            self.filename_to_cub_img_id, self.cub_img_id_to_filename, self.attribute_matrix = self.load_attribute_map(
                pjoin(self.image_folder, "attributes", "attribute_matrix.npy"))
        else:
            self.filename_to_cub_img_id, self.cub_img_id_to_filename, \
            self.attribute_matrix, self.random_idx_to_img_id = self.load_attribute_map_randomized(
                pjoin(self.image_folder, "attributes", "randomized_attribute_matrix.npy"),
                pjoin(self.image_folder, "attributes", "random_idx_to_file_idx.json"))
            self.img_id_to_random_idx = {}
            for k, v in self.random_idx_to_img_id.items():
                self.img_id_to_random_idx[v] = k

        self.randomized = randomized

        self.attr_vocab_ls, self.attr_to_attid = self.get_attribute_vocab()

        self.labels_path = pjoin(self.image_folder, "CUB_label_dict.p")

        # Redirect available questions based on segments or attrs
        self.segments_to_attr_id, self.attr_id_to_segment, self.q_id_to_segments = self.get_attr_segments()

        self.set_label_usage(return_labels)

    def get_attr_segments(self):
        segments_to_attr_id = defaultdict(list)
        attr_id_to_segment = {}

        for i, attr_name in enumerate(self.attr_vocab_ls):
            seg_name = attr_name.split("::")[0]
            segments_to_attr_id[seg_name].append(i)
            attr_id_to_segment[i] = seg_name

        q_id_to_segments = []
        for k in segments_to_attr_id.keys():
            q_id_to_segments.append(k)

        return segments_to_attr_id, attr_id_to_segment, q_id_to_segments

    def load_attribute_map_randomized(self, attribute_matrix_path=None, random_idx_to_file_idx_path=None):

        filename_to_cub_img_id = {}
        cub_img_id_to_filename = {}
        with open(pjoin(self.image_folder, "attributes", "images.txt")) as f:
            for line in f:
                cub_img_id, filename = line.strip().split()
                filename_to_cub_img_id[filename] = int(cub_img_id) - 1
                cub_img_id_to_filename[int(cub_img_id) - 1] = filename

        if attribute_matrix_path is not None:
            attribute_matrix = np.load(attribute_matrix_path)
            random_idx_to_file_idx = pickle.load(open(random_idx_to_file_idx_path, 'rb'))
            return filename_to_cub_img_id, cub_img_id_to_filename, attribute_matrix, random_idx_to_file_idx

        indices = list(range(len(filename_to_cub_img_id)))
        import random
        random.seed(12)
        random.shuffle(indices)
        file_idx_to_random_file_idx = indices  # 5:128 -> original file idx 5, now mapped to idx 128 on attribute matrix
        random_idx_to_file_idx = {}

        attribute_matrix = np.zeros((len(filename_to_cub_img_id), 312))
        with open(pjoin(self.image_folder, "attributes", "image_attribute_labels.txt")) as f:
            for line in tqdm(f, total=3677856):
                # some lines have error, we fix it
                if len(line.strip().split()) > 5:
                    tups = line.strip().split()
                    if tups[0] == '2275' or tups[0] == '9364':
                        # 2275 10 0 1 0  1.509
                        # img_id, att_id, is_present, certainty, 0, time
                        cub_img_id = int(tups[0]) - 1
                        att_id = int(tups[1]) - 1
                        is_present = int(tups[2])
                        random_cub_img_id = file_idx_to_random_file_idx[cub_img_id]
                        random_idx_to_file_idx[random_cub_img_id] = cub_img_id
                        attribute_matrix[random_cub_img_id, att_id] = is_present
                        continue

                cub_img_id, att_id, is_present, certainty, time = line.strip().split()

                cub_img_id = int(cub_img_id) - 1

                # here we map to random one
                random_cub_img_id = file_idx_to_random_file_idx[cub_img_id]
                random_idx_to_file_idx[random_cub_img_id] = cub_img_id

                att_id = int(att_id) - 1
                is_present = int(is_present)
                attribute_matrix[random_cub_img_id, att_id] = is_present

        return filename_to_cub_img_id, cub_img_id_to_filename, attribute_matrix, random_idx_to_file_idx

    def load_attribute_map(self, attribute_matrix_path=None):
        filename_to_cub_img_id = {}
        cub_img_id_to_filename = {}
        with open(pjoin(self.image_folder, "attributes", "images.txt")) as f:
            for line in f:
                cub_img_id, filename = line.strip().split()
                filename_to_cub_img_id[filename] = int(cub_img_id) - 1
                cub_img_id_to_filename[int(cub_img_id) - 1] = filename
                # so we can map to attribute_matrix

        # loading attributes, map it back to filename (our img-id)
        # <image_id> <attribute_id> <is_present> <certainty_id> <time>
        # 312 binary attributes per image;
        # 11788 images; 3677856 attributes.

        if attribute_matrix_path is None:
            attribute_matrix = np.zeros((len(filename_to_cub_img_id), 312))
            with open(pjoin(self.image_folder, "attributes", "image_attribute_labels.txt")) as f:
                for line in tqdm(f, total=3677856):
                    # some lines have error
                    # we fix it
                    if len(line.strip().split()) > 5:
                        tups = line.strip().split()
                        if tups[0] == '2275' or tups[0] == '9364':
                            # 2275 10 0 1 0  1.509
                            # img_id, att_id, is_present, certainty, 0, time
                            cub_img_id = int(tups[0]) - 1
                            att_id = int(tups[1]) - 1
                            is_present = int(tups[2])
                            attribute_matrix[cub_img_id, att_id] = is_present
                            continue
                            # 9364 10 0 3 0  3.535

                    cub_img_id, att_id, is_present, certainty, time = line.strip().split()

                    cub_img_id = int(cub_img_id) - 1
                    att_id = int(att_id) - 1
                    is_present = int(is_present)
                    attribute_matrix[cub_img_id, att_id] = is_present
        else:
            attribute_matrix = np.load(attribute_matrix_path)

        return filename_to_cub_img_id, cub_img_id_to_filename, attribute_matrix

    def get_attribute_vocab(self):
        # {att_id: att_name}
        ls = [""] * 312
        map = {}
        with open(pjoin(self.image_folder, "attributes", "attributes_labels.txt")) as f:
            for line in f:
                att_id, att_text = line.strip().split()
                # map integer
                ls[int(att_id) - 1] = att_text
                map[att_text] = int(att_id) - 1

        return ls, map

    def get_image_id_to_split(self):
        dic = {}
        for split in ['train', 'val', 'test']:
            with open(pjoin(self.image_folder, "{}.txt".format(split))) as f:
                for line in f:
                    dic[line.strip()] = split
        return dic

    def get_image(self, img_id):
        # img_id, again is a string like "030.Fish_Crow/Fish_Crow_0073_25977.jpg"
        split = self.image_id_to_split[img_id]
        return self.split_to_data[split].get_image(img_id)

    def get_train_val_test_loader(self, args):
        split_to_data = {}
        for split in ['train', 'val', 'test']:
            data_prep = DataPreparation(args.dataset, args.data_path)
            dataset, _ = data_prep.get_dataset_and_loader(split, args.pretrained_model,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.num_workers)
            split_to_data[split] = dataset

        return split_to_data

    def get_top_k_from_cell(self, cell, max_cap_per_cell, cell_select_strategy=None):
        if cell_select_strategy == None:
            return cell[:max_cap_per_cell]
        elif cell_select_strategy == 'random':
            random.shuffle(cell)
            return cell[:max_cap_per_cell]
        else:
            raise NotImplemented

    def clone(self, b):
        a = np.empty_like(b)
        a[:] = b
        return a

    def get_valid_segment_qs(self, img_id):
        attr_img_pos = self.filename_to_cub_img_id[img_id]
        if self.randomized:
            attr_img_pos = self.img_id_to_random_idx[attr_img_pos]

        valid_seg_ids = []
        labels = []
        for question_id in range(len(self.q_id_to_segments)):
            focus_attr_ids = self.segments_to_attr_id[self.q_id_to_segments[question_id]]
            attr_vec = self.attribute_matrix[attr_img_pos, focus_attr_ids]

            if sum(attr_vec) > 0:
                valid_seg_ids.append(question_id)
                labels.append(self.q_id_to_segments[question_id])

        return valid_seg_ids, labels

    def get_valid_qs(self, img_id, verbose=True):
        attr_img_pos = self.filename_to_cub_img_id[img_id]
        if self.randomized:
            attr_img_pos = self.img_id_to_random_idx[attr_img_pos]

        attrs = self.attribute_matrix[attr_img_pos]
        valid_attr_inds = attrs.nonzero()[0]  # indices

        labels = []
        for v_ind in valid_attr_inds:
            labels.append(self.attr_vocab_ls[v_ind])
            if verbose:
                print("qid: {}, attr: {}".format(v_ind, self.attr_vocab_ls[v_ind]))

        return valid_attr_inds, labels

    def map_img_pos_to_img_id(self, ls):
        return [self.cub_img_id_to_filename[cub_id] for cub_id in ls]

    def map_img_id_to_img_pos(self, ls):
        return [self.filename_to_cub_img_id[file_name] for file_name in ls]

    def set_label_usage(self, return_labels):
        if return_labels and not hasattr(self, 'class_labels'):
            self.load_class_labels(self.labels_path)
        self.return_labels = return_labels

    def load_class_labels(self, class_labels_path):
        with open(class_labels_path, 'rb') as f:
            label_dict = pickle.load(f, encoding='latin1')

        self.num_classes = len(set(label_dict.values()))
        self.class_labels = label_dict

    def get_class_label(self, img_id):
        class_label = torch.LongTensor([int(self.class_labels[img_id]) - 1])
        return class_label

    def get_batch(self, list_img_ids):
        # this needs to be called seperately for
        # this batches together distractor, similar, and target image
        # needs to be similar to collate_fn()

        # list_img_ids: file names
        images = []
        labels = []
        for img_id in list_img_ids:
            split = self.image_id_to_split[img_id]
            images.append(self.split_to_data[split].get_image(img_id))

            if self.return_labels:
                class_label = self.get_class_label(img_id)
                labels.append(class_label)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        labels = torch.cat(labels, 0)

        # we don't need ids, captions, labels, etc.
        return images, labels

    def get_caption_by_img_id(self, img_id, join_str=False):

        split = self.image_id_to_split[img_id]
        dataset = self.split_to_data[split]

        # base_id = dataset.ids[index]
        base_id = img_id

        img_anns = dataset.coco.imgToAnns[img_id]
        # rand_idx = np.random.randint(len(img_anns))
        ann_ids = [img_anns[rand_idx]['id'] for rand_idx in range(len(img_anns))]

        tokens = [dataset.tokens[ann_id] for ann_id in ann_ids]
        if join_str:
            tokens = [' '.join(t) for t in tokens]

        return tokens

    def get_captions_for_attribute(self, attr_ids, limit=5, print_ready=True):
        # attr_id: needs to be the "binary" attribute ID
        # retrieve images that have the attribute
        # attr_id: can be a number, OR a list of indices
        imgs = []

        for random_img_pos in range(self.attribute_matrix.shape[0]):
            if self.randomized:
                img_pos = self.random_idx_to_img_id[random_img_pos]
            else:
                img_pos = random_img_pos

            attr_vec = self.attribute_matrix[random_img_pos, attr_ids]
            if sum(attr_vec) >= 1:
                imgs.append(img_pos)

        img_cell = self.get_top_k_from_cell(imgs, max_cap_per_cell=limit)

        # get captions
        # map from img_pos to img_id (strings)
        img_ids = self.map_img_pos_to_img_id(img_cell)
        img_name_to_caption = {}

        for img_id in img_ids:
            captions = self.get_caption_by_img_id(img_id, True)
            img_name_to_caption[img_id] = captions

        # we print it out
        if print_ready:
            for img_id, captions in img_name_to_caption.items():
                print('img name: {}'.format(img_id))
                for i, c in enumerate(captions):
                    print(i, ":", c)

                print()

            return

        return img_name_to_caption

    def get_captions_for_segment(self, seg_id, limit=5, print_ready=True):
        focus_attr_ids = self.segments_to_attr_id[self.q_id_to_segments[seg_id]]

        imgs = []

        for random_img_pos in range(self.attribute_matrix.shape[0]):
            if self.randomized:
                img_pos = self.random_idx_to_img_id[random_img_pos]
            else:
                img_pos = random_img_pos

            attr_vec = self.attribute_matrix[random_img_pos, focus_attr_ids]
            if sum(attr_vec) >= 1:
                imgs.append(img_pos)

        img_cell = self.get_top_k_from_cell(imgs, max_cap_per_cell=limit)

        img_ids = self.map_img_pos_to_img_id(img_cell)
        img_name_to_caption = {}

        for img_id in img_ids:
            captions = self.get_caption_by_img_id(img_id, True)
            img_name_to_caption[img_id] = captions

        # we print it out
        if print_ready:
            for img_id, captions in img_name_to_caption.items():
                print('img name: {}'.format(img_id))
                for i, c in enumerate(captions):
                    print(i, ":", c)

                print()

            return

        return img_name_to_caption

def load_model(rsa_dataset, verbose=True):

    print("Loading Model ...")
    ml = ModelLoader(rsa_dataset.args, rsa_dataset.split_to_data['train'])
    model = getattr(ml, rsa_dataset.args.model)()
    if verbose:
        print(model, '\n')
        print("Loading Model Weights...")

    if torch.cuda.is_available():
        evaluation_state_dict = torch.load(rsa_dataset.args.eval_ckpt)
    else:
        evaluation_state_dict = torch.load(rsa_dataset.args.eval_ckpt, map_location='cpu')

    model_dict = model.state_dict(full_dict=True)
    model_dict.update(evaluation_state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    return model


class RSA(object):
    """
    RSA through matrix normalization
    Given a literal matrix of log-prob
        c1  c2  c3
    i   -5  -6  -20
    i'  -5  -9  -20
    i'' -10 -11 -20

    RSA has three cases:
    Case 1: If a sequence (C) has high prob for i, but high also in i', i'', the prob is relatively down-weighted
    Case 2: If a sequence (C) has low prob for i, but low also in i', i'', the prob is then relatively up-weighted (higher than original)
    Case 3: If a seuqnce (C) has high prob for i, but low for i', i'', the prob is relatively up-weighted
    (But this is hard to say due to the final row normalization)

    use logsumexp() to compute normalization constant

    Normalization/division in log-space is just a substraction

    Column normalization means: -5 - logsumexp([-5, -5, -10])
    (Add together a column)

    Row normalization means: -5 - logsumexp([-5, -6, -7])
    (Add together a row)

    We can compute RSA through the following steps:
    Step 1: add image prior: + log P(i) to the row
    Step 2: Column normalize
    - Pragmatic Listener L1: L1(i|c) \propto S0(c|i) P(i)
    Step 3: Multiply the full matrix by rationality parameter (0, infty), when rationality=1, no changes (similar to temperature)
    Step 4: add speaker prior: + log P(c_t|i, c_<t) (basically add the original literal matrix) (very easy)
            OR add a unconditioned speaker prior: + log P(c) (through a language model, like KenLM)
    Step 5: Row normalization
    - Pragmatic Speaker S1: S1(c|i) \propto L1(i|c) p(c), where p(c) can be S0

    The reason for additions is e^{\alpha log L1(i|c) + log p(i)}, where \alpha is rationality parameter
    """

    def __init__(self):
        # can be used to add KenLM language model
        # The "gigaword" one takes too long to load
        pass

    def build_literal_matrix(self, orig_logprob, distractor_logprob):
        """
        :param orig_logprob: [n_sample]
        :param distractor_logprob: [num_distractors, n_sample]
        :return: We put orig_logprob as the FIRST row
                [num_distractors+1 , n_sample]
        """
        return torch.cat([orig_logprob.unsqueeze(0), distractor_logprob], dim=0)

    def compute_pragmatic_speaker(self, literal_matrix,
                                  rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                  return_diagnostics=False):
        """
        Do the normalization over logprob matrix

        literal_matrix: [num_distractor_images+1, captions]
        So row normalization correspond to

        :param literal_matrix: should be [I, C]  (num_images, num_captions)
                               Or [I, Vocab] (num_images, vocab_size)
        :param speaker_prior: turn on, we default to adding literal matrix
        :param speaker_prior_lm_mat: [I, Vocab] (a grammar weighting for previous tokens)

        :return:
               A re-weighted matrix [I, C/Vocab]
        """
        # step 1
        pass
        # step 2
        s0 = literal_matrix.clone()
        norm_const = logsumexp(literal_matrix, dim=0, keepdim=True)
        l1 = literal_matrix.clone() - norm_const
        # step 3
        l1 *= rationality
        # step 4
        if speaker_prior:
            # we add speaker prior
            # this needs to be a LM with shared vocabulary
            if lm_logprobsf is not None:
                s1 = l1 + lm_logprobsf[0]
            else:
                s1 = l1 + s0
        # step 5
        norm_const = logsumexp(s1, dim=1, keepdim=True)  # row normalization
        s1 = s1 - norm_const

        if return_diagnostics:
            return s1, l1, s0

        return s1

    def compute_entropy(self, prob_mat, dim, keepdim=True):
        return -torch.sum(prob_mat * torch.exp(prob_mat), dim=dim, keepdim=keepdim)

    def compute_pragmatic_speaker_w_similarity(self, literal_matrix, num_similar_images,
                                               rationality=1.0, speaker_prior=False, lm_logprobsf=None,
                                               entropy_penalty_alpha=0.0, return_diagnostics=False):
        # step 1
        pass
        # step 2
        s0_mat = literal_matrix
        prior = s0_mat.clone()[0]

        l1_mat = s0_mat - logsumexp(s0_mat, dim=0, keepdim=True)

        # step 3
        # pragmatic_listener_matrix *= rationality
        # step 5: QuD-RSA S1
        # 0). Compute entropy H[P(v|i, q(i)=q(i'))]; normalize "vertically" on

        same_cell_prob_mat = l1_mat[:num_similar_images + 1] - logsumexp(l1_mat[:num_similar_images + 1], dim=0)
        l1_qud_mat = same_cell_prob_mat.clone()
        #         same_cell_norm
        entropy = self.compute_entropy(same_cell_prob_mat, 0, keepdim=True)  # (1, |V|)

        # this means when alpha=0.2, it's more similar to what we have before: -1/H
        utility_2 = entropy

        # then we need to normalize this to [0, 1] and then take log!
        # so that entropy and prob will be in the same space

        # 1). Sum over similar images with target image (vertically)
        # [target_image, [similar_images], [distractor_images]]
        utility_1 = logsumexp(l1_mat[:num_similar_images + 1], dim=0, keepdim=True)  # [1, |V|]
        # This tradeoff may or may not be the best way...we are adding log-probability with entropy

        # now we do 1/H(x), this is more similar to "cost"
        # and we do L1(u) - C(u) style!
        utility = (1 - entropy_penalty_alpha) * utility_1 + entropy_penalty_alpha * utility_2

        s1 = utility * rationality

        # set_trace()

        # apply rationality
        if speaker_prior:
            if lm_logprobsf is None:
                s1 += prior
            else:
                s1 += lm_logprobsf[0]  # lm rows are all the same  # here is two rows summation

        if return_diagnostics:
            # We return RSA-terms only; on the oustide (Debugger), we re-assemble for snapshots of computational process
            # s0, L1, u1, L1*, u2, u1+u2, s1
            # mat, vec, vec, mat, vec, vec, vec
            return s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, utility, s1 - logsumexp(s1, dim=1,
                                                                                                      keepdim=True)

        return s1 - logsumexp(s1, dim=1, keepdim=True)


class IncRSA(RSA):
    def __init__(self, model, rsa_dataset, lm_model=None):
        super().__init__()
        self.model = model
        self.rsa_dataset = rsa_dataset

        args = self.rsa_dataset.args

        trainer_creator = getattr(TrainerLoader, args.model)
        evaluator = trainer_creator(args, model, rsa_dataset.split_to_data['val'], [],
                                    None, rsa_dataset.device)
        evaluator.train = False

        self.evaluator = evaluator
        self.device = self.evaluator.device

    def sentence_decode(self, sampled_ids):
        outputs = sampled_ids
        vocab = self.evaluator.dataset.vocab

        generated_captions = []
        for out_idx in range(len(outputs)):
            sentence = []
            for w in outputs[out_idx]:
                word = vocab.get_word_from_idx(w.data.item())
                if word != vocab.end_token:
                    sentence.append(word)
                else:
                    break
            generated_captions.append(' '.join(sentence))

        return generated_captions

    def semantic_speaker(self, image_id_list, decode_strategy="greedy"):
        # image_id here is a string!
        image_input, labels = self.rsa_dataset.get_batch(image_id_list)
        if decode_strategy == 'greedy':
            image_input = image_input.to(self.device)
            sample_ids = self.model.generate_sentence(image_input, self.evaluator.start_word,
                                                      self.evaluator.end_word, labels, labels_onehot=None,
                                                      max_sampling_length=50, sample=False)
        else:
            raise Exception("not implemented")

        if len(sample_ids.shape) == 1:
            sample_ids = sample_ids.unsqueeze(0)

        return self.sentence_decode(sample_ids)

    def greedy_pragmatic_speaker(self, img_id, question_id, rationality,
                                 speaker_prior, entropy_penalty_alpha,
                                 max_cap_per_cell=5, cell_select_strategy=None,
                                 no_similar=False, verbose=True, return_diagnostic=False, segment=False,
                                 subset_similarity=False):
        # collect_stats: debug mode (or eval mode); collect RSA statistics to understand internal workings

        if max_cap_per_cell == 0:
            return self.semantic_speaker([img_id], decode_strategy="greedy")

        dis_cell, sim_cell, quality = self.rsa_dataset.get_cells_by_partition(img_id, question_id, max_cap_per_cell,
                                                                              cell_select_strategy,
                                                                              no_similar=no_similar,
                                                                              segment=segment,
                                                                              subset_similarity=subset_similarity)

        image_id_list = [img_id] + sim_cell + dis_cell
        with torch.no_grad():
            if not return_diagnostic:
                captions = self.greedy_pragmatic_speaker_free(image_id_list, len(sim_cell),
                                                              rationality, speaker_prior, entropy_penalty_alpha)
            else:
                captions, diags = self.greedy_pragmatic_speaker_free(image_id_list, len(sim_cell),
                                                                     rationality, speaker_prior, entropy_penalty_alpha,
                                                                     return_diagnostic=True)

        if return_diagnostic:
            return captions[0], quality, diags

        return captions[0], quality

    def fill_list(self, items, lists):
        # this is a pass-by-reference update
        for item, ls in zip(items, lists):
            ls.append(item)

    def greedy_pragmatic_speaker_free(self, image_id_list, num_sim, rationality,
                                      speaker_prior, entropy_penalty_alpha, lm_logprobsf=None,
                                      max_sampling_length=50, sample=False, return_diagnostic=False):
        """
        We always assume image_id_list[0] is the target image
        image_id_list[:num_sim] are the within cell
        image_id_list[num_sim:] are the distractors

        Will only return 1 caption for the target image
        :param image_id_list:
        :param num_sim:
        :param num_distractor:
        :param max_sampling_length:
        :return:
        """

        image_input, labels = self.rsa_dataset.get_batch(image_id_list)
        image_inputs = image_input.to(self.device)

        start_word = self.evaluator.start_word
        end_word = self.evaluator.end_word

        feat_func = self.model.get_labels_append_func(labels, None)
        image_features = image_inputs

        image_features = self.model.linear1(image_features)
        image_features = F.relu(image_features)
        image_features = feat_func(image_features)
        image_features = image_features.unsqueeze(1)  # (11, 1, 1200)

        embedded_word = self.model.word_embed(start_word)
        embedded_word = embedded_word.expand(image_features.size(0), -1, -1)

        init_states = (None, None)
        lstm1_states, lstm2_states = init_states

        end_word = end_word.squeeze().expand(image_features.size(0))
        reached_end = torch.zeros_like(end_word.data).byte()

        sampled_ids = []

        if return_diagnostic:
            # their length is the time step length
            s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list = [], [], [], [], [], [], [], []

        # greedy loop, over time step
        i = 0
        while not reached_end.all() and i < max_sampling_length:
            lstm1_input = embedded_word

            # LSTM 1
            lstm1_output, lstm1_states = self.model.lstm1(lstm1_input, lstm1_states)

            lstm1_output = torch.cat((image_features, lstm1_output), 2)

            # LSTM 2
            lstm2_output, lstm2_states = self.model.lstm2(lstm1_output, lstm2_states)

            outputs = self.model.linear2(lstm2_output.squeeze(1))
            # outputs: torch.Size([11, 3012])

            # all our RSA computation is in log-prob space
            log_probs = F.log_softmax(outputs, dim=-1)  # log(softmax(x))

            # rsa!!
            literal_matrix = log_probs

            # diagnostics!!

            if not return_diagnostic:
                pragmatic_array = self.compute_pragmatic_speaker_w_similarity(literal_matrix, num_sim,
                                                                              rationality=rationality,
                                                                              speaker_prior=speaker_prior,
                                                                              entropy_penalty_alpha=entropy_penalty_alpha,
                                                                              lm_logprobsf=lm_logprobsf)
            else:
                s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, combined_u, pragmatic_array = self.compute_pragmatic_speaker_w_similarity(
                    literal_matrix, num_sim,
                    rationality=rationality,
                    speaker_prior=speaker_prior,
                    entropy_penalty_alpha=entropy_penalty_alpha,
                    lm_logprobsf=lm_logprobsf,
                    return_diagnostics=True)
                self.fill_list([s0_mat, l1_mat, utility_1, l1_qud_mat, entropy, utility_2, combined_u, pragmatic_array],
                               [s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list])

            # pragmatic_array:
            # torch.Size([1, 3012])

            # pragmatic array becomes the computational output
            # but we need to repeat it for all
            # beam search / diverse beam search this part is easier to handle.
            outputs = pragmatic_array.expand(len(image_id_list), -1)  # expand along batch dimension
            # rsa augmentation ends

            if sample:
                predicted, log_p = self.sample(outputs)
                active_batches = (~reached_end)
                log_p *= active_batches.float().to(log_p.device)
                # log_probabilities.append(log_p.unsqueeze(1))
                # lengths += active_batches.long()
            else:
                predicted = outputs.max(1)[1]

            reached_end = reached_end | predicted.eq(end_word).data
            sampled_ids.append(predicted.unsqueeze(1))
            embedded_word = self.model.word_embed(predicted)
            embedded_word = embedded_word.unsqueeze(1)

            i += 1

        sampled_ids = torch.cat(sampled_ids, 1).squeeze()

        if return_diagnostic:
            return self.sentence_decode(sampled_ids), [s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list,
                                                       u_list, s1_list]

        return self.sentence_decode(sampled_ids)

    def sample(self, logits):
        dist = Categorical(logits=logits)
        sample = dist.sample()
        return sample, dist.log_prob(sample)

    def diverse_beam_search(self):
        pass

    def nucleus_sampling(self):
        pass


"""
Essentially this is almost exactly the same as IncRSA
Except we add some new method, such as trace, visualize etc.

This is a stateful solution, but makes interaction easy enough.
"""


class IncRSADebugger(IncRSA):

    def greedy_pragmatic_speaker(self, img_id, question_id, rationality,
                                 speaker_prior, entropy_penalty_alpha,
                                 max_cap_per_cell=5, cell_select_strategy=None,
                                 no_similar=False, verbose=True, return_diagnostic=True,
                                 segment=True, subset_similarity=True):

        # we will automatically store last sentence

        return_diagnostic = True
        if max_cap_per_cell == 0:
            sent = super().greedy_pragmatic_speaker(img_id, question_id, rationality,
                                                    speaker_prior, entropy_penalty_alpha, max_cap_per_cell,
                                                    cell_select_strategy,
                                                    no_similar, verbose, return_diagnostic)
            return sent
        else:
            sent, quality, diags = super().greedy_pragmatic_speaker(img_id, question_id, rationality,
                                                                    speaker_prior, entropy_penalty_alpha,
                                                                    max_cap_per_cell, cell_select_strategy,
                                                                    no_similar, verbose, return_diagnostic,
                                                                    segment, subset_similarity)

        self.sent = sent
        self.quality = quality
        self.diags = diags
        self.question_id = question_id
        self.rationality = rationality

        return sent, quality

    def run_full_checks(self):
        # s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list
        self.check_s0_row_stochastic(self.diags[0])
        self.check_l1_column_stochastic(self.diags[1])
        self.check_u1_sum_of_partial_prob(self.diags[2])
        self.check_l1_qud_column_normalized(self.diags[3])
        self.check_u2_entropy_correct(self.diags[4], self.diags[3])
        self.check_s1_row_normalized(self.diags[-1])

    def check_s0_row_stochastic(self, s0_mat):
        rand_time_idx = np.random.randint(len(s0_mat))
        print("S0 - The following value should be 1:", torch.exp(logsumexp(s0_mat[rand_time_idx][0])))

    def check_l1_column_stochastic(self, l1_mat):
        rand_time_idx = np.random.randint(len(l1_mat))

        print("L1 - The following value should be 1:", torch.exp(logsumexp(l1_mat[rand_time_idx][:, 0])))

    def check_u1_sum_of_partial_prob(self, u1_vec):
        # if the summed partial prob should be smaller than 1
        rand_time_idx = np.random.randint(len(u1_vec))
        rand_v_idx = np.random.randint(u1_vec[rand_time_idx].shape[1])

        for v_idx in range(u1_vec[rand_time_idx].shape[1]):
            assert torch.exp(u1_vec[rand_time_idx][0, rand_v_idx]) < 1

        print("U1 - The following value should be less than 1:", torch.exp(u1_vec[rand_time_idx][0, rand_v_idx]))

    def check_l1_qud_column_normalized(self, l1_qud_mat):
        rand_time_idx = np.random.randint(len(l1_qud_mat))
        print("L1 QuD - The following value should be 1:", torch.exp(logsumexp(l1_qud_mat[rand_time_idx][:, 0])))

    def check_u2_entropy_correct(self, u2_vec, l1_qud_mat):
        # use Scipy to compute entropy, check if it's the same
        rand_time_idx = np.random.randint(len(u2_vec))
        rand_v_idx = np.random.randint(l1_qud_mat[rand_time_idx].shape[1])

        v_prob = torch.exp(l1_qud_mat[rand_time_idx][:, rand_v_idx]).cpu().numpy()
        h = entropy(v_prob)

        print("U2 - The following two values should equal {} == {}".format(
            h, u2_vec[rand_time_idx][0, rand_v_idx]
        ))

    def check_s1_row_normalized(self, s1_mat):
        rand_time_idx = np.random.randint(len(s1_mat))
        print("S0 - The following value should be 1:", torch.exp(logsumexp(s1_mat[rand_time_idx][0])))

    def compute_rank(self, a):
        temp = a.argsort()[::-1]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(a))

        # rank: each position corresponds to the original item (rank of each item, same order as original list)
        # temp: [smallest to largest] (order of original list)
        return ranks, temp

    def compute_stats(self, torch_ls, focus_idx):
        # return ranking/medium/mean/min/max
        a = torch_ls.squeeze().cpu().numpy()
        ranks, _ = self.compute_rank(a)

        return ranks[focus_idx], [np.mean(a), np.median(a), np.min(a), np.max(a), np.std(a)]

    def get_index_from_word(self, word):
        vocab = self.evaluator.dataset.vocab
        return vocab(word)

    def get_word_from_index(self, idx):
        vocab = self.evaluator.dataset.vocab
        return vocab.get_word_from_idx(idx)

    def stats_to_str(self, stats):
        concat_str = ""
        for tup in zip("mean/median/min/max/std".split('/'), stats):
            concat_str += "{}: {:.3f} ".format(tup[0], tup[1])

        return concat_str

    def numeric_space(self, matrix, prob_space):
        if prob_space:
            return torch.exp(matrix)
        else:
            return matrix

    def get_ranked_item_index(self, diag_list_idx, timestep, rank_indices, is_item_word=True):
        """
        Used to examine "U1 word "spotted" has value -0.256 ranked 3th/3012"
        What are the words higher than "spotted"? Do they have lower entropy? We need to trace those!

        s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_lis
        :param diag_list_idx: put in 0 to len(self.diags)
        :param rank_indices: the top-k items that we want (i.e., [0,1,2,3]) or [2, 5])
        :param is_item_word: if we are looking for a word, we return word; otherwise we return index
        :return: [(word/index, value)]
        """
        # we return index
        item_list = self.diags[diag_list_idx]
        item_unit = item_list[timestep]
        if diag_list_idx in [0, 1, -1]:
            values = item_unit[0, :].squeeze().cpu().numpy()
        else:
            values = item_unit.squeeze().cpu().numpy()

        ranks, temp = self.compute_rank(values)
        top_k = temp[rank_indices]

        if is_item_word:
            return [self.get_word_from_index(k) for k in top_k], values[top_k]
        else:
            return top_k, values[top_k]

    def compute_rsa_decision_path_for_word(self, timestep, focus_word=None, verbose=True, prob_space=False,
                                           return_rank=False):
        """
        Conclusion: Ranking, and global stats, with negative words should satisfy ALL our debugging needs.
        We can quantify "failure" cases using the stats we collected here

        Basic requirements:
        1. In normal RSA:
            1). For S0, we print the relative ranking of word of choice in row, medium/mean/min/max/CI of the list
            2). For our focus word, in L1, we display word prob in target, and word probs in distractor, display it's ranking (in this limited group), medium/mean/min/max of the list
                -- p(i|u); if our "u" cannot even signal "i", then semantic model S0 failure!
                -- if word prob that we want is not the highest in target, then our S0 failed already (this is a "relative" measure)
                -- Assert: target w needs to be 1st
            3). For our focus word, in pre-prior S1, we compute it's row-stats again (to see if it's increased)
                -- p(u|i); without prior
                -- If target w is not the first, we know how big the gap is between w and w* (because we return max)
            4). For our focus word, in post-prior S1, we compute it's row-stats again (to see if it's increased)
                -- p(i|u)p(u|i)
                -- If target w is not the first, we know how big the gap is between w and w* (because we return max)
                -- between (3) and (4), we see the effect of prior
        2. In QuD-Entropy RSA
            1). Same as 1.1) S0
                -- word ranking; global stats
            2). Same as 1.2) L1
                -- word ranking; global stats
            3). U1 -- QuD RSA thing (summed from L1).
                -- word ranking; global stats
            4). U2/L1* -- Entropy thing (computed from L1*). Relative ranking of the word, stats of the dist.
                -- word ranking; global stats
                -- for negative word, we'll just track them seperately
            5). same as 1.3) pre-prior S1
                -- might need a magnitude comparison with prior
            6). same as 1.4) post-prior S1

        :arg timestep: it can be any word in the vocab list (doesn't have to be present in caption)
        :arg focus_word: we can investigate why this "focus_word" did not appear at timestep t!
        :arg negative_words: a list of negative words we send in, words that SHOULD NOT appear in the current caption

        :return a packaged stats for future decision making (like check what's broken)
        """
        if focus_word is None:
            focus_word = self.sent.split()[timestep]

        focus_word_idx = self.get_index_from_word(focus_word)
        # 1). S0 (row)
        s0_list, l1_list, u1_list, l1_qud_list, entropy_list, u2_list, u_list, s1_list = self.diags
        s0_mat = self.numeric_space(s0_list[timestep], prob_space)

        s0_rank, s0_stats = self.compute_stats(s0_mat[0, :], focus_word_idx)
        if verbose:
            print('S0 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, s0_mat[0, focus_word_idx],
                                                                               s0_rank, len(s0_mat[0, :]),
                                                                               self.stats_to_str(s0_stats)))

        # 2). L1 (column)
        l1_mat = self.numeric_space(l1_list[timestep], prob_space)
        l1_rank, l1_stats = self.compute_stats(l1_mat[:, focus_word_idx],
                                               0)  # here the focus_index is the "target" image p(i|u)
        if verbose:
            print('L1 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, l1_mat[0, focus_word_idx],
                                                                               l1_rank, len(l1_mat[:, focus_word_idx]),
                                                                               self.stats_to_str(l1_stats)))

        # 3). U1 (would be the same as L1 in normal RSA setting)
        u1_vec = self.numeric_space(u1_list[timestep].squeeze(), prob_space)
        u1_rank, u1_stats = self.compute_stats(u1_vec, focus_word_idx)
        if verbose:
            print('U1 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, u1_vec[focus_word_idx],
                                                                               u1_rank, len(u1_vec),
                                                                               self.stats_to_str(u1_stats)))

        # 4). Entropy
        ent_vec = self.numeric_space(entropy_list[timestep].squeeze(), False)
        ent_rank, ent_stats = self.compute_stats(ent_vec, focus_word_idx)
        if verbose:
            print('Entropy word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word,
                                                                                    ent_vec[focus_word_idx],
                                                                                    ent_rank, len(ent_vec),
                                                                                    self.stats_to_str(ent_stats)))

        # 5). U2
        u2_vec = self.numeric_space(u2_list[timestep].squeeze(), False)
        u2_rank, u2_stats = self.compute_stats(u2_vec, focus_word_idx)
        if verbose:
            print('U2 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, u2_vec[focus_word_idx],
                                                                               u2_rank, len(u2_vec),
                                                                               self.stats_to_str(u2_stats)))

        # 5). Alpha impact (this is pre-prior S1)
        # multiply with rationality (temperature); higher rationality means
        # higher rationality means lower influence of this
        u_vec = self.numeric_space(u_list[timestep].squeeze(), False)
        u_rank, u_stats = self.compute_stats(u_vec, focus_word_idx)
        if verbose:
            print('f(U1, U2) word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, u_vec[focus_word_idx],
                                                                                      u_rank, len(u_vec),
                                                                                      self.stats_to_str(u_stats)))

        # 6). Post-prior S1
        s1_vec = self.numeric_space(s1_list[timestep].squeeze(), prob_space)
        s1_rank, s1_stats = self.compute_stats(s1_vec, focus_word_idx)
        if verbose:
            print('S1 word "{}" has value {} ranked {}th/{}, stats: {}'.format(focus_word, s1_vec[focus_word_idx],
                                                                               s1_rank, len(s1_vec),
                                                                               self.stats_to_str(s1_stats)))

        if return_rank:
            # evolution of ranks through RSA computation
            # this is the "decision path"
            # S0, L1, QuD-L1 (U1), f(U1, U2), S1 ~= alpha * f(U1, U2) + prior
            # the ranks in here are all out of vocab rank

            # L1 is still the rank over vocab space
            l1_utt_rank, _ = self.compute_stats(l1_mat[0, :], focus_word_idx)

            # in terms of raw value, we can have all of them... but for now, we can skip
            return [s0_rank, l1_utt_rank, u1_rank, u2_rank, u_rank, s1_rank], []

    def set_plt_style(self):
        params = {'backend': 'pdf',
                  'axes.titlesize': 10,
                  'axes.labelsize': 10,
                  'font.size': 10,
                  'legend.fontsize': 10,
                  'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'font.family': 'DejaVu Serif',
                  'font.serif': 'Computer Modern',
                  }
        matplotlib.rcParams.update(params)

    def visualize_words_decision_paths_at_timestep(self, timestep, words, cmap_name=None):
        # build a line-graph of ranks
        # each line is a word at the timestep

        self.set_plt_style()

        try:
            self.diags
        except:
            print("Need to run 'greedy_pragmatic_speaker' over an image first!")
            return

        word_rank = {}
        word_final_rank = {}
        for w in words:
            r, _ = self.compute_rsa_decision_path_for_word(timestep, w, verbose=False, return_rank=True)
            word_rank[w] = -np.array(r)  # we use "negative" rank
            word_final_rank[w] = r[-1]

        # gradient = np.linspace(0, 1, 256)
        # gradient = np.vstack((gradient, gradient))

        # https://stackoverflow.com/questions/9750699/how-to-display-only-a-left-and-bottom-box-border-in-matplotlib

        plt.figure(figsize=(12, 8))
        # plt.set_cmap("RdBu")
        ax = plt.gca()
        # ax.set_prop_cycle('color', [plt.cm.get_cmap('RdBu')(i) for i in np.linspace(0, 1, len(word_rank))])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.xticks(range(6), ['S0', 'L1', 'U1', 'U2', 'f(U1, U2)', 'S1'])
        # colors = [plt.cm.get_cmap("RdBu")(i) for i in np.linspace(0, 1, len(word_rank))]
        if cmap_name is None:
            colors = [plt.cm.get_cmap("coolwarm")(i) for i in np.linspace(0, 1, len(word_rank))]
        else:
            colors = [plt.cm.get_cmap(cmap_name)(i) for i in np.linspace(0, 1, len(word_rank))]

        for i, (w, r) in enumerate(word_rank.items()):
            plt.plot(range(6), r, marker='o', markersize=3, linewidth=1, color=colors[i], label='"{}"'.format(w))
            # ax.annotate('"{}"'.format(w), xy=(list(range(6))[-1], r[-1]), xytext=(10, 0), textcoords='offset points', va='center')

        # post y-axis modification
        start, end = ax.get_ylim()
        plt.yticks(plt.yticks()[0], [str(int(n)) + "th" for n in -plt.yticks()[0]])
        ax.set_ylim(bottom=start, top=end)

        plt.title("Timestep {}".format(timestep))
        plt.legend()

        plt.show()

        return word_final_rank

    def visualize_word_decision_path_at_timesteps(self, word, cmap_name=None):
        # each line is a time step
        # we scan through the time steps
        timesteps = range(len(self.sent.split()))

        self.set_plt_style()

        try:
            self.diags
        except:
            print("Need to run 'greedy_pragmatic_speaker' over an image first!")
            return

        timestep_rank = {}
        timestep_final_rank = {}
        for t in timesteps:
            r, _ = self.compute_rsa_decision_path_for_word(t, word, verbose=False, return_rank=True)
            timestep_rank[t] = -np.array(r)
            timestep_final_rank[t] = r[-1]

        plt.figure(figsize=(12, 8))
        # plt.set_cmap("RdBu")
        ax = plt.gca()
        # ax.set_prop_cycle('color', [plt.cm.get_cmap('RdBu')(i) for i in np.linspace(0, 1, len(word_rank))])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.xticks(range(6), ['S0', 'L1', 'U1', 'U2', 'f(U1, U2)', 'S1'])
        # colors = [plt.cm.get_cmap("RdBu")(i) for i in np.linspace(0, 1, len(word_rank))]
        if cmap_name is None:
            colors = [plt.cm.get_cmap("coolwarm")(i) for i in np.linspace(0, 1, len(timestep_rank))]
        else:
            colors = [plt.cm.get_cmap(cmap_name)(i) for i in np.linspace(0, 1, len(timestep_rank))]

        for i, (t, r) in enumerate(timestep_rank.items()):
            plt.plot(range(6), r, marker='o', markersize=3, linewidth=1, color=colors[i], label='step {}'.format(t))
            # ax.annotate('At step {}'.format(t), xy=(list(range(6))[-1], r[-1]), xytext=(10, 0), textcoords='offset points',
            #             va='center')

        # post y-axis modification
        start, end = ax.get_ylim()
        plt.yticks(plt.yticks()[0], [str(int(n)) + "th" for n in -plt.yticks()[0]])
        ax.set_ylim(bottom=start, top=end)

        plt.title('Word "{}"'.format(word))
        plt.legend()

        plt.show()

        return timestep_final_rank


if __name__ == '__main__':
    rsa_dataset = BirdDistractorDataset()
    # dataset, data_loader = rsa_dataset.split_to_data['train']
    # dataset[3]
    # image, target, base_id =
    # (tensor([ 0.0202, -0.0080, -0.0014,  ...,  0.0296, -0.0029, -0.0227]), tensor([ 1., 31.,  7., 27.,  4., 77., 28., 29., 60., 38., 74., 12.,  4.,  5.,
    #         68.,  2.]), '195.Carolina_Wren/Carolina_Wren_0029_186212.jpg')
    # dataset.return_label
    # 10946

    # save a randomized attribute map
    _, _, att_map, random_idx_to_file_idx = rsa_dataset.load_attribute_map_randomized()
    # import pdb; pdb.set_trace()
    pickle.dump(random_idx_to_file_idx, open("./data/cub/attributes/random_idx_to_file_idx.json", 'wb'))
    np.save("./data/cub/attributes/randomized_attribute_matrix.npy", att_map)

    # rsa_dataset.get_valid_qs("195.Carolina_Wren/Carolina_Wren_0029_186212.jpg")
    # rsa_dataset.get_cells_by_partition("195.Carolina_Wren/Carolina_Wren_0029_186212.jpg", 117, 5)
    # rsa_dataset.filename_to_cub_img_id["195.Carolina_Wren/Carolina_Wren_0029_186212.jpg"]
    # rsa_dataset.attribute_matrix[11470, 117]
    # rsa_dataset.attribute_matrix[0, 117]
    # rsa_dataset.attribute_matrix[[0, 2, 3], 117]

    # rsa_dataset.get_cells_by_partition("195.Carolina_Wren/Carolina_Wren_0029_186212.jpg", 117, 5, "random")

    import IPython

    IPython.embed()

    # pdb.set_trace()
