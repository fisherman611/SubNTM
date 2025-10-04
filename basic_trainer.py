import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy
from time import time


class BasicTrainer:
    def __init__(self, model, epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, device="cuda"):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device

        self.logger = logging.getLogger('main')

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            # lr_scheduler = StepLR(
            #     optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)

        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            from collections import defaultdict
            loss_rst_dict = defaultdict(float)

            for batch in dataset_handler.train_dataloader:
                x = batch['data'].to(self.device, dtype=torch.float32, non_blocking=True)

                rows = batch['sub_rows'].to(self.device, non_blocking=True)
                cols = batch['sub_cols'].to(self.device, non_blocking=True)
                vals = batch['sub_vals'].to(self.device, dtype=torch.float32, non_blocking=True)
                M, V = batch['sub_shape']
                row2doc = batch['row2doc'].to(self.device, non_blocking=True)  # (M,)
                B = batch['B']

                if M > 0:
                    idx = torch.stack([rows, cols], dim=0)
                    X_m = torch.sparse_coo_tensor(idx, vals, (M, V), device=self.device).coalesce()
                    x_sub = (X_m, row2doc, B)
                else:
                    x_sub = None  # no subdocs in this batch → doc-only path

                rst_dict = self.model(x, x_sub)
                loss = rst_dict['loss']
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Accumulate logs
                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * len(batch['data'])
                    except Exception:
                        loss_rst_dict[key] += rst_dict[key] * len(batch)

            if self.lr_scheduler:
                lr_scheduler.step()

            if verbose and epoch % self.log_interval == 0:
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'
                print(output_log)
                self.logger.info(output_log)

    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = []

        # make sure model is on the right device
        self.model.eval()
        self.model.to(self.device)

        # ensure float32 (doc encoder expects float)
        if input_data.dtype != torch.float32:
            input_data = input_data.float()

        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            for idx in all_idx:
                # move this batch to the same device as the model
                batch_input = input_data[idx].to(self.device)
                batch_theta = self.model.get_theta(batch_input)  # (B, T)
                theta.extend(batch_theta.cpu().tolist())

        return np.asarray(theta)

    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        train_theta = self.test(dataset_handler.train_data)
        time1 = time()
        test_theta = self.test(dataset_handler.test_data)
        time2 = time()
        print("Time inference: " + str(time2 - time1))
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        if hasattr(self.model, 'group_embeddings'):
            group_embeddings = self.model.group_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
            self.logger.info(
                f'group_embeddings size: {group_embeddings.shape}')

            group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
            np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

        return word_embeddings, topic_embeddings