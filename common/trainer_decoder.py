import os
import time
import torch
import datetime
import numpy as np
from common.utils import multi_acc, multi_mse, load_data_decoder
from models.get_optim import get_Adam_optim
from cfgs import constants
from sklearn import metrics
from cfgs.constants import DATASET_MAP, DATASET_PATH_MAP



class DTrainer(object):
    def __init__(self, config):
        self.config = config
        self.train_itr, self.dev_itr, self.test_itr = load_data_decoder(config, load_pretrained=True)
        model = constants.DECODER[config.gencoder](config)

        # if self.config.n_gpu > 1:
        #    self.net = torch.nn.DataParallel(model).to(config.device)
        # else:
        config.device = 'cuda:3'
        self.config.device = 'cuda:3'

        self.net = model.to(config.device)
        self.optim = get_Adam_optim(config, self.net)
        dataset = DATASET_MAP[config.dataset]()
        config.is_multilabel = dataset.IS_MULTILABEL


        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.best_dev_f1 = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
        elif run_mode == 'val':
            eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, "evaluating")
            print("\r" + eval_logs)
        elif run_mode == 'test':
            eval_loss, eval_acc, eval_rmse = self.eval(self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, "evaluating")
            print("\r" + eval_logs)
        else:
            exit(-1)

    def empty_log(self, version):
        if (os.path.exists(self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + version + '.txt')):
            os.remove(self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + version + '.txt')
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, loss, acc, rmse, eval='training'):
        if self.config.is_multilabel:
            logs = \
                '==={:10} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
                '\t'.join(["{:<6}"] * 3).format("loss", "acc", "f1") + '\n' + \
                '\t'.join(["{:^6.3f}"] * 3).format(loss, acc, rmse) + '\n'

        else:
            logs = \
                '==={:10} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
                '\t'.join(["{:<6}"] * 3).format("loss", "acc", "rmse") + '\n' + \
                '\t'.join(["{:^6.3f}"] * 3).format(loss, acc, rmse) + '\n'

        return logs

    def f(self,x):
        all = []
        for item in x:
            tmp = list(map(lambda x: int(x), item))
            all.append(tmp)
        return torch.tensor(all, dtype=torch.float)

    def train(self):
        # Save log information
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt',
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.close()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.net.train()
            train_loss, train_acc, train_rmse = self.train_epoch()

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(
                self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt',
                logs)
            self.net.eval()
            eval_loss, eval_acc, eval_f1 = self.eval(self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_f1, eval="evaluating")
            print("\r" + eval_logs)

            # logging testing logs
            self.logging(
                self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt',
                eval_logs)

            eval1_loss, eval1_acc, eval1_f1 = self.eval(self.dev_itr)
            eval1_logs = self.get_logging(eval1_loss, eval1_acc, eval1_f1, eval="evaluating_val")
            print("\r" + eval1_logs)

            # logging evaluating logs
            self.logging(
                self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt',
                eval1_logs)

            # early stopping
            if self.config.is_multilabel:
                if eval_f1 > self.best_dev_f1:
                    self.unimproved_iters = 0
                    self.best_dev_f1 = eval_f1
                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt' + "\n" + \
                                          "Early Stopping. Epoch: {}, Best Dev f1: {}".format(epoch, self.best_dev_f1)
                        print(early_stop_logs)
                        self.logging(
                            self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt',
                            early_stop_logs)
                        break
            else:
                if eval_acc > self.best_dev_acc:
                    self.unimproved_iters = 0
                    self.best_dev_acc = eval_acc
                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt' + "\n" + \
                                          "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                        print(early_stop_logs)
                        self.logging(
                            self.config.log_path + '/log_run_' + self.config.dataset + '_decoder_' + self.config.version + '.txt',
                            early_stop_logs)
                        break

    def train_epoch(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        predicted_labels, target_labels = list(), list()

        for step, batch in enumerate(self.train_itr):
            start_time = time.time()
            embeddings, labels, lengths = batch
            mask = []
            for l in lengths:
                mask.append((torch.arange(max(lengths)) < l).long())
            mask = torch.stack(mask, 0).to(self.config.device)
            if self.config.is_multilabel:
                labels = self.f(labels)

            labels = labels.long().to(self.config.device)

            if self.config.gencoder in ['e1', 'e3']:
                logits = self.net(embeddings.to(self.config.device), mask.to(self.config.device))
            else:
                logits = self.net(embeddings.to(self.config.device), mask.to(self.config.device))
            if self.config.is_multilabel:
                loss = torch.binary_cross_entropy_with_logits(logits, labels.float()).mean()
                pre = torch.sigmoid(logits).round().long().cpu().detach().numpy()

                predicted_labels.extend(pre)
                target_labels.extend(labels.cpu().detach().numpy())

                accuracy = metrics.accuracy_score(labels.cpu().numpy(), pre)
                total_loss.append(loss.data.cpu().numpy())
                total_acc.append(accuracy)
            else:
                loss = loss_fn(logits, labels)
                metric_acc = acc_fn(labels, logits)
                metric_mse = mse_fn(labels, logits)

                total_loss.append(loss.data.cpu().numpy())
                total_acc.append(metric_acc.data.cpu().numpy())
                total_mse.append(metric_mse.data.cpu().numpy())

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(self.train_itr)) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)  Loss: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(self.train_itr)),
                    100 * (step) / int(len(self.train_itr)),
                    loss,
                    int(h), int(m), int(s)),
                end="")
        f1 = metrics.f1_score(np.array(target_labels), np.array(predicted_labels), average='micro')
        if self.config.is_multilabel:
            return np.array(total_loss).mean(), np.array(total_acc).mean(), f1
        else:
            return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())

    def eval(self, eval_itr):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        predicted_labels, target_labels = list(), list()

        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            embeddings, labels, lengths = batch
            mask = []
            for l in lengths:
                mask.append((torch.arange(max(lengths)) < l).long())
            mask = torch.stack(mask, 0).to(self.config.device)
            if self.config.is_multilabel:
                labels = self.f(labels)
            labels = labels.long().to(self.config.device)

            if self.config.gencoder in ['e1', 'e2']:
                logits = self.net(embeddings.to(self.config.device), mask.to(self.config.device))
            else:
                logits = self.net(embeddings.to(self.config.device), mask.to(self.config.device))

            if self.config.is_multilabel:
                loss = torch.binary_cross_entropy_with_logits(logits, labels.float()).mean()

                pre = torch.sigmoid(logits).round().long().cpu().detach().numpy()
                predicted_labels.extend(pre)
                target_labels.extend(labels.cpu().detach().numpy())

                accuracy = metrics.accuracy_score(labels.cpu().numpy(), pre)

                total_loss.append(loss.data.cpu().numpy())
                total_acc.append(accuracy)
            else:

                loss = loss_fn(logits, labels)
                metric_acc = acc_fn(labels, logits)
                metric_mse = mse_fn(labels, logits)
                total_loss.append(loss.data.cpu().numpy())
                total_acc.append(metric_acc.data.cpu().numpy())
                total_mse.append(metric_mse.data.cpu().numpy())

            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(eval_itr)) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%)  Loss: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr)),
                    100 * (step) / int(len(eval_itr)),
                    loss,
                    int(h), int(m), int(s)),
                end="")
        f1 = metrics.f1_score(np.array(target_labels), np.array(predicted_labels), average='micro')

        if self.config.is_multilabel:
            return np.array(total_loss).mean(), np.array(total_acc).mean(), f1,
        else:
            return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())
