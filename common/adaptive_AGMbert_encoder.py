import os
import time
import torch
import torch.optim as optim
from sklearn import metrics
import datetime
import numpy as np
from common.utils import load_bert_sentences, multi_acc, multi_mse, generate_over_tokenizer
from models.get_optim import get_Adam_optim_v2,get_Adam_optim_v3
from cfgs import constants
from cfgs.constants import ensureDirs
from models.bert_adaptive_encoder import *
from models import policy_network
from transformers import BertTokenizer, BertConfig

class adaSTrainer(object):
    def __init__(self, config):
        self.config = config
        self.train_itr, self.dev_itr, self.test_itr = load_bert_sentences(config)

        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = BertForSequenceClassificationadptive.from_pretrained(pretrained_weights, num_labels=config.num_labels)
        model.bert.encoder.copyParameters()
        bertconfig = BertConfig.from_pretrained(pretrained_weights)
        bert_embdding = model.bert.embeddings

        for name, param in model.named_parameters():
            if 'encoder.layer' in name:
                param.requires_grad = False


        agent = policy_network.Transformer(bertconfig)
        agent.init_embedding_layer(bert_embdding)

        #if self.config.n_gpu > 1:
        #    self.net = torch.nn.DataParallel(model).to(config.device)
        #    self.agent = torch.nn.DataParallel(agent).to(config.device)
        #else:
        USE_CUDA = torch.cuda.is_available()
        config.device = torch.device("cuda:3" if USE_CUDA else "cpu")

        self.net = model.to(config.device)
        self.agent = agent.to(config.device)


        self.optim, self.scheduler = get_Adam_optim_v3(config, self.net)
        self.agent_optimizer = optim.Adam(agent.parameters(), lr=config.TRAIN.lr_agent)

        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.best_dev_f1 =0
        self.unimproved_iters = 0
        self.iters_not_improved = 0

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
            # self.train_one_epoch()
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
        if (os.path.exists(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')):
            os.remove(self.config.log_path + '/log_run_' + self.config.dataset + '_' + version + '.txt')
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

    def train(self):
        # Save log information
        logfile = open(
            self.config.log_path +
            '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
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
            self.agent.train()

            train_loss, train_acc, train_rmse = self.train_epoch()

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         logs)

            self.net.eval()
            self.agent.eval()
            eval_loss, eval_acc, eval_f1 = self.eval(self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_f1, eval="evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         eval_logs)

            eval1_loss, eval1_acc, eval1_f1 = self.eval(self.dev_itr)
            eval1_logs = self.get_logging(eval1_loss, eval1_acc, eval1_f1, eval="evaluating_val")
            print("\r" + eval1_logs)

            # logging evaluating logs
            self.logging(self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                         eval1_logs)
            # early stopping
            if self.config.is_multilabel:
                if eval_f1 > self.best_dev_f1:
                    self.unimproved_iters = 0
                    self.best_dev_f1 = eval_f1
                    # saving models
                    ensureDirs(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    self.tokenizer.save_pretrained(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    #if self.config.n_gpu > 1:
                    #    self.net.module.save_pretrained(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    #else:
                    self.net.save_pretrained(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    torch.save(self.agent, os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset)+'/agent.pkl')

                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt' + "\n" + \
                                              "Early Stopping. Epoch: {}, Best Dev f1: {}".format(epoch, self.best_dev_f1)
                        print(early_stop_logs)
                        self.logging(
                            self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
                            early_stop_logs)
                        break
            else:
                if eval_acc > self.best_dev_acc:
                    self.unimproved_iters = 0
                    self.best_dev_acc = eval_acc
                    # saving models
                    ensureDirs(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    self.tokenizer.save_pretrained(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    # if self.config.n_gpu > 1:
                    #    self.net.module.save_pretrained(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    # else:
                    self.net.save_pretrained(os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset))
                    torch.save(self.agent, os.path.join(constants.SAVED_MODEL_PATH, self.config.dataset) + '/agent.pkl')

                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt' + "\n" + \
                                              "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch,
                                                                                                   self.best_dev_acc)
                        print(early_stop_logs)
                        self.logging(
                            self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt',
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
            text, label = batch

            input_ids, attention_mask = generate_over_tokenizer(text, self.tokenizer, max_length=80)
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            # print(label)
            # labels = [float(x) for x in label]
            # print(label)


            labels = label.long().to(self.config.device)

            pro = self.agent(input_ids)
            action = gumbel_softmax(pro.view(pro.size(0), -1, 2))
            policy = action[:, :, 1]
            logits = self.net(input_ids=input_ids, policy=policy,
                              attention_mask=attention_mask)[0]






            # print(logits)
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

            if self.config.TRAIN.gradient_accumulation_steps > 1:
                loss = loss / self.config.TRAIN.gradient_accumulation_steps

            if (step + 1) % self.config.TRAIN.gradient_accumulation_steps == 0:
                self.optim.zero_grad()
                self.agent_optimizer.zero_grad()
                loss.backward()
                self.optim.step()
                self.agent_optimizer.step()
                self.scheduler.step()


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
            text, label = batch
            input_ids, attention_mask = generate_over_tokenizer(text, self.tokenizer, max_length=80)
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            labels = label.long().to(self.config.device)

            with torch.no_grad():
                pro = self.agent(input_ids)
                action = gumbel_softmax(pro.view(pro.size(0), -1, 2))
                policy = action[:, :, 1]

                logits = self.net(input_ids=input_ids,
                                  policy=policy,
                                  attention_mask=attention_mask)[0]

            if self.config.is_multilabel:
                loss = torch.binary_cross_entropy_with_logits(logits, labels.float()).mean()

                pre = torch.sigmoid(logits).round().long().cpu().detach().numpy()
                predicted_labels.extend(pre)
                target_labels.extend(labels.cpu().detach().numpy())


                accuracy = metrics.accuracy_score(labels.cpu().numpy(), pre)

                total_loss.append(loss.data.cpu().numpy())
                total_acc.append(accuracy)
            else:

                loss = loss_fn(logits,  labels)
            # loss = loss_fn(logits, labels)
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
                    return np.array(total_loss).mean(), np.array(total_acc).mean(), f1
        else:
                    return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())
