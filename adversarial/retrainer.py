from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from model import *
from .utils import *
from .loggers import *
from .dataset import *
from .dataloader import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd.gradcheck import zero_gradients
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
import math
import faiss
import numpy as np
from abc import *
from pathlib import Path


class PoisonedGroupRetrainer(metaclass=ABCMeta):
    def __init__(self, args, wb_model_spec, wb_model, bb_model, original_test_loader, bb_model_root=None):
        self.args = args
        self.device = args.device
        self.num_items = args.num_items
        self.max_len = args.bert_max_len
        self.wb_model_spec = wb_model_spec
        self.wb_model = wb_model.to(self.device)
        self.bb_model = bb_model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.bb_model = nn.DataParallel(self.bb_model)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.original_test_loader = original_test_loader
        if bb_model_root == None:
            self.bb_model_root = 'experiments/' + args.model_code + '/' + args.dataset_code
        else:
            self.bb_model_root = bb_model_root
        
        if isinstance(self.wb_model, BERT):
            self.item_embeddings = self.wb_model.embedding.token.weight.detach().cpu().numpy()[1:-1]
        else:
            self.item_embeddings = self.wb_model.embedding.token.weight.detach().cpu().numpy()[1:]
        
        self.faiss_index = faiss.IndexFlatL2(self.item_embeddings.shape[-1])
        self.faiss_index.add(self.item_embeddings)
        self.item_embeddings = torch.tensor(self.item_embeddings).to(self.device)

        self.CLOZE_MASK_TOKEN = args.num_items + 1
        self.adv_ce = nn.CrossEntropyLoss(ignore_index=0)
        if isinstance(self.bb_model, BERT) or isinstance(self.bb_model, NARM):
            self.ce = nn.CrossEntropyLoss(ignore_index=0)
        elif isinstance(self.bb_model, SASRec):
            self.ce = nn.BCEWithLogitsLoss()


    def train_ours(self, targets, ratio, popular_items, num_items):
        num_poisoned, num_original, poisoning_users = self.generate_poisoned_data(targets, popular_items, num_items)
        target_spec = '_'.join([str(target) for target in targets])
        self.train_loader, self.val_loader, self.test_loader = poi_train_loader_factory(self.args, target_spec, self.wb_model_spec, num_poisoned, num_original)
        self.bb_model.load_state_dict(torch.load(os.path.join(self.bb_model_root, 'models', 'best_acc_model.pth'), map_location='cpu').get(STATE_DICT_KEY))
        self.export_root = 'experiments/retrained/' + self.wb_model_spec + '/' + self.args.dataset_code + '/ratio_' + str(ratio) + '_target_' + target_spec
        self.writer, self.train_loggers, self.val_loggers = self._create_loggers()
        self.logger_service = LoggerService(
            self.train_loggers, self.val_loggers)
        self.log_period_as_iter = self.args.log_period_as_iter
        metrics_before, metrics_after = self.train(targets)
        
        return metrics_before, metrics_after


    def generate_poisoned_data(self, targets, popular_items, num_items, batch_size=50, sample_prob=0.0):
        print('## Generate Biased Data with Target {} ##'.format(targets))
        target_spec = '_'.join([str(target) for target in targets])
        dataset = poi_dataset_factory(self.args, target_spec, self.wb_model_spec)
        # if dataset.check_data_present():
        #     print('Dataset already exists. Skip generation')
        #     return

        if isinstance(self.wb_model, BERT):
            self.item_embeddings = self.wb_model.embedding.token.weight.detach().cpu().numpy()[1:-1]
        else:
            self.item_embeddings = self.wb_model.embedding.token.weight.detach().cpu().numpy()[1:]
        self.item_embeddings = torch.tensor(self.item_embeddings).to(self.device)
        
        batch_num = math.ceil(self.args.num_poisoned_seqs / batch_size)
        print('Generating poisoned dataset...')
        for i in tqdm(range(batch_num)):
            if i == batch_num - 1 and self.args.num_poisoned_seqs % batch_size != 0:
                batch_size = self.args.num_poisoned_seqs % batch_size
            seqs = torch.tensor(np.random.choice(targets, size=batch_size)).reshape(batch_size, 1).to(self.device)

            for j in range(self.max_len - 1):
                self.wb_model.eval()
                
                if j % 2 == 0:
                    selected_targets = torch.tensor(np.random.choice(targets, size=batch_size)).to(self.device)
                    rand_items = torch.tensor(np.random.choice(self.num_items, size=seqs.size(0))+1).to(self.device)
                    seqs = torch.cat((seqs, rand_items.unsqueeze(1)), 1)

                    if isinstance(self.wb_model, BERT):
                        mask_items = torch.tensor([self.CLOZE_MASK_TOKEN] * seqs.size(0)).to(self.device)
                        input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                        if j < self.max_len - 2:
                            input_seqs[:, (self.max_len-3-j):-1] = seqs
                        elif j == self.max_len - 2:
                            input_seqs[:, :-1] = seqs[:, 1:]
                        input_seqs[:, -1] = mask_items
                        wb_embedding, mask = self.wb_model.embedding(input_seqs.long())
                    elif isinstance(self.wb_model, SASRec):
                        input_seqs = torch.zeros((seqs.size(0), self.max_len)).to(self.device)
                        input_seqs[:, (self.max_len-2-j):] = seqs
                        wb_embedding, mask = self.wb_model.embedding(input_seqs.long())
                    elif isinstance(self.wb_model, NARM):
                        input_seqs = seqs
                        lengths = torch.tensor([j + 2] * seqs.size(0))
                        wb_embedding, mask = self.wb_model.embedding(input_seqs, lengths)

                    self.wb_model.train()
                    wb_embedding = wb_embedding.detach().clone()
                    wb_embedding.requires_grad = True
                    zero_gradients(wb_embedding)

                    if isinstance(self.wb_model, BERT) or isinstance(self.wb_model, SASRec):
                        wb_scores = self.wb_model.model(wb_embedding, self.wb_model.embedding.token.weight, mask)[:, -1, :]
                    elif isinstance(self.wb_model, NARM):
                        wb_scores = self.wb_model.model(wb_embedding, self.wb_model.embedding.token.weight, lengths, mask)

                    loss = self.adv_ce(wb_scores, selected_targets)
                    self.wb_model.zero_grad()
                    loss.backward()
                    wb_embedding_grad = wb_embedding.grad.data
                    
                    self.wb_model.eval()
                    with torch.no_grad():
                        if isinstance(self.wb_model, BERT):
                            current_embedding = wb_embedding[:, -2]
                            current_embedding_grad = wb_embedding_grad[:, -2]
                        else:
                            current_embedding = wb_embedding[:, -1]
                            current_embedding_grad = wb_embedding_grad[:, -1]
                        
                        all_embeddings = self.item_embeddings.unsqueeze(1).repeat_interleave(current_embedding.size(0), 1)
                        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                        multipication_results = torch.t(cos(current_embedding-current_embedding_grad.sign(), all_embeddings))
                        multipication_results[torch.arange(seqs.size(0)), selected_targets-1] = multipication_results[torch.arange(seqs.size(0)), selected_targets-1] + 2
                        
                        _, candidate_indicies = torch.sort(multipication_results, dim=1, descending=False)
                        sample_indices = torch.randint(0, 10, [seqs.size(0)])
                        seqs[:, -1] = candidate_indicies[torch.arange(seqs.size(0)), sample_indices] + 1
                        seqs = torch.cat((seqs, selected_targets.unsqueeze(1)), 1)
            
            seqs = seqs[:, :self.max_len]
            try:
                batch_tokens = np.concatenate((batch_tokens, seqs.cpu().numpy()))
            except:
                batch_tokens = seqs.cpu().numpy()

        num_poisoned, num_original, poisoning_users = dataset.save_dataset(batch_tokens.tolist(), original_dataset_size=self.args.num_original_seqs)
        return num_poisoned, num_original, poisoning_users

    def train(self, targets):
        self.optimizer = self._create_optimizer()
        if self.args.enable_lr_schedule:
            if self.args.enable_lr_warmup:
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, self.args.warmup_steps, len(train_loader) * self.num_epochs)
            else:
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.args.decay_step, gamma=self.args.gamma)

        print('## Biased Retrain on Item {} ##'.format(targets))
        accum_iter = 0
        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)        
        
        print('## Clean Black-Box Model Targeted Test on Item {} ##'.format(targets))
        metrics_before = self.targeted_test(targets, load_retrained=False)
        print('## Retrained Black-Box Model Targeted Test on Item {} ##'.format(targets))
        metrics_after = self.targeted_test(targets, load_retrained=True)
        
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

        return metrics_before, metrics_after

    def train_one_epoch(self, epoch, accum_iter):
        self.bb_model.train()
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.optimizer.zero_grad()
            if isinstance(self.bb_model, BERT):
                seqs, labels = batch
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                logits = self.bb_model(seqs)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = self.ce(logits, labels)
            elif isinstance(self.bb_model, SASRec):
                seqs, labels, negs = batch
                seqs, labels, negs = seqs.to(self.device), labels.to(self.device), negs.to(self.device)
                logits = self.bb_model(seqs)  # F.softmax(self.bb_model(seqs), dim=-1)
                pos_logits = logits.gather(-1, labels.unsqueeze(-1))[seqs > 0].squeeze()
                pos_targets = torch.ones_like(pos_logits)
                neg_logits = logits.gather(-1, negs.unsqueeze(-1))[seqs > 0].squeeze()
                neg_targets = torch.zeros_like(neg_logits)
                loss = self.ce(torch.cat((pos_logits, neg_logits), 0), torch.cat((pos_targets, neg_targets), 0))
            elif isinstance(self.bb_model, NARM):
                seqs, lengths, labels = batch
                lengths = lengths.flatten()
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                logits = self.bb_model(seqs, lengths)
                loss = self.ce(logits, labels.squeeze())

            loss.backward()
            self.clip_gradients(5)
            self.optimizer.step()
            if self.args.enable_lr_schedule:
                self.lr_scheduler.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += seqs.size(0)

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch + 1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)
        
        self.validate(epoch, accum_iter)
        return accum_iter

    def validate(self, epoch, accum_iter):
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            # self.log_extra_val_info(log_data)
            self.logger_service.log_val(log_data)

    def test(self, load_retrained=False):
        if load_retrained:
            best_model_dict = torch.load(os.path.join(
                self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
            self.bb_model.load_state_dict(best_model_dict)
        else:
            bb_model_dict = torch.load(os.path.join(
                self.bb_model_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
            self.bb_model.load_state_dict(bb_model_dict)

        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.original_test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                metrics = self.calculate_metrics(batch)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            return average_metrics

    def targeted_test(self, targets, load_retrained=False):
        if load_retrained:
            best_model_dict = torch.load(os.path.join(
                self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
            self.bb_model.load_state_dict(best_model_dict)
        else:
            bb_model_dict = torch.load(os.path.join(
                self.bb_model_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
            self.bb_model.load_state_dict(bb_model_dict)
        
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.original_test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if isinstance(self.bb_model, BERT) or isinstance(self.bb_model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    scores = self.bb_model(seqs)[:, -1, :]
                elif isinstance(self.bb_model, NARM):
                    seqs, lengths, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    lengths = lengths.flatten()
                    scores = self.bb_model(seqs, lengths)
                
                for target in targets:
                    candidates[:, 0] = torch.tensor([target] * seqs.size(0)).to(self.device)
                    metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
                    self._update_meter_set(average_meter_set, metrics)
                
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
        return average_metrics
    
    def targeted_test_item(self, targets, load_retrained=False):
        if load_retrained:
            best_model_dict = torch.load(os.path.join(
                self.export_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
            self.bb_model.load_state_dict(best_model_dict)
        else:
            bb_model_dict = torch.load(os.path.join(
                self.bb_model_root, 'models', 'best_acc_model.pth')).get(STATE_DICT_KEY)
            self.bb_model.load_state_dict(bb_model_dict)
        
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        item_average_meter_set = {target: AverageMeterSet() for target in targets}
        
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.original_test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if isinstance(self.bb_model, BERT) or isinstance(self.bb_model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    scores = self.bb_model(seqs)[:, -1, :]
                elif isinstance(self.bb_model, NARM):
                    seqs, lengths, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    lengths = lengths.flatten()
                    scores = self.bb_model(seqs, lengths)
                
                for target in targets:
                    candidates[:, 0] = torch.tensor([target] * seqs.size(0)).to(self.device)
                    metrics = recalls_and_ndcgs_for_ks(scores.gather(1, candidates), labels, self.metric_ks)
                    self._update_meter_set(average_meter_set, metrics)
                    self._update_meter_set(item_average_meter_set[target], metrics)
                    
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            for target in targets:
                item_average_meter_set[target] = item_average_meter_set[target].averages()
        return average_metrics, item_average_meter_set

    def calculate_metrics(self, batch):
        self.bb_model.eval()

        if isinstance(self.bb_model, BERT) or isinstance(self.bb_model, SASRec):
            seqs, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            scores = self.bb_model(seqs)[:, -1, :]
        elif isinstance(self.bb_model, NARM):
            seqs, lengths, candidates, labels = batch
            seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
            lengths = lengths.flatten()
            scores = self.bb_model(seqs, lengths)

        scores = scores.gather(1, candidates)  # B x C
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def clip_gradients(self, limit=5):
        for p in self.bb_model.parameters():
            nn.utils.clip_grad_norm_(p, 5)

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]
                               ] + ['Recall@%d' % k for k in self.metric_ks[:3]]
        description = 'Eval: ' + \
            ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(
            *(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)

    def _create_optimizer(self):
        args = self.args
        param_optimizer = list(self.bb_model.named_parameters())
        no_decay = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if args.optimizer.lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        elif args.optimizer.lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        # based on hugging face get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='epoch',
                               graph_name='Epoch', group_name='Train'),
            MetricGraphPrinter(writer, key='loss',
                               graph_name='Loss', group_name='Train'),
        ]

        val_loggers = []
        for k in self.metric_ks:
            val_loggers.append(
                MetricGraphPrinter(writer, key='NDCG@%d' % k, graph_name='NDCG@%d' % k, group_name='Validation'))
            val_loggers.append(
                MetricGraphPrinter(writer, key='Recall@%d' % k, graph_name='Recall@%d' % k, group_name='Validation'))
        val_loggers.append(RecentModelLogger(model_checkpoint))
        val_loggers.append(BestModelLogger(
            model_checkpoint, metric_key=self.best_metric))
        return writer, train_loggers, val_loggers

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.bb_model.module.state_dict() if self.is_parallel else self.bb_model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
