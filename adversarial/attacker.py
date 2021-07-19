from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY
from model import *
from .utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from tqdm import tqdm

import json
import faiss
import numpy as np
from abc import *
from pathlib import Path


class AdversarialRankAttacker(metaclass=ABCMeta):
    def __init__(self, args, wb_model, bb_model, test_loader):
        self.args = args
        self.device = args.device
        self.num_items = args.num_items
        self.max_len = args.bert_max_len
        self.wb_model = wb_model.to(self.device)
        self.bb_model = bb_model.to(self.device)

        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.test_loader = test_loader
        self.CLOZE_MASK_TOKEN = args.num_items + 1
        self.adv_ce = nn.CrossEntropyLoss(ignore_index=0)

        if isinstance(self.wb_model, BERT):
            self.item_embeddings = self.wb_model.embedding.token.weight.detach().cpu().numpy()[1:-1]
        else:
            self.item_embeddings = self.wb_model.embedding.token.weight.detach().cpu().numpy()[1:]
        
        self.faiss_index = faiss.IndexFlatL2(self.item_embeddings.shape[-1])
        self.faiss_index.add(self.item_embeddings)
        self.item_embeddings = torch.tensor(self.item_embeddings).to(self.device)

        if isinstance(self.bb_model, BERT):
            self.bb_item_embeddings = self.bb_model.embedding.token.weight.detach().cpu().numpy()[1:-1]
        else:
            self.bb_item_embeddings = self.bb_model.embedding.token.weight.detach().cpu().numpy()[1:]
        self.bb_item_embeddings = torch.tensor(self.bb_item_embeddings).to(self.device)


    def attack(self, target, num_attack=10, repeated_search=10):
        print('## Targeted Attack on Item {} ##'.format(str(target)))
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.test_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            self.wb_model.eval()
            with torch.no_grad():
                if isinstance(self.bb_model, BERT) or isinstance(self.bb_model, SASRec):
                    seqs, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    if isinstance(self.bb_model, BERT):
                        seqs[:, 1:] = seqs[:, :-1]
                        seqs[:, 0] = 0
                elif isinstance(self.bb_model, NARM):
                    seqs, lengths, candidates, labels = batch
                    seqs, candidates, labels = seqs.to(self.device), candidates.to(self.device), labels.to(self.device)
                    seqs = self.post2pre_padding(seqs)
                
                perturbed_seqs = seqs.clone()
                append_items = torch.tensor([target]*(perturbed_seqs.size(0)*num_attack)).reshape(-1, num_attack)
                perturbed_seqs = torch.cat((perturbed_seqs, torch.tensor(append_items).to(self.device)), 1)

                perturbed_seqs = perturbed_seqs[:, -self.max_len:]
                if isinstance(self.wb_model, BERT):
                    mask_items = torch.tensor([self.CLOZE_MASK_TOKEN] * perturbed_seqs.size(0)).to(self.device)
                    perturbed_seqs[:, :-1] = perturbed_seqs[:, 1:]
                    perturbed_seqs[:, -1] = mask_items
                    wb_embedding, mask = self.wb_model.embedding(perturbed_seqs.long())
                elif isinstance(self.wb_model, SASRec):
                    wb_embedding, mask = self.wb_model.embedding(perturbed_seqs.long())
                elif isinstance(self.wb_model, NARM):
                    perturbed_seqs = self.pre2post_padding(perturbed_seqs)
                    lengths = (perturbed_seqs > 0).sum(-1).cpu().flatten()
                    wb_embedding, mask = self.wb_model.embedding(perturbed_seqs.long(), lengths)

            self.wb_model.train()
            wb_embedding = wb_embedding.detach().clone()
            wb_embedding.requires_grad = True
            zero_gradients(wb_embedding)

            if isinstance(self.wb_model, BERT) or isinstance(self.wb_model, SASRec):
                wb_scores = self.wb_model.model(wb_embedding, self.wb_model.embedding.token.weight, mask)[:, -1, :]
            elif isinstance(self.wb_model, NARM):
                wb_scores = self.wb_model.model(wb_embedding, self.wb_model.embedding.token.weight, lengths, mask)

            loss = self.adv_ce(wb_scores, torch.tensor([target] * perturbed_seqs.size(0)).to(self.device))
            self.wb_model.zero_grad()
            loss.backward()
            wb_embedding_grad = wb_embedding.grad.data

            self.wb_model.eval()
            with torch.no_grad():
                appended_indicies = (perturbed_seqs != self.CLOZE_MASK_TOKEN)
                appended_indicies = (perturbed_seqs != 0) * appended_indicies
                appended_indicies = torch.arange(perturbed_seqs.shape[1]).to(self.device) * appended_indicies
                _, appended_indicies = torch.sort(appended_indicies, -1, descending=True)
                appended_indicies = appended_indicies[:, :num_attack]
                
                best_seqs = perturbed_seqs.clone().detach()
                for num in range(num_attack):
                    row_indices = torch.arange(seqs.size(0))
                    col_indices = appended_indicies[:, num]

                    current_embedding = wb_embedding[row_indices, col_indices]
                    current_embedding_grad = wb_embedding_grad[row_indices, col_indices]
                    all_embeddings = self.item_embeddings.unsqueeze(1).repeat_interleave(current_embedding.size(0), 1)
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    multipication_results = torch.t(cos(current_embedding-current_embedding_grad.sign(), all_embeddings))
                    _, candidate_indicies = torch.sort(multipication_results, dim=1, descending=True)

                    if num == 0:
                        multipication_results[:, target-1] = multipication_results[:, target-1] - 100000000
                        _, candidate_indicies = torch.sort(multipication_results, dim=1, descending=True)
                        best_seqs[row_indices, col_indices] = candidate_indicies[:, 0] + 1

                        if isinstance(self.wb_model, BERT) or isinstance(self.wb_model, SASRec):
                            logits = F.softmax(self.wb_model(best_seqs)[:, -1, :], dim=-1)
                        elif isinstance(self.wb_model, NARM):
                            logits = F.softmax(self.wb_model(best_seqs, lengths), dim=-1)
                        best_scores = torch.gather(logits, -1, torch.tensor([target] * best_seqs.size(0)).unsqueeze(1).to(self.device)).squeeze()

                    elif num > 0:
                        prev_col_indices = appended_indicies[:, num-1]
                        if_prev_target = (best_seqs[row_indices, prev_col_indices] == target)
                        multipication_results[:, target-1] = multipication_results[:, target-1] + (if_prev_target * -100000000)
                        _, candidate_indicies = torch.sort(multipication_results, dim=1, descending=True)
                        best_seqs[row_indices, col_indices] = best_seqs[row_indices, col_indices] * ~if_prev_target + \
                            (candidate_indicies[:, 0] + 1) * if_prev_target
                        
                        if isinstance(self.wb_model, BERT) or isinstance(self.wb_model, SASRec):
                            logits = F.softmax(self.wb_model(best_seqs)[:, -1, :], dim=-1)
                        elif isinstance(self.wb_model, NARM):
                            logits = F.softmax(self.wb_model(best_seqs, lengths), dim=-1)
                        best_scores = torch.gather(logits, -1, torch.tensor([target] * best_seqs.size(0)).unsqueeze(1).to(self.device)).squeeze()

                    for time in range(repeated_search):
                        temp_seqs = best_seqs.clone().detach()
                        temp_seqs[row_indices, col_indices] = candidate_indicies[:, time] + 1

                        if isinstance(self.wb_model, BERT) or isinstance(self.wb_model, SASRec):
                            logits = F.softmax(self.wb_model(temp_seqs)[:, -1, :], dim=-1)
                        elif isinstance(self.wb_model, NARM):
                            logits = F.softmax(self.wb_model(temp_seqs, lengths), dim=-1)
                        temp_scores = torch.gather(logits, -1, torch.tensor([target] * temp_seqs.size(0)).unsqueeze(1).to(self.device)).squeeze()

                        best_seqs[row_indices, col_indices] = temp_seqs[row_indices, col_indices] * (temp_scores >= best_scores) + best_seqs[row_indices, col_indices] * (temp_scores < best_scores)
                        best_scores = temp_scores * (temp_scores >= best_scores) + best_scores * (temp_scores < best_scores)
                        best_seqs = best_seqs.detach()
                        best_scores = best_scores.detach()
                        del temp_scores
            
            perturbed_seqs = best_seqs.detach()
            if isinstance(self.wb_model, BERT) and isinstance(self.bb_model, BERT):
                perturbed_scores = self.bb_model(perturbed_seqs)[:, -1, :]
            elif isinstance(self.wb_model, BERT) and isinstance(self.bb_model, SASRec):
                temp_seqs = torch.cat((torch.zeros(perturbed_seqs.size(0)).long().unsqueeze(1).to(self.device), perturbed_seqs[:, :-1]), dim=1)
                perturbed_scores = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.wb_model, BERT) and isinstance(self.bb_model, NARM):
                temp_seqs = torch.cat((torch.zeros(perturbed_seqs.size(0)).long().unsqueeze(1).to(self.device), perturbed_seqs[:, :-1]), dim=1)
                temp_seqs = self.pre2post_padding(temp_seqs)
                temp_lengths = (temp_seqs > 0).sum(-1).cpu().flatten()
                perturbed_scores = self.bb_model(temp_seqs, temp_lengths)
            elif isinstance(self.wb_model, SASRec) and isinstance(self.bb_model, SASRec):
                perturbed_scores = self.bb_model(perturbed_seqs)[:, -1, :]
            elif isinstance(self.wb_model, SASRec) and isinstance(self.bb_model, BERT):
                temp_seqs = torch.cat((perturbed_seqs[:, 1:], torch.tensor([self.CLOZE_MASK_TOKEN] * perturbed_seqs.size(0)).unsqueeze(1).to(self.device)), dim=1)
                perturbed_scores = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.wb_model, SASRec) and isinstance(self.bb_model, NARM):
                temp_seqs = self.pre2post_padding(perturbed_seqs)
                temp_lengths = (temp_seqs > 0).sum(-1).cpu().flatten()
                perturbed_scores = self.bb_model(temp_seqs, temp_lengths)
            elif isinstance(self.wb_model, NARM) and isinstance(self.bb_model, NARM):
                perturbed_scores = self.bb_model(perturbed_seqs, lengths)
            elif isinstance(self.wb_model, NARM) and isinstance(self.bb_model, BERT):
                temp_seqs = self.post2pre_padding(perturbed_seqs)
                temp_seqs = torch.cat((temp_seqs[:, 1:], torch.tensor([self.CLOZE_MASK_TOKEN] * perturbed_seqs.size(0)).unsqueeze(1).to(self.device)), dim=1)
                perturbed_scores = self.bb_model(temp_seqs)[:, -1, :]
            elif isinstance(self.wb_model, NARM) and isinstance(self.bb_model, SASRec):
                temp_seqs = self.post2pre_padding(perturbed_seqs)
                perturbed_scores = self.bb_model(temp_seqs)[:, -1, :]
            
            candidates[:, 0] = torch.tensor([target] * candidates.size(0)).to(self.device)    
            perturbed_scores = perturbed_scores.gather(1, candidates)
            metrics = recalls_and_ndcgs_for_ks(perturbed_scores, labels, self.metric_ks)
            self._update_meter_set(average_meter_set, metrics)
            self._update_dataloader_metrics(tqdm_dataloader, average_meter_set)

        average_metrics = average_meter_set.averages()
        return average_metrics
    

    def test(self, target=None):
        if target is not None:
            print('## Black-Box Targeted Test on Item {} ##'.format(str(target)))
        else:
            print('## Black-Box Untargeted Test on Item Level ##')
        
        self.bb_model.eval()
        average_meter_set = AverageMeterSet()
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
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
                
                if target is not None:
                    candidates[:, 0] = torch.tensor([target] * seqs.size(0)).to(self.device)
                scores = scores.gather(1, candidates)
                metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
                self._update_meter_set(average_meter_set, metrics)
                self._update_dataloader_metrics(
                    tqdm_dataloader, average_meter_set)

            average_metrics = average_meter_set.averages()
            return average_metrics

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

    def pre2post_padding(self, seqs):
        processed = torch.zeros_like(seqs)
        lengths = (seqs > 0).sum(-1).squeeze()
        for i in range(seqs.size(0)):
            processed[i, :lengths[i]] = seqs[i, seqs.size(1)-lengths[i]:]
        return processed

    def post2pre_padding(self, seqs):
        processed = torch.zeros_like(seqs)
        lengths = (seqs > 0).sum(-1).squeeze()
        for i in range(seqs.size(0)):
            processed[i, seqs.size(1)-lengths[i]:] = seqs[i, :lengths[i]]
        return processed

    def _update_meter_set(self, meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(self, tqdm_dataloader, meter_set):
        description_metrics = ['Recall@%d' % k for k in self.metric_ks[:3]] + ['NDCG@%d' % k for k in self.metric_ks[1:3]]
        description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
        description = description.replace('NDCG', 'N').replace('Recall', 'R')
        description = description.format(*(meter_set[k].avg for k in description_metrics))
        tqdm_dataloader.set_description(description)
