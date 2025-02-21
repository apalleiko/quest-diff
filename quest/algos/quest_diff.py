import copy
import torch
import torch.nn.functional as F
import numpy as np
import quest.utils.tensor_utils as TensorUtils
import itertools

from quest.algos.base import ChunkPolicy, DoubleChunkPolicy


class QueST_diff(DoubleChunkPolicy):
    def __init__(self,
                 autoencoder,
                 policy_prior,
                 diffuser,
                 stage,
                 loss_fn,
                 l1_loss_scale,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.diffuser = diffuser
        self.stage = stage

        self.start_token = self.policy_prior.start_token
        self.l1_loss_scale = l1_loss_scale if stage == 3 else 0
        self.codebook_size = np.array(autoencoder.fsq_level).prod()
        
        self.loss = loss_fn
        
    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self,
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder','policy_prior','base','task_encoder'))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            self.task_encoder.requires_grad_(False)
            return optimizers
        elif self.stage == 3:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            decoder_decay, decoder_no_decay = TensorUtils.separate_no_decay(self.autoencoder.decoder)
            optimizers = [
                self.optimizer_factory(params=itertools.chain(decay, decoder_decay)),
                self.optimizer_factory(params=itertools.chain(no_decay, decoder_no_decay), weight_decay=0.)
            ]
            return optimizers

    def get_context(self, data):
        obs_emb = self.obs_encode(data,mode='base')
        task_emb = self.get_task_emb(data).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context
    
    def get_cond(self, data):
        obs_emb = self.obs_encode(data,mode='adj')
        obs_emb = obs_emb.reshape(obs_emb.shape[0], -1)
        lang_emb = self.get_task_emb(data)
        cond = torch.cat([obs_emb, lang_emb], dim=-1)
        return cond

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            data2 = copy.deepcopy(data)
            prior_loss, prior_info = self.compute_prior_loss(data)
            diff_loss, diff_info = self.compute_diffuser_loss(data2)
            loss = prior_loss + diff_loss
            info = prior_info | diff_info
            return loss, info
        elif self.stage == 2:
            return self.compute_diffuser_loss(data)
        elif self.stage == 3:
            data2 = copy.deepcopy(data)
            prior_loss, prior_info = self.compute_prior_loss(data)
            diff_loss, diff_info = self.compute_diffuser_loss(data2)
            loss = prior_loss + diff_loss
            info = prior_info | diff_info

    def compute_autoencoder_loss(self, data):
        pred, pp, pp_sample, aux_loss, _ = self.autoencoder(data["actions"])
        recon_loss = self.loss(pred, data["actions"])
        if self.autoencoder.vq_type == 'vq':
            loss = recon_loss + aux_loss
        else:
            loss = recon_loss
            
        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.sum().item(),
            'pp': pp.item(),
            'pp_sample': pp_sample.item(),
        }
        return loss, info

    def compute_prior_loss(self, data):
        data = self.preprocess_input(data, mode='base', train_mode=True)
        with torch.no_grad():
            indices = self.autoencoder.get_indices(data["actions"]).long()
        context = self.get_context(data)
        start_tokens = (torch.ones((context.shape[0], 1), device=self.device, dtype=torch.long) * self.start_token)
        x = torch.cat([start_tokens, indices[:,:-1]], dim=1)
        targets = indices.clone()
        logits = self.policy_prior(x, context)
        prior_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        with torch.no_grad():
            logits = logits[:,:,:self.codebook_size]
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1,logits.shape[-1]),1)
            sampled_indices = sampled_indices.view(-1,logits.shape[1])
        
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        l1_loss = self.loss(pred_actions, data["actions"])
        total_loss = prior_loss + self.l1_loss_scale * l1_loss
        info = {
            'prior_loss': total_loss.item(),
            'nll_loss': prior_loss.item(),
            'l1_loss': l1_loss.item()
        }
        return total_loss, info
    
    def compute_diffuser_loss(self, data):
        data = self.preprocess_input(data, mode='adj', train_mode=True)
        cond = self.get_cond(data)
        loss = self.diffuser(cond, data["actions"][:,:self.diffuser.skill_block_size,:])
        info = {
            'diff_loss': loss.item(),
        }
        return loss, info

    def sample_actions(self, data):
        data = self.preprocess_input(data, mode='base', train_mode=False)
        context = self.get_context(data)
        sampled_indices = self.policy_prior.get_indices_top_k(context, self.codebook_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        return pred_actions.detach()
    
    def adjust_actions(self, data, actions):
        data = self.preprocess_input(data, mode='adj', train_mode=False)
        cond = self.get_cond(data)
        pred_actions = self.diffuser.get_action(cond, actions, inf_steps=50, step_start=10)
        pred_actions = pred_actions.permute(1,0,2)
        return pred_actions.detach().cpu().numpy()
