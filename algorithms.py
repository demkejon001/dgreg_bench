# Modified from https://github.com/facebookresearch/DomainBed
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import networks as networks
from misc import Nonparametric, chunk_tuple


ALGORITHMS = [
    'ERM',
    'SD',
    'VREx',
    'IB_ERM',
    'RDM',
    'IRM',
    'IB_IRM',
    'GroupDRO',
    'EQRM',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'ANDMask',
    'SANDMask',
    'Fish',
    'CORAL',
    'MMD',
    'IGA',
    'DAEL',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__()
        self.hparams = hparams
        self.n_domains = n_domains

    def update(self, x, y, loss_fn):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.regressor = networks.Regressor(
            input_shape,
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_regressor'])

        self.network = nn.Sequential(self.featurizer, self.regressor)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, x, y, loss_fn):
        y_hat = self.predict(x)
        results = loss_fn(y_hat, y)
        loss = results["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["max_grad_norm"], )
        results["grad_norm"] = grad_norm
        self.optimizer.step()

        return results

    def predict(self, x):
        return self.network(x)


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self._sd_reg = hparams["sd_reg"]
        self.ctr = 0
        self.anneal_step = self.hparams.get("anneal", 0)

    @property
    def sd_reg(self):
        if self.ctr >= self.anneal_step:
            return self._sd_reg
        else:
            return self.anneal_step / self.ctr * self._sd_reg

    def update(self, x, y, loss_fn):
        y_hat = self.predict(x)
        results = loss_fn(y_hat, y)
        loss = results["loss"]

        penalty = (y_hat ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["max_grad_norm"], )
        results["grad_norm"] = grad_norm
        self.optimizer.step()

        results["loss"] = objective.item()
        results["mse"] = loss.item()
        results["penalty"] = penalty.item()

        self.ctr += 1
        return results


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.ctr = 0
        self.anneal_step = self.hparams.get("anneal", 0)
    
    @property
    def penalty_weight(self):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0
        
        if self.update_count >= self.anneal_step:
            return penalty_weight
        else:
            return self.anneal_step / self.ctr * penalty_weight

    def update(self, x, y, loss_fn):
        penalty_weight = self.penalty_weight

        y_hat = self.network(x)
        results = loss_fn(y_hat, y, reduction="none")

        losses = results["loss"].view(self.n_domains, -1).mean(1)
        mean = losses.mean()
        variance = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * variance

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["max_grad_norm"], )
        results["grad_norm"] = grad_norm
        self.optimizer.step()

        self.update_count += 1
        results["loss"] = loss.item()
        results["mse"] = mean.item()
        results["variance"] = variance.item()
        return results


class IB_ERM(ERM):
    """Information Bottleneck based ERM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.regressor.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))
        self.anneal_step = self.hparams.get("anneal", 0)
    
    @property
    def penalty_weight(self):
        penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                        >= self.hparams['ib_penalty_anneal_iters'] else
                        0.0)
        
        if self.update_count >= self.anneal_step:
            return penalty_weight
        else:
            return self.anneal_step / self.ctr * penalty_weight

    def update(self, x, y, loss_fn):
        ib_penalty_weight = self.penalty_weight
        ib_penalty = 0.

        feats = self.featurizer(x)
        y_hat = self.regressor(feats)
        results = loss_fn(y_hat, y)

        n_samples_per_domain = len(y) // self.n_domains
        domain_feats = feats.view(self.n_domains, n_samples_per_domain, -1)
        ib_penalty = torch.var(domain_feats, dim=1).mean()

        mse = results["loss"]
        loss = mse + ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.regressor.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["max_grad_norm"], )
        results["grad_norm"] = grad_norm
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        results["mse"] = mse.item()
        results["loss"] = loss.item()
        results["IB_penalty"] = ib_penalty.item()
        return results


class RDM(ERM):
    """RDM - Domain Generalization via Risk Distribution Matching (https://arxiv.org/abs/2310.18598) """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy
    
    def update(self, x, y, loss_fn):
        matching_penalty_weight = (self.hparams['rdm_lambda'] if self.update_count
                          >= self.hparams['rdm_penalty_anneal_iters'] else
                          0.)

        variance_penalty_weight = (self.hparams['variance_weight'] if self.update_count
                          >= self.hparams['rdm_penalty_anneal_iters'] else
                          0.)

        # all_x = torch.cat([x for x, y in minibatches])
        y_hat = self.predict(x)

        results = loss_fn(y_hat, y, reduction="none")
        all_confs_envs = results["loss"].view(self.n_domains, -1)
                
        domain_losses = all_confs_envs.mean(1)
        mse = domain_losses.mean()
        
        ## find the worst domain
        worst_env_idx = torch.argmax(domain_losses)
        all_confs_worst_env = all_confs_envs[worst_env_idx]

        ## flatten the risk
        # all_confs_worst_env_flat = torch.flatten(all_confs_worst_env)
        all_confs_all_envs_flat = torch.flatten(all_confs_envs)

        matching_penalty = self.mmd(all_confs_worst_env.unsqueeze(1), all_confs_all_envs_flat.unsqueeze(1)) 
        
        ## variance penalty
        variance_penalty = torch.var(all_confs_all_envs_flat)
        variance_penalty += torch.var(all_confs_worst_env)
        
        total_loss = mse + matching_penalty_weight * matching_penalty + variance_penalty_weight * variance_penalty
            
        if self.update_count == self.hparams['rdm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["rdm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.update_count += 1

        results["loss"] = total_loss.item()
        results["mse"] = mse.item()
        results["matching_penalty"] = matching_penalty.item()
        results["rdm_lambda"] = self.hparams["rdm_lambda"]
        results["variance_penalty"] = variance_penalty.item()
        return results


class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        dummy_w = torch.eye(num_classes).requires_grad_()
        self.register_buffer("dummy_w", dummy_w)

    def _irm_penalty(self, losses: torch.Tensor):
        return autograd.grad(losses, self.dummy_w, create_graph=True)[0].pow(2).mean()

    def update(self, x, y, loss_fn):
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)

        y_hat = self.network(x)
        y_hat = y_hat @ self.dummy_w
        results = loss_fn(y_hat, y, reduction="none")

        mse = results["loss"]
        penalty = 0.
        domain_losses = mse.view(self.n_domains, -1).mean(1)
        for mse_i in domain_losses:
            penalty += self._irm_penalty(mse_i)
        penalty /= self.n_domains
        mse = mse.mean()

        loss = mse + penalty_weight * penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        results["loss"] = loss.item()
        results["mse"] = mse.item()
        results["penalty"] = penalty.item()

        return results


class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.regressor.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))
        dummy_w = torch.eye(num_classes).requires_grad_()
        self.register_buffer('dummy_w', dummy_w)

    def _irm_penalty(self, losses: torch.Tensor):
        return autograd.grad(losses, self.dummy_w, create_graph=True)[0].pow(2).mean()

    def update(self, x, y, loss_fn):
        irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)

        feats = self.featurizer(x)
        y_hat = self.regressor(feats)
        y_hat = y_hat @ self.dummy_w
        results = loss_fn(y_hat, y, reduction="none")
        losses = results["loss"]

        irm_penalty = 0
        for loss_i in losses.view(self.n_domains, -1).mean(1):
            irm_penalty += self._irm_penalty(loss_i)
        irm_penalty /= self.n_domains

        n_samples_per_domain = len(y) // self.n_domains
        domain_feats = feats.view(self.n_domains, n_samples_per_domain, -1)
        ib_penalty = domain_feats.var(dim=1).mean()

        # Compile loss
        mse = losses.mean()
        loss = mse
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.regressor.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        results["loss"] = loss.item()
        results["mse"] = mse.item()
        results["IRM_penalty"] = irm_penalty.item()
        results["IB_penalty"] = ib_penalty.item()

        return results


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.register_buffer("q", torch.ones(self.n_domains))

    def update(self, x, y, loss_fn):
        y_hat = self.predict(x)
        results = loss_fn(y_hat, y, reduction="none")
        losses = results["loss"]
        domain_losses = losses.view(self.n_domains, -1).mean(1)

        self.q *= (self.hparams["groupdro_eta"] * domain_losses.data).exp()
        self.q /= self.q.sum()

        loss = torch.dot(domain_losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["max_grad_norm"], )
        results["grad_norm"] = grad_norm
        self.optimizer.step()

        results["loss"] = loss.item()
        results["mse"] = domain_losses.mean().item()

        return results


class EQRM(ERM):
    """
    Empirical Quantile Risk Minimization (EQRM).
    Algorithm 1 from [https://arxiv.org/pdf/2207.09944.pdf].
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams, dist=None):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer('alpha', torch.tensor(self.hparams["eqrm_quantile"], dtype=torch.float64))
        if dist is None:
            self.dist = Nonparametric()
        else:
            self.dist = dist

    def update(self, x, y, loss_fn):
        y_hat = self.predict(x)
        results = loss_fn(y_hat, y, reduction="none")
        losses = results["loss"]
        env_risks = losses.view(self.n_domains, -1).mean(dim=1)

        if self.update_count < self.hparams["eqrm_burnin_iters"]:
            # Burn-in/annealing period uses ERM like penalty methods (which set penalty_weight=0, e.g. IRM, VREx.)
            loss = torch.mean(env_risks)
        else:
            # Loss is the alpha-quantile value
            self.dist.estimate_parameters(env_risks)
            loss = self.dist.icdf(self.alpha)

        if self.update_count == self.hparams['eqrm_burnin_iters']:
            # Reset Adam (like IRM, VREx, etc.), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["eqrm_lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        results["loss"] = loss.item()

        return results


class AbstractCausIRL(ERM):
    '''Abstract class for Causality based invariant representation learning 
    algorithm from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, n_domains, hparams, gaussian):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
            rbf_gamma = torch.tensor([0.001, 0.01, 0.1, 1, 10, 100, 1000]).unsqueeze(-1).unsqueeze(-1)
            self.register_buffer("rbf_gamma", rbf_gamma)
        else:
            self.kernel_type = "mean_cov"

    def cdist_squared(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y):
        D = self.cdist_squared(x, y)
        D = D.unsqueeze(0)
        K = (D * -self.rbf_gamma).exp().sum(0)
        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, loss_fn):
        feats = self.featurizer(x)
        y_hat = self.regressor(feats)
        results = loss_fn(y_hat, y)
        mse = results["loss"]

        indices = np.arange(len(feats))
        np.random.shuffle(indices)
        slice = np.random.randint(2, len(feats)-2)
        first = feats[indices[:slice]]
        second = feats[indices[slice:]]
        penalty = torch.nan_to_num(self.mmd(first, second))

        loss = mse + (self.hparams['mmd_gamma'] * penalty)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results["loss"] = loss.item()
        results["mse"] = mse.item()
        results["penalty"] = penalty.item()
        return results


class CausIRL_CORAL(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the CORAL distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams, gaussian=False)


class CausIRL_MMD(AbstractCausIRL):
    '''Causality based invariant representation learning algorithm using the MMD distance from (https://arxiv.org/abs/2206.11646)'''
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams, gaussian=True)


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.tau = hparams["tau"]
        self.register_buffer("I_ndomain", torch.eye(self.n_domains))

    def update(self, x, y, loss_fn):
        y_hat = self.predict(x)
        results = loss_fn(y_hat, y, reduction="none")
        losses = results["loss"]
        domain_losses = losses.view(self.n_domains, -1).mean(dim=1)

        # Will return a tuple of gradients for each loss in domain_losses
        def get_param_gradients(v):
            return torch.autograd.grad(domain_losses, self.network.parameters(), v)

        param_gradients = torch.vmap(get_param_gradients)(self.I_ndomain)

        self.optimizer.zero_grad()
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()

        results["loss"] = domain_losses.mean().item()
        return results

    def mask_grads(self, gradients, params):
        for param, grads in zip(params, gradients):
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0


class SANDMask(ANDMask):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    (https://arxiv.org/abs/2106.02266)
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.k = hparams["k"]

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021. (https://arxiv.org/abs/2104.09937)
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        featurizer = networks.Featurizer(input_shape, self.hparams)
        regressor = networks.Regressor(
            input_shape,
            featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_regressor'])
        self.network = nn.Sequential(featurizer, regressor)

        featurizer = networks.Featurizer(input_shape, self.hparams)
        regressor = networks.Regressor(
            input_shape,
            featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_regressor'])
        self.network_inner = nn.Sequential(featurizer, regressor)

        for name, param in self.network.named_buffers():
            if "num_batches_tracked" in name:
                param.data = param.data.float()
        for name, param in self.network_inner.named_buffers():
            if "num_batches_tracked" in name:
                param.data = param.data.float()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def fish(self, meta_weights, inner_weights, lr_meta):
        for k in meta_weights:
            meta_weights[k] += lr_meta * (inner_weights[k] - meta_weights[k])
        return meta_weights

    def update(self, x, y, loss_fn):
        n_samples_per_domain = len(y) // self.n_domains 
        y = y.view(self.n_domains, n_samples_per_domain, -1)
        if isinstance(x, tuple):
            x = chunk_tuple(x, self.n_domains)
        else:
            x = x.view(self.n_domains, n_samples_per_domain, -1)

        self.network_inner.load_state_dict(self.network.state_dict())

        domain_indices = torch.randperm(self.n_domains)
        mean_results = None
        for i in domain_indices:
            results = loss_fn(self.network_inner(x[i]), y[i])
            
            # Logging
            if mean_results is None:
                mean_results = results.copy()
                mean_results["loss"] = mean_results["loss"].item()
            else:
                for k, v in mean_results.items():
                    if k == "loss":
                        mean_results[k] += results[k].item()
                    else:
                        mean_results[k] += results[k]

            loss = results["loss"]
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.load_state_dict(meta_weights)

        for k in mean_results:
            mean_results[k] /= self.n_domains
        return mean_results

    def predict(self, x):
        return self.network(x)


class MMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.n_domains_choose_2 = (n_domains * (n_domains - 1)) // 2
        mmd_scale = torch.tensor([-2 for _ in range(self.n_domains_choose_2)] + [n_domains-1 for _ in range(n_domains)]).float()
        self.register_buffer("mmd_scale", mmd_scale)
        rbf_gamma = torch.tensor([0.001, 0.01, 0.1, 1, 10, 100, 1000]).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("rbf_gamma", rbf_gamma)

    # equivalent to torch.cdist(x1, x2)**2
    def cdist_squared(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = x1_norm + x2_norm.transpose(-1, -2) - 2*torch.bmm(x1, x2.transpose(-1, -2))
        return res.clamp_min_(1e-30)

    def get_pairwise_tensors(self, x):
        left_side = []
        right_side = []
        for i in range(self.n_domains):
            for j in range(i+1, self.n_domains):
                left_side.append(x[i])
                right_side.append(x[j])
        for i in range(self.n_domains):
            left_side.append(x[i])
            right_side.append(x[i])
        return torch.stack(left_side), torch.stack(right_side)

    def gaussian_kernel(self, x, y):
        D = self.cdist_squared(x, y)
        D = D.unsqueeze(1)
        K = (D * -self.rbf_gamma).exp().sum(1)
        return K

    # Although confusing at first, this is the same as the original mmd() function 
    #   from Facebook, but this is vectorized
    def mmd_vec(self, x):
        xl, xr = self.get_pairwise_tensors(x)
        K = self.gaussian_kernel(xl, xr)
        K = K.mean([1, 2])
        return (K * self.mmd_scale).sum() / self.n_domains_choose_2

    def update(self, x, y, loss_fn):
        feats = self.featurizer(x)
        y_hat = self.regressor(feats)
        results = loss_fn(y_hat, y)
        mse = results["loss"]

        n_samples_per_domain = len(y) // self.n_domains
        domain_feats = feats.view(self.n_domains, n_samples_per_domain, -1)
        penalty = self.mmd_vec(domain_feats)

        loss = mse + (self.hparams['mmd_gamma']*penalty)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results["loss"] = loss.item()
        results["mse"] = mse.item()
        results["penalty"] = penalty.item()
        return results


class CORAL(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)

    def mmd(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, x, y, loss_fn):
        feats = self.featurizer(x)
        y_hat = self.regressor(feats)
        results = loss_fn(y_hat, y)
        mse = results["loss"]

        n_samples_per_domain = len(y) // self.n_domains
        domain_feats = feats.view(self.n_domains, n_samples_per_domain, -1)

        domain_feat_mean = torch.mean(domain_feats, dim=1, keepdim=True)
        domain_centered_feats = domain_feats - domain_feat_mean
        domain_feat_cov = torch.bmm(domain_centered_feats, domain_centered_feats.transpose(-1, -2))

        penalty = 0
        for i in range(self.n_domains):
            for j in range(i + 1, self.n_domains):
                mean_diff = (domain_feat_mean[i] - domain_feat_mean[j]).pow(2).mean()
                cov_diff = (domain_feat_cov[i] - domain_feat_cov[j]).pow(2).mean()
                penalty += mean_diff + cov_diff

        penalty /= (self.n_domains * (self.n_domains - 1) / 2)

        loss = mse + (self.hparams['mmd_gamma']*penalty)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results["loss"] = loss.item()
        results["mse"] = mse.item()
        results["penalty"] = penalty.item()
        return results


class IGA(ERM):
    """
    Inter-environmental Gradient Alignment (https://arxiv.org/abs/2008.01883v2)
    """
    def __init__(self, in_features, num_classes, n_domains, hparams):
        super().__init__(in_features, num_classes, n_domains, hparams)
        I_ndomain = torch.eye(self.n_domains)
        self.register_buffer("I_ndomain", I_ndomain)

    def update(self, x, y, loss_fn):
        y_hat = self.predict(x)
        results = loss_fn(y_hat, y, reduction="none")
        losses = results["loss"]
        domain_loss = losses.view(self.n_domains, -1).mean(1)

        def get_param_gradients(v):
            return torch.autograd.grad(domain_loss, self.network.parameters(), v, create_graph=True)
        param_gradients = torch.vmap(get_param_gradients)(self.I_ndomain)

        # compute trace penalty
        penalty_value = 0
        for g in param_gradients:
            penalty_value += (g - g.mean(0)).pow(2).sum()     

        mean_loss = domain_loss.mean()
        objective = mean_loss + self.hparams['penalty_coef'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        results["loss"] = objective.item()
        results["mse"] = mean_loss.item()
        results["penalty"] = penalty_value.item()
        return results


class DAEL(Algorithm):
    """
    Domain Adaptive Ensemble Learning
    https://arxiv.org/abs/2003.07325
    """
    def __init__(self, input_shape, num_classes, n_domains, hparams):
        super().__init__(input_shape, num_classes, n_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.regressors = nn.ModuleList([
                networks.Regressor(
                input_shape,
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_regressor'])
            for _ in range(n_domains)
        ])

        # self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.featurizer.parameters(), self.regressors.parameters()),
            # self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        # def wa(x): return x + torch.randn_like(x) * .001
        # def sa(x): return x + torch.randn_like(x) * .01
        # self.weak_aug = wa
        # self.strong_aug = sa
        self.weak_aug = hparams["weak_aug"]
        self.strong_aug = hparams["strong_aug"]

    # def update(self, minibatches, unlabeled=None):
    def update(self, x, y, loss_fn):
        weak_aug = self.weak_aug(x)
        strong_aug = self.strong_aug(x)

        weak_feats = self.featurizer(weak_aug)
        strong_feats = self.featurizer(strong_aug)

        n_samples_per_domain = len(y) // self.n_domains
        weak_domain_feats = weak_feats.view(self.n_domains, n_samples_per_domain, -1)
        non_expert_y_hats = [0 for _ in range(self.n_domains)]
        expert_y_hats = []
        for i in range(self.n_domains):
            expert_y_hats.append(self.regressors[i](weak_domain_feats[i]))
            non_expert_y_hat = self.regressors[i](strong_feats).view(self.n_domains, n_samples_per_domain, -1)
            for j in range(self.n_domains):
                if i != j:
                    non_expert_y_hats[j] = non_expert_y_hats[j] + non_expert_y_hat[j]

        expert_y_hats = torch.cat(expert_y_hats, dim=0)
        non_expert_y_hats = torch.cat(non_expert_y_hats, dim=0) / (self.n_domains - 1)

        results = loss_fn(expert_y_hats, y)
        expert_loss = results["loss"]
        non_expert_loss = F.mse_loss(non_expert_y_hats, expert_y_hats.detach())
        loss = expert_loss + non_expert_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams["max_grad_norm"], )
        results["grad_norm"] = grad_norm
        self.optimizer.step()

        results["non_expert_loss"] = non_expert_loss.item()
        results["expert_loss"] = expert_loss.item()
        results["loss"] = loss.item()

        return results

    def predict(self, x):
        feats = self.featurizer(x)
        output = None
        for regressor in self.regressors:
            if output is None:
                output = regressor(feats)
            else:
                output = output + regressor(feats)
        return output / self.n_domains


