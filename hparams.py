# Modified from https://github.com/facebookresearch/DomainBed
import numpy as np
import torch
import misc
from reg_datasets import DATASETS


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}
    dataset_info = DATASETS[dataset]

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        if random_val_fn is None:
            hparams[name] = (default_val, default_val)
        else:
            random_state = np.random.RandomState(
                misc.seed_hash(random_seed, name)
            )
            hparams[name] = (default_val, random_val_fn(random_state))

    _hparam('class_balanced', False, lambda r: False)
    _hparam('nonlinear_regressor', True, None)

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.
    if algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RDM": 
        _hparam('rdm_penalty_anneal_iters', 0, None)
        _hparam('rdm_lambda', .1, lambda r: 10**r.uniform(-3, 0))
        _hparam('variance_weight', .01, lambda r: 10**r.uniform(-4, -1))
        lr_exp = np.log10(dataset_info.lr)
        _hparam('rdm_lr', dataset_info.lr, lambda r: 10**r.uniform(lr_exp-1.5, lr_exp+1.5))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('irm_penalty_anneal_iters', 0, None)

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-3, lambda r: 10**r.uniform(-4, -2))

    elif algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  r.choice([1, 2]))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 3))
        _hparam('vrex_penalty_anneal_iters', 0, None)

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -2))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty_coef', .1, lambda r: 10**r.uniform(-3, 1))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('ib_penalty_anneal_iters', 0, None)

    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('irm_penalty_anneal_iters', 0, None)
        _hparam('ib_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('ib_penalty_anneal_iters', 0, None)

    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    elif algorithm == 'EQRM':
        _hparam('eqrm_quantile', 0.75, lambda r: r.uniform(0.5, 0.99))
        _hparam('eqrm_burnin_iters', 2500, lambda r: 10 ** r.uniform(2.5, 3.5))
        _hparam('eqrm_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))

    elif algorithm == 'ERMPlusPlus':
        _hparam('linear_lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    elif algorithm == "DAEL":
        if "distshift" in dataset:
            def get_noise_aug(std):
                def noise_aug(x):
                    return torch.normal(x[0], std), x[1]
                return noise_aug

            def stronger_aug(x):
                def mask(states):
                    with torch.no_grad():
                        states = states.clone()
                    B, C, H, W = states.shape
                    x1, y1, x2, y2 = np.random.randint(0, H, 4)
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1
                    
                    states[:, :, x1:x2, y1:y2] = .33
                    return states
                return mask(x[0]), x[1]

            weak_aug = get_noise_aug(.05)
            strong_aug = get_noise_aug(.15)
            _hparam("weak_aug", weak_aug, lambda r: get_noise_aug(10**r.uniform(-5, -1)))
            _hparam("strong_aug", strong_aug, lambda r: r.choice([get_noise_aug(10**r.uniform(-5, -1)), stronger_aug]))
        else:
            def weak_aug(x):
                return torch.normal(x, .001)
            def strong_aug(x):
                return torch.normal(x, .003)

            def make_aug_fn(std):
                def aug_fn(x):
                    return torch.normal(x, std)
                return aug_fn
            
            _hparam("weak_aug", weak_aug, lambda r: make_aug_fn(10**r.uniform(-5, -1)))
            _hparam("strong_aug", strong_aug, lambda r: make_aug_fn(10**r.uniform(-5, -1)))

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    _hparam('weight_decay', 0., lambda r: 10**r.uniform(-7, -4))

    _hparam('batch_size', dataset_info.batch_size, None)
    _hparam('max_steps', dataset_info.max_steps, None)
    lr_exp = np.log10(dataset_info.lr)
    _hparam('lr', dataset_info.lr, lambda r: 10**r.uniform(lr_exp-1.5, lr_exp+1.5))
    _hparam('max_grad_norm', dataset_info.max_grad_norm, None)

    if "distshift" in dataset:
        _hparam('cnn_hidden', 32, None)
        _hparam('norm', "dropout", lambda r: r.choice(["none", "batch", "dropout", "layer"]))
    else:
        _hparam('mlp_width', 32, lambda r: r.choice([16, 32, 64]))
        _hparam('mlp_depth', 2, lambda r: r.choice([2, 3]))
        _hparam('norm', "none", lambda r: r.choice(["none", "layer", "dropout"]))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
