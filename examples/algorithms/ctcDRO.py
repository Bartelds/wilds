import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class CTCDRO(SingleModelAlgorithm):
    """
    CTC-DRO.

    Original paper:
    @misc{bartelds2025ctcdro,
        title={CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition}, 
        author={Martijn Bartelds and Ananjan Nandi and Moussa Koulako Bala Doumbouya and Dan Jurafsky and Tatsunori Hashimoto and Karen Livescu},
        year={2025},
        eprint={2502.01777},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2502.01777}, 
    } 
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        # check config
        assert config.uniform_over_groups
        # initialize model
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # additional logging
        self.logged_fields.append('group_weight')
        # step size
        self.group_weights_step_size = config.group_dro_step_size
        self.smoothing_hyperparameter = config.smoothing_hyperparameter
        # initialize adversarial weights
        self.group_weights = torch.zeros(grouper.n_groups)
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.group_weights = self.group_weights.to(self.device)
        self.unseen_groups = torch.zeros(grouper.n_groups)
        self.unseen_groups[is_group_in_train] = 1
        self.group_losses = {}

    def process_batch(self, batch, unlabeled_batch=None):
        results = super().process_batch(batch)
        results['group_weight'] = self.group_weights
        return results

    def objective(self, results):
        """
        Takes an output of SingleModelAlgorithm.process_batch() and computes the
        optimized objective. For CTC-DRO and group DRO, the objective is the weighted average
        of losses, where groups have weights groupDRO.group_weights.
        Args:
            - results (dictionary): output of SingleModelAlgorithm.process_batch()
        Output:
            - objective (Tensor): optimized objective; size (1,).
        """
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.grouper.n_groups,
            return_dict=False)
        return (group_losses @ self.group_weights) * self.grouper.n_groups

    def _update(self, results, should_step=True):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """
        # compute group losses
        batch_group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.grouper.n_groups,
            return_dict=False)

        current_group = results['g'][0]
        self.unseen_groups[current_group] = 0
        if current_group not in self.group_losses:
            self.group_losses[current_group] = []
        self.group_losses[current_group].append(batch_group_losses[current_group]*len(results['g']))

        if not torch.sum(self.unseen_groups).item():
            # update group weights
            update_group_losses = [0 for _ in range(len(self.group_weights))]
            for group in self.group_losses:
                update_group_losses[group] = sum(self.group_losses[group])/len(self.group_losses[group])
            
            update_group_losses = torch.stack(update_group_losses)
            self.group_weights = self.group_weights * torch.exp((self.group_weights_step_size*update_group_losses)/(self.group_weights + self.smoothing_hyperparameter))
            self.group_weights = (self.group_weights/(self.group_weights.sum()))
            self.group_weights = self.group_weights.detach()
        # save updated group weights
        results['group_weight'] = self.group_weights

        # update model
        super()._update(results, should_step=should_step)
