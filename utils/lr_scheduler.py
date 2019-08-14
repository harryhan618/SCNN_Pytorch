from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, pow, max_iter, min_lrs=1e-20, last_epoch=-1, warmup=0):
        """
        :param warmup: how many steps for linearly warmup lr
        """
        self.pow = pow
        self.max_iter = max_iter
        if not isinstance(min_lrs, list) and not isinstance(min_lrs, tuple):
            self.min_lrs = [min_lrs] * len(optimizer.param_groups)

        assert isinstance(warmup, int), "The type of warmup is incorrect, got {}".format(type(warmup))
        self.warmup = max(warmup, 0)

        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [base_lr / self.warmup * (self.last_epoch+1) for base_lr in self.base_lrs]

        if self.last_epoch < self.max_iter:
            coeff = (1 - (self.last_epoch-self.warmup) / (self.max_iter-self.warmup)) ** self.pow
        else:
            coeff = 0
        return [(base_lr - min_lr) * coeff + min_lr
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
