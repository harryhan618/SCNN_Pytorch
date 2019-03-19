from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, pow, max_iter, min_lrs=1e-5, last_epoch=-1):
        self.pow = pow
        self.max_iter = max_iter
        if not isinstance(min_lrs, list) and not isinstance(min_lrs, tuple):
            self.min_lrs = [min_lrs] * len(optimizer.param_groups)

        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.max_iter:
            coeff = (1 - self.last_epoch / self.max_iter) ** self.pow
        else:
            coeff = 0
        return [(base_lr - min_lr) * coeff + min_lr
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
