import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class OneCycleCooldown(_LRScheduler):
    """LR scheduler that implements the one-cycle learning rate policy.

    Followed by a cooldown period where the learning rate is gradually reduced to a minimum value.
    """

    def __init__(
        self,
        optimizer,
        warmup,
        cooldown,
        cooldown_final,
        initial_lr,
        max_lr,
        final_lr=1e-6,
    ):
        """optimizer (Optimizer): Wrapped optimizer.

        warmup: number of epochs to warmup for.
        cooldown: number of epochs to cooldown for.
        initial_lr: initial learning rate.
        max_lr: maximum learning rate.
        final_lr: final learning rate.
        """
        self.warmup = warmup
        self.cooldown = cooldown
        self.cooldown_final = cooldown_final
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr = self.get_lr_factor(epoch=self.last_epoch)
        return [lr for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * epoch / self.warmup
        elif epoch <= self.warmup + self.cooldown:
            lr = (
                self.max_lr
                - (self.max_lr - self.initial_lr) * (epoch - self.warmup) / self.cooldown
            )
        elif epoch <= self.warmup + self.cooldown + self.cooldown_final:
            lr = (
                self.initial_lr
                - (self.initial_lr - self.final_lr)
                * (epoch - self.warmup - self.cooldown)
                / self.cooldown_final
            )
        else:
            lr = self.final_lr
        return lr


class WarmupToConstant(_LRScheduler):
    """Gradually warm-up learning rate in optimizer to a constant value."""

    def __init__(self, optimizer: Optimizer, num_steps: int = 100) -> None:
        """
        args:
            optimizer (Optimizer): Wrapped optimizer.
            num_steps: target learning rate is reached at num_steps.
        """
        self.num_steps = num_steps
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        if self.last_epoch > self.num_steps:
            return [base_lr for base_lr in self.base_lrs]
        return [(base_lr / self.num_steps) * self.last_epoch for base_lr in self.base_lrs]
