import math


class OneCycleScheduler:
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 50.0,
        final_div_factor: float = 1e4,
    ):
        """
        max_lr : peak learning rate
        total_steps : total number of optimizer.step() calls (epochs * steps_per_epoch)
        pct_start : fraction of total_steps spent increasing LR
        div_factor : initial_lr = max_lr / div_factor
        final_div_factor : min_lr = initial_lr / final_div_factor
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.curr_lr = self.initial_lr # for debugging purposes

        # compute step counts
        self.step_num = 0
        self.up_steps = int(pct_start * total_steps)
        self.down_steps = total_steps - self.up_steps

        # set lr to initial
        self.optimizer.set_lr(self.initial_lr)

    def step(self):
        self.step_num += 1

        if self.step_num <= self.up_steps:
            # linear warmup
            pct = self.step_num / self.up_steps
            lr = self.initial_lr + pct * (self.max_lr - self.initial_lr)
        else:
            # cosine annealing down
            pct = (self.step_num - self.up_steps) / max(1, self.down_steps)
            cos_out = 0.5 * (1 + math.cos(math.pi * pct))
            lr = self.final_lr + cos_out * (self.max_lr - self.final_lr)

        self.curr_lr = lr

        self.optimizer.set_lr(lr)
