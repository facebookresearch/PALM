
import pytorch_lightning
from pytorch_lightning.callbacks.progress import TQDMProgressBar

class TrialNameCallback(pytorch_lightning.Callback):
    def __init__(self, trial_name):
        self.trial_name = trial_name
    def on_fit_start(self, trainer, pl_module):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                # Broadcast the trial_name to all other processes
                dist.broadcast_object_list([self.trial_name], src=0)
            else:
                # Receive the broadcasted trial_name
                trial_name_list = [None]
                dist.broadcast_object_list(trial_name_list, src=0)
                self.trial_name = trial_name_list[0]
        pl_module.trial_name = self.trial_name

    def on_test_start(self, trainer, pl_module):
        self.on_fit_start(trainer, pl_module)


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        items["step"] = self.trainer.global_step
        return items
