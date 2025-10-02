import os, time, numpy as np
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, logdir_root: str):
        run_name = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(os.path.join(logdir_root, run_name))
        self.global_train_steps = 0

    # ---- notes / hp ----
    def add_notes(self, text: str):
        self.writer.add_text("run/notes", text, 0)

    def add_hp(self, name: str, value: float, step: int = 0):
        self.writer.add_scalar(f"hp/{name}", value, step)

    # ---- scalars ----
    def scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    # ---- histos ----
    def histos_from_model(self, model, step: int):
        for name, p in model.named_parameters():
            self.writer.add_histogram(f"params/{name}", p.detach().cpu().numpy(), step)
            if p.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", p.grad.detach().cpu().numpy(), step)

    # ---- images ----
    def image_frame(self, tag: str, frame_hwc_uint8, step: int):
        img_chw = np.transpose(frame_hwc_uint8, (2, 0, 1))
        self.writer.add_image(tag, img_chw, step)

    def close(self):
        self.writer.close()
