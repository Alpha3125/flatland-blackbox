"""Optional TensorBoard logging.

Uses ``torch.utils.tensorboard`` (needs ``pip install tensorboard``). If it is
missing, or a write fails, training carries on and a single warning is printed.
View the logs with:

    tensorboard --logdir outputs
"""

from __future__ import annotations

from mapf_nco.paths import fresh_dir


class TB:
    def __init__(self, log_dir, log=print):
        self.log = log
        self.writer = None
        self._warned = False
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:  # tensorboard not installed
            log("  (TensorBoard logging off: run 'pip install tensorboard' to enable it)")
            return
        fresh_dir(log_dir)  # each run replaces the previous logs
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def _call(self, name, *args, **kwargs):
        if self.writer is None:
            return
        try:
            getattr(self.writer, name)(*args, **kwargs)
        except Exception as e:  # never let logging kill a long run
            if not self._warned:
                self.log(f"  (TensorBoard write failed, continuing without it: {e})")
                self._warned = True

    def scalar(self, tag, value, step):
        if value is not None:
            self._call("add_scalar", tag, float(value), step)

    def histogram(self, tag, values, step):
        self._call("add_histogram", tag, values, step)

    def text(self, tag, text, step=0):
        self._call("add_text", tag, text, step)

    def image_file(self, tag, png_path, step=0):
        if self.writer is None:
            return
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(png_path)[..., :3]
            self._call("add_image", tag, img, step, dataformats="HWC")
        except Exception as e:
            if not self._warned:
                self.log(f"  (TensorBoard image failed: {e})")
                self._warned = True

    def close(self):
        self._call("flush")
        self._call("close")
