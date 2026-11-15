class JumpTrigger:
    """Turns a stream of fist / no-fist readings into single jump events."""

    def __init__(self, frames_to_close: int = 2, frames_to_open: int = 3) -> None:
        self.frames_to_close = frames_to_close   # fist frames in a row to trigger a jump
        self.frames_to_open = frames_to_open     # open-hand frames needed before the next jump
        self.fist_frames = 0
        self.open_frames = 0
        self.armed = True

    def update(self, is_fist: bool) -> bool:
        """Feed one reading; returns True on the frame a jump should happen."""
        if is_fist:
            self.fist_frames += 1
            self.open_frames = 0
            if self.armed and self.fist_frames >= self.frames_to_close:
                self.armed = False
                return True
        else:
            self.open_frames += 1
            self.fist_frames = 0
            if self.open_frames >= self.frames_to_open:
                self.armed = True
        return False