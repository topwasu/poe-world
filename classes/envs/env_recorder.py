import numpy as np


class AtariEnvRecorder:
    def __init__(self, recording_with_bb=False):
        self.recording = False
        self.frames = []
        self.recording_with_bb = recording_with_bb

    def record(self, atari_env):
        if self.recording_with_bb:
            frame = atari_env.env.render()
            frame = np.rot90(frame, k=-1)
            frame = np.fliplr(frame)
        else:
            frame = atari_env.env._env.render()
        self.frames.append(frame)

    def start_recording(self):
        self.recording = True
        self.frames = []

    def end_recording(self):
        res = self.frames

        self.recording = False
        self.frames = []
        return res
