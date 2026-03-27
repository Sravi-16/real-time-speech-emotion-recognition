import queue
import threading
import time
from typing import Optional, Tuple

import librosa
import numpy as np
import sounddevice as sd

from backend.audio_utils import TARGET_SR


class MicrophoneStreamer:
    def __init__(self, chunk_seconds: float = 2.0, overlap_seconds: float = 1.0):
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.raw_queue: "queue.Queue[Tuple[np.ndarray, float, int]]" = queue.Queue(maxsize=32)
        self.running = False
        self.stream: Optional[sd.InputStream] = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            return
        mono = np.mean(indata, axis=1).astype(np.float32)
        try:
            self.raw_queue.put_nowait((mono, time.time(), int(self.stream.samplerate if self.stream else TARGET_SR)))
        except queue.Full:
            pass

    def start(self, samplerate: int = TARGET_SR):
        with self._lock:
            if self.running:
                return
            self.running = True
            self._buffer = np.array([], dtype=np.float32)
            self.stream = sd.InputStream(
                channels=1,
                samplerate=samplerate,
                dtype="float32",
                blocksize=int(0.1 * samplerate),
                callback=self._callback,
            )
            self.stream.start()

    def stop(self):
        with self._lock:
            self.running = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self._buffer = np.array([], dtype=np.float32)
            while not self.raw_queue.empty():
                try:
                    self.raw_queue.get_nowait()
                except queue.Empty:
                    break

    def get_chunk(self) -> Optional[Tuple[np.ndarray, float]]:
        if not self.running:
            return None

        chunk_size = int(self.chunk_seconds * TARGET_SR)
        overlap_size = int(self.overlap_seconds * TARGET_SR)
        latest_ts = time.time()

        while not self.raw_queue.empty():
            try:
                block, ts, block_sr = self.raw_queue.get_nowait()
                latest_ts = ts
            except queue.Empty:
                break

            if block_sr != TARGET_SR:
                block = librosa.resample(block, orig_sr=block_sr, target_sr=TARGET_SR)
            self._buffer = np.concatenate([self._buffer, block], axis=0)

        if len(self._buffer) < chunk_size:
            return None

        chunk = self._buffer[:chunk_size].copy()
        self._buffer = self._buffer[chunk_size - overlap_size :]
        return chunk, latest_ts
