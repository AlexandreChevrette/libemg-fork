import platform
import time
from multiprocessing import Event, Process, Lock
from queue import Queue, Empty
from typing import Callable, List, Optional

import numpy as np
import threading

from libemg.shared_memory_manager import SharedMemoryManager



class EmagerDummy:
    """
    Reader for Emager v3.0 frame format (8192 bytes total):

      [0]    0xAA
      [1]    0x55
      [2..8065]    EMG payload (8064 bytes) = 5376 packed 12-bit values
      [8066..8173] IMU payload area (108 bytes max)
      [8174]       IMU sample count (expected 8 or 9)
      [8175]       Alignment IMU (ignored)
      [8176..8179] Counter (big-endian uint32)  <-- frame_id
      [8180..8189] Empty (ignored)
      [8190]       0x55
      [8191]       0xAA
    """

    HDR0, HDR1 = 0xAA, 0x55
    TLR0, TLR1 = 0x55, 0xAA

    FRAME_SIZE = 8192

    EMG_START = 2
    EMG_LEN = 8064
    EMG_END = EMG_START + EMG_LEN  # 8066

    IMU_START = 8066
    IMU_AREA_LEN = 108
    IMU_NSAMPLES_I = 8174

    CTR_START = 8176

    TRAILER0_I = 8190
    TRAILER1_I = 8191

    EMG_VALUES_PER_FRAME = 5376
    CHANNELS = 64
    SAMPLES_PER_CH_PER_FRAME = EMG_VALUES_PER_FRAME // CHANNELS  # 84

    IMU_AXES = 6
    IMU_BYTES_PER_SAMPLE = IMU_AXES * 2  # int16 per axis

    def __init__(self, 
        emg_data_paths: Optional[List[str]] = None,
        frames_per_gesture: int = 74,
        fps: float = 30.0):


        self._emulated = True
        if self._emulated:
            self.ser = None
            self._emu_frame_id = 0
            self._emu_last_time = 0.0
            self._emu_fps = fps
            self._emu_frames_per_gesture = frames_per_gesture

            # Load gesture data: each entry is a (N_frames, 84, 64) uint16 array
            self._emu_gestures: List[np.ndarray] = []
            paths = list(emg_data_paths or [])

            for path in paths:
                raw = np.loadtxt(path, delimiter=",", dtype=np.uint16)
                # raw shape: (total_rows, 64) — slice into full frames only
                n_frames = raw.shape[0] // self.SAMPLES_PER_CH_PER_FRAME
                trimmed = raw[: n_frames * self.SAMPLES_PER_CH_PER_FRAME]
                blocks = trimmed.reshape(n_frames, self.SAMPLES_PER_CH_PER_FRAME, self.CHANNELS)
                self._emu_gestures.append(blocks)

            # Counters that advance through gestures and their frames
            self._emu_gesture_idx = 0          # which gesture we're on
            self._emu_gesture_frame = 0        # which frame within that gesture
            self._emu_frames_emitted_for_gesture = 0  # frames sent so far for current gesture


        

        self._buf = bytearray()
        self.pos = 0

        # stats
        self.frames_ok = 0
        self.bad_tlr = 0
        self.resyncs = 0
        self.last_ctr = None
        self.ctr_miss = 0

        # handlers
        self.frame_handlers = []

        # imu dtype
        self.imu_dtype = np.dtype(">i2") # if self.imu_endianness == "be" else np.dtype("<i2")
        self._hdr = bytes([self.HDR0, self.HDR1])


    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def clear_buffer(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def add_frame_handler(self, closure: Callable[[int, np.ndarray, np.ndarray], None]):
        """
        closure(frame_id:int, emg_block:(84,64) uint16, imu_block:(N,6) int16)
        """
        self.frame_handlers.append(closure)

    def _emit_frame(self, frame_id: int, emg_block: np.ndarray, imu_block: np.ndarray):
        for h in self.frame_handlers:
            h(int(frame_id), emg_block, imu_block)

    
    def _next_emg_block(self) -> np.ndarray:
        """
        Return the next (84, 64) uint16 block, cycling through gestures.

        Gesture rotation logic:
          - Stay on the current gesture for `_emu_frames_per_gesture` frames.
          - Then advance to the next gesture (wraps around).
          - Within each gesture, cycle through its real frames sequentially,
            wrapping back to frame 0 if the gesture data runs out before the
            dwell period ends.
        Falls back to random noise if no gesture data was loaded.
        """
        if not self._emu_gestures:
            return np.random.randint(1800, 2200,
                                     size=(self.SAMPLES_PER_CH_PER_FRAME, self.CHANNELS),
                                     dtype=np.uint16)

        # Check if we should switch to the next gesture
        if self._emu_frames_emitted_for_gesture >= self._emu_frames_per_gesture:
            self._emu_gesture_idx = (self._emu_gesture_idx + 1) % len(self._emu_gestures)
            self._emu_gesture_frame = 0
            self._emu_frames_emitted_for_gesture = 0

        gesture_data = self._emu_gestures[self._emu_gesture_idx]

        # Wrap frame index within this gesture's available data
        frame_idx = self._emu_gesture_frame % gesture_data.shape[0]
        block = gesture_data[frame_idx].copy()

        self._emu_gesture_frame += 1
        self._emu_frames_emitted_for_gesture += 1
        return block

    def _get_data_emulated(self) -> bool:
        """Rate-limited emulated frame emission — mirrors get_data()'s contract."""
        now = time.monotonic()
        if now - self._emu_last_time < 1.0 / self._emu_fps:
            return False
        self._emu_last_time = now

        emg_block = self._next_emg_block()

        # Dummy IMU: flat zeros (real device idles similarly at rest)
        imu_block = np.zeros((8, self.IMU_AXES), dtype=np.int16)

        frame_id = self._emu_frame_id
        self._emu_frame_id += 1

        self.frames_ok += 1
        if self.last_ctr is not None:
            expected = (self.last_ctr + 1) & 0xFFFFFFFF
            if frame_id != expected:
                self.ctr_miss += 1
        self.last_ctr = frame_id

        self._emit_frame(frame_id, emg_block, imu_block)
        return True

    def get_data(self) -> bool:
        return self._get_data_emulated()



class EmagerDummyStreamer(Process):
    def __init__(self, shared_memory_items, emg_data_paths: Optional[List[str]] = None,
        frames_per_gesture: int = 74,
        fps: float = 30.0):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items
        self._stop_event = Event()
        self.e = None
        self.emg_data_paths = emg_data_paths
        self.frames_per_gesture = frames_per_gesture
        self.fps = fps
        # cache shapes for ring writes
        self._shapes = {item[0]: item[1] for item in shared_memory_items if len(item) >= 2}

        # writer thread plumbing (created in run)
        self._q = None
        self._writer = None

    def run(self):
       
        self.smm = SharedMemoryManager()

        # Create shared memory variables
        for item in self.shared_memory_items:
            print("ITEM TYPE:", type(item))
            print("ITEM:", item)
            print("LEN:", len(item))
            self.smm.create_variable(*item)

        for item in self.shared_memory_items:
            self.smm.create_variable(*item)


        self.e = EmagerDummy(emg_data_paths=self.emg_data_paths, frames_per_gesture=self.frames_per_gesture, fps=self.fps)
        self.e.clear_buffer()

        # Queue carries ONE bundled item per frame to prevent desync
        self._q = Queue(maxsize=200)  # 1 item/frame; bump if you want

        def buffer_write(tag: str, data: np.ndarray) -> None:
            """
            Prepend `data` (N,D) to the shared memory buffer `tag` (H,D),
            keeping buffer size fixed, and increment `{tag}_count` by N.
            """
            if data is None:
                return

            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.ndim != 2:
                return

            count_tag = f"{tag}_count"

            def add_to_buffer(buffer, new=data):
                # prepend new rows
                new_buffer = np.vstack((new[::-1], buffer))
                # keep buffer size fixed
                return new_buffer[:buffer.shape[0], :]

            # write data
            self.smm.modify_variable(tag, add_to_buffer)

            # increment count by number of rows written
            nb_row = data.shape[0]
            self.smm.modify_variable(count_tag, lambda x, r=nb_row: x + r)

        def writer_thread_fn():
            while not self._stop_event.is_set():
                try:
                    frame_id, emg_block, imu_block = self._q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    # EMG: store as uint16 (your shared memory is now uint16)
                    emg = np.asarray(emg_block, dtype=np.uint16)

                    # IMU: int16
                    imu = np.asarray(imu_block, dtype=np.int16)

                    # Sample ID per EMG row: (frame_id * CHANNELS) + [0..N-1]
                    base = int(frame_id) * EmagerDummy.SAMPLES_PER_CH_PER_FRAME
                    sample_id = (base + np.arange(emg.shape[0], dtype=np.int64)).reshape(-1, 1)

                    # Write all three (still separate locks per tag, but all-or-nothing per frame at queue level)
                    buffer_write("emg", emg)
                    buffer_write("imu", imu)
                    buffer_write("sample_id", sample_id)

                except Exception:
                    pass
                finally:
                    self._q.task_done()

        self._writer = threading.Thread(target=writer_thread_fn, daemon=True)
        self._writer.start()

        # Frame handler: decode is already done in EmagerDummy; this just enqueues ONE item
        self.drop_count = 0
        def on_frame(frame_id, emg_block, imu_block):
            try:
                self._q.put_nowait((int(frame_id), emg_block, imu_block))
            except Exception:
                self.drop_count += 1
                if self.drop_count % 10 == 0:
                    print("DROPPED frames:", self.drop_count, "qsize:", self._q.qsize())

        self.e.add_frame_handler(on_frame)

        # Main streaming loop (avoid busy spin)
        try:
            while not self._stop_event.is_set():
                did = self.e.get_data()
                if not did:
                    time.sleep(0.001)  # 1 ms backoff when no complete frame parsed
        finally:
            self._cleanup()

    def stop(self):
        self._stop_event.set()
        self.join()

    def _cleanup(self):
        try:
            if self.e is not None:
                self.e.close()
        except Exception:
            pass
        try:
            if hasattr(self, "smm") and self.smm is not None:
                self.smm.cleanup()
        except Exception:
            pass
