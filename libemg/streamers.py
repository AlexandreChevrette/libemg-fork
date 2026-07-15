import time
import numpy as np

from multiprocessing import Process, Event, Lock
from libemg._streamers._emagerv3_streamer import EmagerV3Streamer
from libemg._streamers._rfid_streamer import RFIDStreamer
from libemg._streamers._dummy_data_streamer import EmagerDummyStreamer

def emagerv3_streamer(shared_memory_items=None, **kwargs):
    """The streamer for the emager armband (v3).

    Connects to the EMaGer v3 cuff (8192-byte framed protocol with packed
    12-bit EMG, IMU, and a frame counter) over a serial port and exposes
    decoded samples via shared memory. For v1.0/v1.1 hardware, use
    :func:`emager_streamer` instead.

    Parameters
    ----------
    shared_memory_items : list (optional)
        Shared memory configuration parameters for the streamer in format:
        ["tag", (size), datatype]. Defaults expose the modalities below.
    emager_kwargs: dict passed to Emager3. Supported keys:
          baud_rate (int, default 3000000), com_name (str),
          vid_pid (tuple, default (12259, 256)), debug (bool).

    The default shared memory exposes:
      - 'emg'       : (2000, 64) uint16 rolling buffer (rows = EMG samples)
      - 'imu'       : (2000, 6)  int16  rolling buffer (rows = IMU samples)
      - 'sample_id' : (2000, 1)  int64  aligned row-for-row with EMG samples
      - 'emg_count', 'imu_count', 'sample_id_count' as (1, 1) int64

    Returns
    ----------
    Object: streamer
        The emager streamer object.
    Object: shared memory
        The shared memory object.
    Examples
    ---------
    >>> streamer, shared_memory = emagerv3_streamer()
    """
    if shared_memory_items is None:
        shared_memory_items = []
        shared_memory_items.append(['emg', (2000, 64), np.uint16])
        shared_memory_items.append(['emg_count', (1, 1), np.int64])
        shared_memory_items.append(['imu', (2000, 6), np.int16])
        shared_memory_items.append(['imu_count', (1, 1), np.int64])
        shared_memory_items.append(['sample_id', (2000, 1), np.int64])
        shared_memory_items.append(['sample_id_count', (1, 1), np.int64])

    for item in shared_memory_items:
        if len(item) == 3:
            item.append(Lock())

    ema = EmagerV3Streamer(shared_memory_items, emager_kwargs=kwargs)
    ema.start()
    return ema, shared_memory_items


def rfid_streamer(shared_memory_items=None):
    
    if shared_memory_items is None:
        shared_memory_items = []
        shared_memory_items.append(['rfid_tag', (2000, 1+1), np.uint16]) # Timestamp, tag
        shared_memory_items.append(['rfid_tag_count', (1, 1), np.int64])


    for item in shared_memory_items:
        if len(item) == 3:
            item.append(Lock())

    rfidProcess = RFIDStreamer(shared_memory_items)
    rfidProcess.start()
    return rfidProcess, shared_memory_items



def dummy_streamer(shared_memory_items=None, emg_data_paths=None, frames_per_gesture=74, fps=30.0):
    if shared_memory_items is None:
        shared_memory_items = [
            ['emg',            (2000, 64), np.uint16],
            ['emg_count',      (1, 1),     np.int64],
            ['imu',            (2000, 6),  np.int16],
            ['imu_count',      (1, 1),     np.int64],
            ['sample_id',      (2000, 1),  np.int64],
            ['sample_id_count',(1, 1),     np.int64],
        ]

    for item in shared_memory_items:
        if len(item) == 3:
            item.append(Lock())

    ema = EmagerDummyStreamer(shared_memory_items, emg_data_paths=emg_data_paths, frames_per_gesture=frames_per_gesture, fps=fps)
    ema.start()
    return ema, shared_memory_items