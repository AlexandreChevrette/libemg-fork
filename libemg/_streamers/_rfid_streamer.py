import platform
import serial
import serial.tools.list_ports
import time
from multiprocessing import Event, Process

import numpy as np

from libemg.shared_memory_manager import SharedMemoryManager

# ============================================================
# RFID tag mapping
# ============================================================

RFID_TAGS = {
    "004255E8": 0b01000001,   # No object
    "0041A995": 0b00100010,   # Cup
    "00000000": 0b00000000,   # Spoon
    "12345678": 0b00001000,   # Bottle
}


class RFID:

    def __init__(self, baud_rate=115200, com_name=None, vid_pid=(12259, 256)):
        self.com_name = com_name
        self.vid_pid = vid_pid

        com_port = "COM3"  # Default COM port for Windows

        while True:
            self.ser = serial.Serial(com_port, baud_rate, timeout=0.1)
            if self.ser.is_open:
                
                break

    def close(self):
        self.ser.close()

    def clear_buffer(self):
        self.ser.reset_input_buffer()

    def get_tag(self):
        """
        Returns
        -------
        int
            Associated tag ID.
        None
            No new tag or unknown tag.
        """

        if not self.ser.in_waiting:
            return None

        try:
            line = self.ser.readline().decode("ascii").strip()
        except UnicodeDecodeError:
            return None

        if not line.startswith("TAG:"):
            return None

        tag = line[4:]

        if tag in RFID_TAGS:
            return RFID_TAGS[tag]

        print(f"Unknown RFID tag: {tag}")
        return None
    



class RFIDStreamer(Process):

    def __init__(self, shared_memory_items, rfid_kwargs=None):
        super().__init__(daemon=True)

        self.shared_memory_items = shared_memory_items
        self.rfid_kwargs = rfid_kwargs or {}

        self._stop_event = Event()

    def run(self):

        self.smm = SharedMemoryManager()

        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        self.rfid = RFID(**self.rfid_kwargs)
        self.rfid.clear_buffer()

        try:
            while not self._stop_event.is_set():

                tag = self.rfid.get_tag()

                if tag is None:
                    continue

                timestamp = time.perf_counter()
                def insert_rfid_tag(data):
                    input_size = self.smm.variables['rfid_tag']["shape"][0]
                    data[:] = np.vstack((
                        np.hstack([timestamp, np.array([tag], dtype=np.uint16)]),
                        data
                    ))[:input_size, :]
                    return data

                self.smm.modify_variable(
                    "rfid_tag",
                    insert_rfid_tag
                )

                self.smm.modify_variable(
                    "rfid_tag_count",
                    lambda x: x + 1
                )

        finally:
            self.rfid.close()
            self.smm.cleanup()

    def stop(self):
        self._stop_event.set()
        self.join()