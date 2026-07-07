import platform
import time
import threading
from multiprocessing import Event, Process
from queue import Queue, Empty
from typing import Callable

import numpy as np
import serial  # pyserial
import serial.tools.list_ports

from libemg.shared_memory_manager import SharedMemoryManager



# ============================================================
# Emager3 (v3.0) 8192-byte frames with packed 12-bit EMG + IMU + counter
# ============================================================

class RFID:
    

    def __init__(self, baud_rate: int, com_name=None, vid_pid=(12259, 256)):
        self.com_name = com_name
        self.vid_pid = vid_pid

        ports = list(serial.tools.list_ports.comports())
        com_port = None

        for p in ports:
            if self.com_name is None:
                if (p.vid, p.pid) == self.vid_pid:
                    com_port = p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
                    break
            else:
                if self.com_name in (p.description or ""):
                    com_port = p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
                    break

        if com_port is None:
            ports_info = []
            for p in ports:
                dev = getattr(p, "device", None) or getattr(p, "name", None) or "<unknown>"
                desc = getattr(p, "description", "") or "<no description>"
                vid = getattr(p, "vid", None)
                pid = getattr(p, "pid", None)
                ports_info.append(f"{dev} - {desc} (VID: {vid}, PID: {pid})")
            avail = "\n".join(f"  - {pi}" for pi in ports_info) if ports_info else "  (no serial ports found)"
            raise RuntimeError(f"Could not find serial port for Emager3. Available ports:\n{avail}")

        # non-blocking; we buffer ourselves
        self.ser = serial.Serial(com_port, baud_rate, timeout=0)
        self.ser.close()

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

    def connect(self):
        self.ser.open()

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

    def _unpack_12bit_be(self, packed: bytes, n_values: int) -> np.ndarray:
        """
        Emager3 (v3.0) 8192-byte frames with packed 12-bit EMG + IMU + counter
        Unpacks the 8064-byte EMG payload into 5376 uint16 values.
        """
        
        b = np.frombuffer(packed, dtype=np.uint8).astype(np.uint16)
        out = np.empty((2 * (len(b) // 3),), dtype=np.uint16)
        out[0::2] = (b[0::3] << 4) | (b[1::3] >> 4)
        out[1::2] = ((b[1::3] & 0x0F) << 8) | b[2::3]
        return out[:n_values]

    def get_data(self) -> bool:
        """
        Returns True if at least one full frame was parsed+emitted in this call.
        """
        try:
            n_av = self.ser.in_waiting
        except Exception:
            return False
        if n_av <= 0:
            return False

        data = self.ser.read(n_av)
        if not data:
            return False

        self._buf += data

        emitted_any = False

        while True:
            h = self._buf.find(self._hdr, self.pos)

            if h < 0:
                keep = min(len(self._buf), self.FRAME_SIZE - 1)
                self._buf = self._buf[-keep:] if keep else bytearray()
                self.pos = 0
                return emitted_any

            if len(self._buf) - h < self.FRAME_SIZE:
                if h > 0:
                    self._buf = self._buf[h:]
                    self.pos = 0
                else:
                    self.pos = h
                return emitted_any

            


# ============================================================
# Streamer process (fast path: parse -> enqueue ONE tuple; writer thread updates SMM)
# ============================================================

class RfidStreamer(Process):
    def __init__(self, shared_memory_items, emager_kwargs: dict | None = None):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items
        self._stop_event = Event()
        self.e = None
        self.emager_kwargs = emager_kwargs or {}

        # cache shapes for ring writes
        self._shapes = {item[0]: item[1] for item in shared_memory_items if len(item) >= 2}

        # writer thread plumbing (created in run)
        self._q = None
        self._writer = None

    def run(self):
        # Create shared memory manager IN CHILD PROCESS
        self.smm = SharedMemoryManager()

        # Create shared memory variables
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        # Device
        bw = self.emager_kwargs
        baud = int(bw.get("baud_rate", 3000000))
        com_name = bw.get("com_name", None)
        vid_pid = bw.get("vid_pid", (12259, 256))

        self.e = RFID(baud_rate=baud, com_name=com_name, vid_pid=vid_pid)
        self.e.connect()
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
                    frame_id, rfid_block = self._q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    # EMG: store as uint16 (your shared memory is now uint16)
                    rfid = np.asarray(rfid_block, dtype=np.uint16)



                    # Write all three (still separate locks per tag, but all-or-nothing per frame at queue level)
                    buffer_write("rfid", rfid)


                except Exception:
                    pass
                finally:
                    self._q.task_done()

        self._writer = threading.Thread(target=writer_thread_fn, daemon=True)
        self._writer.start()

        # Frame handler: decode is already done in Emager3; this just enqueues ONE item
        self.drop_count = 0
        def on_frame(frame_id, rfid_block):
            try:
                self._q.put_nowait((int(frame_id), rfid_block))
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
