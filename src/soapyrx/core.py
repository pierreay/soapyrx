"""SDR classes using SoapySDR as backend.

Allows to use a single or multiple SDRs in parallel using threads.

"""

# * Importation

# Standard import.
import time
from time import sleep, time
import os
from os import path
import errno
from threading import Thread

# External import.
import numpy as np
import SoapySDR

# Internal import.
from soapyrx import logger as l

# * Global variables

# Path of the FIFO files used between SoapyServer and SoapyClient.
# Client -> FIFO -> Server
PATH_FIFO_C2S = "/tmp/soapysdr.fifo"
# Server -> FIFO -> Client
PATH_FIFO_S2C = "/tmp/soapysdr_client.fifo"

# Polling interval for a while True loop , i.e. sleeping time, i.e. interval to
# check whether a command is queued in the FIFO or not. High enough to not
# consume too much CPU (here, 5%) but small enough to not introduce noticeable
# delay to the recording.
POLLING_INTERVAL = 1e-6

# * Classes

class SoapyServer():

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __init__(self):
        l.LOGGER.debug("SoapyServer.__init__()")
        # List of registered SDRs.
        self.sdrs = []
        # List of registered SDRs' indexes.
        # NOTE: The IDXs must be unique, as the IDX is used as filename
        # identifier and as SoapySDR's result index.
        self.registered_idx = []

    def register(self, sdr):
        l.LOGGER.debug("SoapyServer.register(idx={})".format(sdr.idx))
        # Check if SDR is not already initialized.
        if sdr.idx in self.registered_idx:
            raise Exception("The same SDR is registered twice!")
        # Proceed to the registration.
        self.sdrs.append(sdr)
        self.registered_idx.append(sdr.idx)

    def open(self):
        l.LOGGER.debug("SoapyServer.open()")
        for sdr in self.sdrs:
            sdr.open()

    def close(self):
        l.LOGGER.debug("SoapyServer.close()")
        for sdr in self.sdrs:
            sdr.close()
        if path.exists(PATH_FIFO_C2S):
            # Delete the FIFO.
            os.remove(PATH_FIFO_C2S)
        if path.exists(PATH_FIFO_S2C):
            # Delete the FIFO.
            os.remove(PATH_FIFO_S2C)

    def record_start(self):
        """Asynchronous version of record().

        Start recording for a pre-configured amount of time.

        """
        l.LOGGER.debug("SoapyServer.record_start().enter")
        # Use multi-threaded implementation.
        l.LOGGER.debug("Start recording threads...")
        self.thr = [None] * len(self.sdrs)
        for idx, sdr in enumerate(self.sdrs):
            self.thr[idx] = Thread(target=sdr.record, args=(None,))
            self.thr[idx].start()
        l.LOGGER.debug("SoapyServer.record_start().exit")

    def record_stop(self):
        """Asynchronous version of record().

        Wait recording after threads stopped.

        """
        l.LOGGER.debug("SoapyServer.record_stop().enter")
        l.LOGGER.debug("Wait recording threads...")
        for idx, sdr in enumerate(self.sdrs):
            self.thr[idx].join()
        l.LOGGER.debug("SoapyServer.record_stop().exit")

    def record(self, duration = None):
        """Perform a recording of DURATION seconds.

        Spawn a thread for each radio and start recording. Block until all
        recordings finished and all threads join.

        """
        l.LOGGER.debug("SoapyServer.record(duration={}).enter".format(duration))
        # Use multi-threaded implementation.
        if len(self.sdrs) > 1:
            # XXX: Should we switch to processes instead of threads to counter
            # the GIL? But maybe it can be slower to spawn?
            l.LOGGER.debug("Start threads for multiple SDRs recording...")
            thr = [None] * len(self.sdrs)
            for idx, sdr in enumerate(self.sdrs):
                thr[idx] = Thread(target=sdr.record, args=(duration,))
                thr[idx].start()
            l.LOGGER.debug("Wait threads for multiple SDRs recording...")
            for idx, sdr in enumerate(self.sdrs):
                thr[idx].join()
        # Don't use multi-threading if only one recording is needed.
        elif len(self.sdrs) == 1:
            self.sdrs[0].record(duration)
        l.LOGGER.debug("SoapyServer.record(duration={}).exit".format(duration))

    def accept(self):
        l.LOGGER.debug("SoapyServer.accept()")
        for sdr in self.sdrs:
            sdr.accept()

    def save(self, dir = None, reinit = True):
        l.LOGGER.debug("SoapyServer.save(dir={})".format(dir))
        for sdr in self.sdrs:
            sdr.save(dir, reinit=reinit)

    def disable(self):
        for sdr in self.sdrs:
            sdr.disable()

    def start(self):
        """Start the server mode.

        This command will create a FIFO and listen for commands on it. The
        SoapyClient class can be instantiated in another process to
        communicate with this server.

        """
        def __ack__(cmd):
            """Acknoledge the end of the command execution by opening-closing
            the FIFO in W mode.

            """
            with open(PATH_FIFO_S2C, "w") as fifo_w:
                ack = "ack:{}".format(cmd)
                l.LOGGER.debug("[server] Opened FIFO_CLIENT at {}".format(PATH_FIFO_S2C))
                fifo_w.write(ack)
                # NOTE: Help but not enough to prevent a bug where two
                # successive ack message are read concatenated in one single
                # read from the client:
                fifo_w.write("")
                l.LOGGER.debug("[server] FIFO_CLIENT <- {}".format(ack))

        def __create_fifo():
            """Create the named pipe (FIFO)."""
            # Remove previously created FIFO.
            try:
                os.remove(PATH_FIFO_C2S)
                os.remove(PATH_FIFO_S2C)
            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    raise e
            # Create the named pipe (FIFO).
            try:
                os.mkfifo(PATH_FIFO_C2S)
                os.mkfifo(PATH_FIFO_S2C)
            except OSError as oe:
                raise
                # if oe.errno != errno.EEXIST:
                #     raise

        # Create the FIFO.
        __create_fifo()
        # Open the FIFO.
        l.LOGGER.info("SDR process #{} ready for listening!".format(os.getpid()))
        with open(PATH_FIFO_C2S, "r") as fifo:
            l.LOGGER.debug("[server] Opened FIFO at {}".format(PATH_FIFO_C2S))
            # Infinitely listen for commands and execute the radio commands accordingly.
            while True:
                cmd = fifo.read()
                if len(cmd) > 0:
                    l.LOGGER.debug("[server] FIFO -> {}".format(cmd))
                    # Available commands on server-side.
                    cmds = {"record": self.record, "accept": self.accept, "save": self.save, "disable": self.disable, "record_start": self.record_start, "record_stop": self.record_stop}
                    # Execute the received command and acknowledge its execution.
                    if cmd in cmds:
                        cmds[cmd]()
                        __ack__(cmd)
                    elif cmd == "quit":
                        l.LOGGER.info("Quit the listening mode!")
                        break
                # Smart polling.
                sleep(POLLING_INTERVAL)

    def get_nb(self):
        """Get the number of currently registed SDRs."""
        return len(self.sdrs)

    def get_signal(self, idx):
        """Return the receveid signal of radio indexed by IDX."""
        return self.sdrs[idx].get_signal()

class SoapyRadio():
    """SoapySDR controlled radio.

    Typical workflow:

    1. Initialize: __init__() -> open()

    2. Records: [ record() -> accept() ] ... -> save()

    3. Records: [ record() -> accept() ] ... -> save()

    4. Deinitialize: close()

    """
    # * Constants.
    
    # Default length (power of 2) of RX temporary buffer. This length
    # corresponds to the number of samples.
    RX_BUFF_LEN_EXP = 20
    # Lower bound of RX temporary buffer. Using dtype of a tuple of uint16, a
    # length of 2^20 ~= 4 MB.
    RX_BUFF_LEN_EXP_LB = 20
    # Upper bound of RX temporary buffer. 2^28 ~= 1 GB.
    RX_BUFF_LEN_EXP_UB = 28

    # * Variables.

    # RX temporary buffer allocated at runtime.
    rx_buff = None

    # * Context manager functions.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # * Static functions.

    def __init__(self, fs, freq, idx = 0, enabled = True, duration = 1, dir = "/tmp", gain = 76):
        l.LOGGER.debug("SoapyRadio.__init__(fs={},freq={},idx={},enabled={},duration={},dir={},gain={})".format(fs, freq, idx, enabled, duration, dir, gain))
        assert gain >= 0, "Gain should be positive!"
        # NOTE: Automatically convert floats to integers (allows using scentific notation, e.g. e6 or e9).
        self.fs = int(fs)
        self.freq = int(freq)
        self.gain = int(gain)
        self.idx = idx
        self.enabled = enabled
        # Default duration if nothing is specified during self.record().
        self.duration = duration
        # Default directory if nothing is specified during self.save().
        self.dir = dir
        # Recording acceptation flag.
        self.accepted = False # Set to True by accept() and to False by save().
        # Recording buffers.
        self.rx_signal = None
        self.rx_signal_candidate = None
        # Long operations.
        if self.enabled:
            # Initialize the SDR driver.
            results = SoapySDR.Device.enumerate()
            # Check result of device detection and requested index.
            if len(results) == 0:
                raise Exception("SoapySDR didn't detected any device!")
            if idx > len(results):
                raise Exception("SoapySDR didn't detected the requested radio index!")
            # Initialize the radio with requested parameters.
            self.sdr = SoapySDR.Device(results[idx])
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, fs)
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "TX/RX")
            # Initialize the RX buffer with a sufficient size to hold the
            # default duration.
            self._rx_buff_init(self._rx_buff_len_exp_auto(self.duration * self.fs))

    @staticmethod
    def _rx_buff_len_exp_auto(nsamples):
        """Compute an automatic RX buffer length.

        :param nsamples: The number of samples that the SDR is configured to
        record.

        :return: The according rx_buff_len_exp value.

        """
        candidate = int(np.log2(nsamples)) + 1
        candidate = min(candidate, SoapyRadio.RX_BUFF_LEN_EXP_UB)
        candidate = max(candidate, SoapyRadio.RX_BUFF_LEN_EXP_LB)
        return candidate

    def _rx_buff_init(self, rx_buff_len_exp = RX_BUFF_LEN_EXP):
        """Initialize the RX buffer.

        Allocate memory for the RX buffer based on parameters. This allocation
        can take up to a few seconds for GB of allocation, hence we try to only
        allocate once.

        :param rx_buff_len_exp: Exponent used for the power of 2 defining
        memory allocation chunk size.

        """
        assert self.rx_buff is None
        assert type(rx_buff_len_exp) == int, "Length of RX buffer should be an integer!"
        assert rx_buff_len_exp <= self.RX_BUFF_LEN_EXP_UB, "Bad RX buffer exponent value!"
        assert rx_buff_len_exp >= self.RX_BUFF_LEN_EXP_LB, "Bad RX buffer exponent value!"
        l.LOGGER.debug("Allocate 2^{} dtype-elements in memory for RX buffer...".format(rx_buff_len_exp))
        self.rx_buff = np.array([0] * pow(2, rx_buff_len_exp), np.complex64)

    def _rx_buff_deinit(self):
        """Deinitialize the RX buffer.

        Deallocate memory for the RX buffer and set it to None.

        """
        assert self.rx_buff is not None
        del self.rx_buff
        self.rx_buff = None

    def rx_buff_config(self, rx_buff_len_exp):
        """Reconfigure the RX buffer size.

        It will de-initialize the previsouly initialized RX buffer and
        re-initializing it with the new size.

        :param rx_buffer_len_exp: Length used in `_rx_buff_init()'.

        """
        assert self.rx_buff is not None
        self._rx_buff_deinit()
        self._rx_buff_init(rx_buff_len_exp)

    def open(self):
        # Initialize the SoapySDR streams.
        if self.enabled:
            l.LOGGER.debug("[{}:{}] Setup and active SoapySDR stream".format(type(self).__name__, self.idx))
            # From SoapySDR/include/SoapySDR/Device.h:
            # - "CF32" - complex float32 (8 bytes per element)
            # - "CS16" - complex int16   (4 bytes per element)
            # From SoapyUHD/SoapyUHDDevice.cpp/getNativeStreamFormat():
            # UHD and the hardware use "CS16" format in the underlying transport layer.
            self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream)
            # Initialize for first recordings.
            self.reinit()

    def close(self):
        if self.rx_stream is not None:
            l.LOGGER.debug("SoapyRadio(idx={}).close().enter".format(self.idx))
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
            l.LOGGER.debug("SoapyRadio(idx={}).close().leave".format(self.idx))

    def record(self, duration = None, log = True):
        # Choose default duration configured during __init__ if None is given.
        if duration is None:
            duration = self.duration
        if self.enabled:
            # Initialize the buffer (0-length) that will contains the final
            # recorded signal from this function.
            self.rx_signal_candidate = np.array([0], np.complex64)
            # Number of samples requested to read.
            samples = int(duration * self.fs)
            # Number of samples that can fit in the RX buffer.
            rx_buff_len = len(self.rx_buff)
            if log is True:
                l.LOGGER.info("[{}:{}] Start recording during {:.2}s...".format(type(self).__name__, self.idx, duration))
            while len(self.rx_signal_candidate) < samples:
                # Number of samples that the readStream() function will try to
                # read from the SDR. It is equal to the minimum between: 1)
                # Number of samples needed to fullfil our buffer with the
                # requested number of samples. 2) Size of RX buffer. If the
                # requested number of samples is higher than the size of the RX
                # buffer, the RX buffer will be re-used after saving the first
                # bunch of samples into the `self.rx_signal_candidate'
                # variable.
                readStream_len = min(samples - len(self.rx_signal_candidate), rx_buff_len)
                assert readStream_len <= len(self.rx_buff)
                l.LOGGER.debug("[{}:{}] Read SoapySDR stream...".format(type(self).__name__, self.idx))
                sr = self.sdr.readStream(self.rx_stream, [self.rx_buff], readStream_len, timeoutUs=int(1e7))
                l.LOGGER.debug("[{}:{}] Read stream: ret:{} flags:{:b}".format(type(self).__name__, self.idx, sr.ret, sr.flags))
                # Recording at requested size, e.g., using USRP.
                if sr.ret == readStream_len and sr.flags == 1 << 2:
                    self.rx_signal_candidate = np.concatenate((self.rx_signal_candidate, self.rx_buff[:readStream_len]))
                # Recording smaller than requested, e.g., using HackRF (smaller buffer).
                elif sr.ret > 0 and sr.flags == 0:
                    self.rx_signal_candidate = np.concatenate((self.rx_signal_candidate, self.rx_buff[:sr.ret]))
            if log is True:
                l.LOGGER.info("[{}:{}] Recording finished!".format(type(self).__name__, self.idx, duration))
        else:
            time.sleep(duration)

    def accept(self):
        if self.enabled:
            l.LOGGER.debug("SoapyRadio(idx={}).accept()".format(self.idx))
            self.accepted = True
            self.rx_signal = np.concatenate((self.rx_signal, self.rx_signal_candidate))

    def save(self, dir = None, reinit = True):
        """Save the last accepted recording on disk.

        The saved .npy file will use the np.complex64 data type.

        :param reinit: If set to False, do not re-initialize the radio for a
        next recording. SoapyRadio.reinit() should be called manually later.

        """
        if dir is None:
            dir = self.dir
        if self.enabled is True and self.accepted is True:
            dir = path.expanduser(dir)
            l.LOGGER.info("save recording of radio #{} into directory {}".format(self.idx, dir))
            assert(path.exists(dir))
            np.save(path.join(dir, "raw_{}_{}.npy".format(self.idx, 0)), self.rx_signal)
            # Re-initialize for further recordings if requested [default].
            if reinit is True:
                self.reinit()

    def reinit(self):
        """Re-initialize the recording state and buffers such that a new
        recording can occur."""
        l.LOGGER.debug("re-initialization")
        self.accepted = False
        # Delete the signals since buffers can be large.
        if self.rx_signal is not None:
            del self.rx_signal
        self.rx_signal = np.array([0], np.complex64)
        if self.rx_signal_candidate is not None:
            del self.rx_signal_candidate
        self.rx_signal_candidate = None

    def disable(self):
        """Disable the radio."""
        l.LOGGER.info("disable radio #{}".format(self.idx))
        self.enabled = False

    def get_signal(self):
        """Return the receveid signal.

        The returned signal will be I/Q represented using np.complex64 numbers.

        """
        sig = self.rx_signal
        assert sig.dtype == np.complex64, "Signal should be complex numbers!"
        return sig

class SoapyClient():
    """Control a SoapyServer object living in another process.

    This class implements a command sending mechanism through a named pipe
    (FIFO) mechanism allowing to control another process that initialized the
    radio at startup. It allows to perform multiple radio recordings without
    re-initializing the radio's driver while using different Python process
    because of different calls from Bash.

    """
    # Time sleeped to simulate wait a SoapySDR server's command return [s].
    STUB_WAIT = 0.5

    def __init__(self, enabled = True):
        l.LOGGER.info("Initialize a SoapySDR client... (enabled={})".format(enabled))
        self.enabled = enabled

    def __cmd__(self, cmd):
        """Send a command through the FIFO."""
        # NOTE: The only way I found to reliably send the commands individually
        # is to open/close/sleep for each commands. Otherwise, the commands
        # arrived concatenated at the reader process.
        if self.enabled is True:
            l.LOGGER.debug("[client] FIFO <- {}".format(cmd))
            with open(PATH_FIFO_C2S, "w") as fifo:
                fifo.write(cmd)
            sleep(0.1)

    def __wait__(self, cmd):
        """Wait for the previous command to complete."""
        if self.enabled is True:
            time_start = time()
            l.LOGGER.debug("[client] Waiting for: {}".format(cmd))
            with open(PATH_FIFO_S2C, "r") as fifo:
                l.LOGGER.debug("[client] Opened FIFO_CLIENT at {}".format(PATH_FIFO_S2C))
                while True:
                    ack = fifo.read()
                    if len(ack) > 0:
                        l.LOGGER.debug("[client] FIFO_CLIENT -> {}".format(ack))
                        if ack == "ack:{}".format(cmd):
                            l.LOGGER.debug("[client] Wait completed!")
                            break
                    sleep(POLLING_INTERVAL)
                    # Timeout for waiting a command.
                    if (time() - time_start) > 3: # [s]
                        raise Exception("Timeout exceeded!")
        else:
            l.LOGGER.debug("Waiting stub for disabled SoapySDR client by sleeping {}s".format(self.STUB_WAIT))
            sleep(self.STUB_WAIT)

    def record(self):
        """Call the SoapyServer.record() method through the FIFO. Wait for the
        command to complete.

        """
        self.__cmd__("record")
        self.__wait__("record")

    def record_start(self):
        self.__cmd__("record_start")
        # NOTE: Don't need to wait because record_start is fast enough.

    def record_stop(self):
        self.__cmd__("record_stop")
        self.__wait__("record_stop")

    def accept(self):
        """Call the SoapyServer.accept() method through the FIFO. Returns
        immediately."""
        self.__cmd__("accept")

    def save(self):
        """Call the SoapyServer.save() method through the FIFO. Returns
        immediately."""
        self.__cmd__("save")
        self.__wait__("save")

    def disable(self):
        """Call the SoapyServer.disable() method through the FIFO. Returns
        immediately."""
        self.__cmd__("disable")

    def quit(self):
        """Send instruction to quit the radio listening in server mode. Returns
        immediately.

        """
        self.__cmd__("quit")
