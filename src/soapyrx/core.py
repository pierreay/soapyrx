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
import sys
import signal
from enum import Enum

# External import.
import numpy as np
import SoapySDR

# Internal import.
from soapyrx import logger as l
from soapyrx import config

# * Global variables

# Path of the FIFO files used between SoapyServer and SoapyClient.
# Client -> FIFO -> Server
PATH_FIFO_C2S_CMD = "/tmp/c2s-cmd.fifo"
PATH_FIFO_C2S_DATA = "/tmp/c2s-data.fifo"
# Server -> FIFO -> Client
PATH_FIFO_S2C_CMD = "/tmp/s2c-cmd.fifo"
PATH_FIFO_S2C_DATA = "/tmp/s2c-data.fifo"

# Polling interval for a while True loop , i.e. sleeping time, i.e. interval to
# check whether a command is queued in the FIFO or not. High enough to not
# consume too much CPU (here, 5%) but small enough to not introduce noticeable
# delay to the recording.
POLLING_INTERVAL = 1e-6

# * Enumerations

# Models supported by SoapyRadio class.
SoapyRadioModel = Enum('SoapyRadioModel', ['GENERIC', 'HACKRF', 'USRP', 'SDRPLAY'])

# * Classes

class SoapyServer():
    # If set to True, the server main loop will stop. [bool]
    should_stop = False
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __sig_handler__(self, signum, frame):
        self.stop()

    def __init__(self):
        l.LOGGER.debug("[{}] Initialization...".format(type(self).__name__))
        # List of registered SDRs.
        self.sdrs = []
        # List of registered SDRs' indexes.
        # NOTE: The IDXs must be unique, as the IDX is used as filename
        # identifier and as SoapySDR's result index.
        self.registered_idx = []
        signal.signal(signal.SIGINT, self.__sig_handler__)
        signal.signal(signal.SIGTERM, self.__sig_handler__)

    def __ack__(self, cmd):
        """Acknoledge the end of the command CMD."""
        with open(PATH_FIFO_S2C_CMD, "w") as fifo:
            ack = "ack:{}".format(cmd)
            l.LOGGER.debug("[{}] Opened FIFO (w): {}".format(type(self).__name__, PATH_FIFO_S2C_CMD))
            fifo.write(ack)
            # NOTE: Help but not enough to prevent a bug where two
            # successive ack message are read concatenated in one single
            # read from the client:
            fifo.write("")
            l.LOGGER.debug("[{}] FIFO <- {}".format(type(self).__name__, ack))    

    def __send__(self, data):
        """Send DATA to the client FIFO."""
        with open(PATH_FIFO_S2C_DATA, "wb") as fifo:
            l.LOGGER.debug("[{}] Opened FIFO (wb): {}".format(type(self).__name__, PATH_FIFO_S2C_CMD))
            fifo.write(data)
            l.LOGGER.debug("[{}] FIFO <- {} bytes".format(type(self).__name__, sys.getsizeof(data)))    

    def register(self, sdr):
        l.LOGGER.debug("[{}] Register: idx={}".format(type(self).__name__, sdr.idx))
        # Check if SDR is not already initialized.
        if sdr.idx in self.registered_idx:
            raise Exception("Same SDR is registered twice!")
        # Proceed to the registration.
        self.sdrs.append(sdr)
        self.registered_idx.append(sdr.idx)

    def open(self):
        l.LOGGER.debug("[{}] Open all SDRs...".format(type(self).__name__))
        for sdr in self.sdrs:
            sdr.open()

    def close(self):
        l.LOGGER.debug("[{}] Close all SDRs...".format(type(self).__name__))
        for sdr in self.sdrs:
            sdr.close()
        if path.exists(PATH_FIFO_C2S_CMD):
            os.remove(PATH_FIFO_C2S_CMD)
        if path.exists(PATH_FIFO_S2C_CMD):
            os.remove(PATH_FIFO_S2C_CMD)
        if path.exists(PATH_FIFO_C2S_DATA):
            os.remove(PATH_FIFO_C2S_DATA)
        if path.exists(PATH_FIFO_S2C_DATA):
            os.remove(PATH_FIFO_S2C_DATA)

    def record_start(self):
        """Asynchronous version of record().

        Start recording for a pre-configured amount of time.

        """
        l.LOGGER.debug("[{}] Start recording threads for all SDRs...".format(type(self).__name__))
        # Use multi-threaded implementation.
        self.thr = [None] * len(self.sdrs)
        for idx, sdr in enumerate(self.sdrs):
            self.thr[idx] = Thread(target=sdr.record, args=(None,))
            self.thr[idx].start()
        l.LOGGER.debug("[{}] Recording threads started for all SDRs!".format(type(self).__name__))

    def record_stop(self):
        """Asynchronous version of record().

        Wait recording after threads stopped.

        """
        l.LOGGER.debug("[{}] Wait recording threads for all SDRs...".format(type(self).__name__))
        for idx, sdr in enumerate(self.sdrs):
            self.thr[idx].join()
        l.LOGGER.debug("[{}] Recording threads finished for all SDRs!".format(type(self).__name__))

    def record(self, duration = None):
        """Perform a recording of DURATION seconds.

        Spawn a thread for each radio and start recording. Block until all
        recordings finished and all threads join.

        """
        # Use multi-threaded implementation.
        if len(self.sdrs) > 1:
            l.LOGGER.debug("[{}] Start recording threads for all SDRs...".format(type(self).__name__))
            thr = [None] * len(self.sdrs)
            for idx, sdr in enumerate(self.sdrs):
                thr[idx] = Thread(target=sdr.record, args=(duration,))
                thr[idx].start()
            l.LOGGER.debug("[{}] Wait recording threads for all SDRs...".format(type(self).__name__))
            for idx, sdr in enumerate(self.sdrs):
                thr[idx].join()
            l.LOGGER.debug("[{}] Recording threads finished for all SDRs!".format(type(self).__name__))
        # Don't use multi-threading if only one recording is needed.
        elif len(self.sdrs) == 1:
            self.sdrs[0].record(duration)

    def accept(self):
        l.LOGGER.debug("[{}] Accept recording for all SDRs...".format(type(self).__name__))
        for sdr in self.sdrs:
            sdr.accept()

    def reinit(self):
        l.LOGGER.debug("[{}] Reinitialize all SDRs...".format(type(self).__name__))
        for sdr in self.sdrs:
            sdr.reinit()

    def disable(self):
        for sdr in self.sdrs:
            sdr.disable()

    def start(self):
        """Start the server mode.

        This command will create a FIFO and listen for commands on it. The
        SoapyClient class can be instantiated in another process to
        communicate with this server.

        """
        def __create_fifo():
            """Create the named pipes (FIFOs)."""
            # Remove previously created FIFO.
            try:
                os.remove(PATH_FIFO_C2S_CMD)
                os.remove(PATH_FIFO_S2C_CMD)
                os.remove(PATH_FIFO_C2S_DATA)
                os.remove(PATH_FIFO_S2C_DATA)
            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    raise e
            # Create the named pipes (FIFOs).
            try:
                os.mkfifo(PATH_FIFO_C2S_CMD)
                os.mkfifo(PATH_FIFO_S2C_CMD)
                os.mkfifo(PATH_FIFO_C2S_DATA)
                os.mkfifo(PATH_FIFO_S2C_DATA)
            except OSError as oe:
                raise

        # Available commands on server-side.
        cmds = {"record":       self.record,
                "accept":       self.accept,
                "reinit":       self.reinit,
                "disable":      self.disable,
                "record_start": self.record_start,
                "record_stop":  self.record_stop,
                "get":          self.get}

        # Create the FIFO.
        __create_fifo()
        # NOTE: Put logging before FIFO is opened before its wait for a client
        # to return.
        l.LOGGER.info("[{}] Server started!".format(type(self).__name__))
        # Open the FIFO.
        with open(PATH_FIFO_C2S_CMD, "r") as fifo:
            l.LOGGER.debug("[{}] Opened FIFO (r): {}".format(type(self).__name__, PATH_FIFO_C2S_CMD))
            # Infinitely listen for commands and execute the radio commands accordingly.
            while self.should_stop is False:
                cmd = fifo.read()
                if len(cmd) > 0:
                    l.LOGGER.debug("[{}] FIFO -> {}".format(type(self).__name__, cmd))
                    # Execute the received command and acknowledge its execution.
                    if cmd in cmds:
                        cmds[cmd]()
                        self.__ack__(cmd)
                    elif cmd == "stop":
                        break
                # Smart polling.
                sleep(POLLING_INTERVAL)
            l.LOGGER.info("[{}] Server shutdown!".format(type(self).__name__))

    def stop(self):
        self.should_stop = True

    def get_nb(self):
        """Get the number of currently registed SDRs."""
        return len(self.sdrs)

    def get(self, idx = 0):
        """Send the receveid signal of radio indexed by IDX on client FIFO."""
        sig = self.sdrs[idx].get()
        self.__send__(sig)
        l.LOGGER.debug("[{}] FIFO <- signal: len={}".format(type(self).__name__, len(sig)))
        l.LOGGER.debug("[{}] FIFO <- signal: head={}".format(type(self).__name__, sig[:3]))
        l.LOGGER.debug("[{}] FIFO <- signal: tail={}".format(type(self).__name__, sig[-3:]))

class SoapyRadio():
    """SoapySDR controlled radio.

    Typical workflow:

    1. Initialize: __init__() -> open()

    2. Records: [ record() -> accept() ] ... -> get()

    3. Records: [ record() -> accept() ] ... -> get()

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

    # Model of current SDR [SoapyRadioModel]
    model = None
    # AGC enable flag [bool].
    agc = None
    # Gain [dB].
    gain = None
    # RX SoapySDR stream.
    rx_stream = None
    # RX temporary buffer allocated at runtime.
    rx_buff = None

    # * Context manager functions.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # * Static functions.

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

    def __init__(self, fs, freq, idx = 0, enabled = True, duration = 1, agc = True, gain = None):
        l.LOGGER.debug("[{}:{}] Initialization: fs={} freq={} enabled={} duration={}".format(type(self).__name__, idx, fs, freq, enabled, duration))
        # NOTE: Automatically convert floats to integers (allows using scentific notation, e.g. e6 or e9).
        self.fs = int(fs)
        self.freq = int(freq)
        self.agc = agc
        if gain is not None:
            self.gain = int(gain)
        self.idx = idx
        self.enabled = enabled
        # Default duration if nothing is specified during self.record().
        self.duration = duration
        # Recording acceptation flag.
        self.accepted = False
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
            # Find radio type.
            self.sdr = SoapySDR.Device(results[idx])
            if "HackRF" in str(self.sdr):
                self.model = SoapyRadioModel.HACKRF
            elif "b200" in str(self.sdr):
                self.model = SoapyRadioModel.USRP
            elif "SDRplay" in str(self.sdr):
                self.model = SoapyRadioModel.SDRPLAY
            else:
                self.model = SoapyRadioModel.GENERIC
            # Initialize the radio with requested parameters.
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, fs)
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
            self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "TX/RX")
            self._setup_gain(agc=agc, gain=gain)
            # Initialize the RX buffer with a sufficient size to hold the
            # default duration.
            self._rx_buff_init(self._rx_buff_len_exp_auto(self.duration * self.fs))

    def _setup_gain(self, agc = True, gain = None):
        """Setup gain settings (AGC and absolute value).

        By default, enable the automatic gain control (AGC) and do not setup an
        absolute gain value. Specific gain setup (multiple amplifier) can be
        configured through the configuration file.

        :param agc: Boolean enabling or disable the AGC.
        :param gain: Integer setting absolute gain vluae [db].

        """
        assert agc == True or agc == False, "AGC should be enabled or disabled!"
        assert gain is None or gain >= 0, "Gain should be positive!"
        # Setup automatic gain control (AGC).
        self.sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, agc)
        l.LOGGER.debug("gainMode={}".format(self.sdr.getGainMode(SoapySDR.SOAPY_SDR_RX, 0)))
        if gain is not None:
            # Setup gain value.
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            l.LOGGER.debug("gain={}".format(self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)))
        # SDRPlay-specific:
        if self.model == SoapyRadioModel.SDRPLAY and config.loaded() is True:
            if config.get()[SoapyRadioModel.SDRPLAY.name]["gain_ifgr"] >= 0:
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR", config.get()[SoapyRadioModel.SDRPLAY.name]["gain_ifgr"])
                l.LOGGER.debug("gain_ifgr={}".format(self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, "IFGR")))
            if config.get()[SoapyRadioModel.SDRPLAY.name]["gain_rfgr"] >= 0:
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR", config.get()[SoapyRadioModel.SDRPLAY.name]["gain_rfgr"])
                l.LOGGER.debug("gain_rfgr={}".format(self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, "RFGR")))

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
        l.LOGGER.debug("[{}:{}] Allocate memory for RX buffer: 2^{} dtype-elements...".format(type(self).__name__, self.idx, rx_buff_len_exp))
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
            l.LOGGER.debug("[{}:{}] Setup and activate SoapySDR stream".format(type(self).__name__, self.idx))
            # From SoapySDR/include/SoapySDR/Device.h:
            # - "CF32" - complex float32 (8 bytes per element)
            # - "CS16" - complex int16   (4 bytes per element)
            # From SoapyUHD/SoapyUHDDevice.cpp/getNativeStreamFormat():
            # UHD and the hardware use "CS16" format in the underlying transport layer.
            self.rx_stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            ret = self.sdr.activateStream(self.rx_stream)
            if ret != 0:
                raise Exception("Driver error, cannot activate SoapySDR stream: {}".format(ret))
            # Initialize for first recordings.
            self.reinit()

    def close(self):
        if self.rx_stream is not None:
            l.LOGGER.debug("[{}:{}] Deactivate and close SoapySDR stream".format(type(self).__name__, self.idx))
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)

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
                l.LOGGER.info("[{}:{}] Start recording: {:.2}s...".format(type(self).__name__, self.idx, duration))
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
            l.LOGGER.debug("[{}:{}] Accept recording!".format(type(self).__name__, self.idx))
            self.accepted = True
            self.rx_signal = np.concatenate((self.rx_signal, self.rx_signal_candidate))

    def reinit(self):
        """Re-initialize the recording state and buffers such that a new
        recording can occur."""
        l.LOGGER.debug("[{}:{}] Reinitialization!".format(type(self).__name__, self.idx))
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
        l.LOGGER.info("[{}:{}] Disabling!".format(type(self).__name__, self.idx))
        self.enabled = False

    def get(self):
        """Return the receveid signal.

        The returned signal will be I/Q represented using np.complex64 numbers.

        """
        assert self.rx_signal.dtype == np.complex64, "Signal should be complex numbers!"
        return self.rx_signal

class SoapyClient():
    """Control a SoapyServer object living in another process.

    This class implements a command sending mechanism through a named pipe
    (FIFO) mechanism allowing to control another process that initialized the
    radio at startup. It allows to perform multiple radio recordings without
    re-initializing the radio's driver while using different client processes.

    """
    # Time sleeped to simulate wait a SoapySDR server's command return [s].
    STUB_WAIT = 0.5

    def __init__(self, enabled = True):
        l.LOGGER.debug("[{}] Initialization: enabled={}".format(type(self).__name__, enabled))
        self.enabled = enabled

    def __cmd__(self, cmd):
        """Send the command CMD through the FIFO."""
        # NOTE: The only way I found to reliably send the commands individually
        # is to open/close/sleep for each commands. Otherwise, the commands
        # arrived concatenated at the reader process.
        if self.enabled is True:
            l.LOGGER.debug("[{}] FIFO <- {}".format(type(self).__name__, cmd))
            with open(PATH_FIFO_C2S_CMD, "w") as fifo:
                fifo.write(cmd)
            sleep(0.1)

    def __wait__(self, cmd):
        """Wait for the command CMD to complete."""
        if self.enabled is True:
            time_start = time()
            l.LOGGER.debug("[{}] Waiting: {}".format(type(self).__name__, cmd))
            with open(PATH_FIFO_S2C_CMD, "r") as fifo:
                l.LOGGER.debug("[{}] Opened FIFO (r): {}".format(type(self).__name__, PATH_FIFO_S2C_CMD))
                while True:
                    ack = fifo.read()
                    if len(ack) > 0:
                        l.LOGGER.debug("[{}] FIFO -> {}".format(type(self).__name__, ack))
                        if ack == "ack:{}".format(cmd):
                            l.LOGGER.debug("[{}] Wait completed!".format(type(self).__name__))
                            break
                    sleep(POLLING_INTERVAL)
                    # Timeout for waiting a command.
                    if (time() - time_start) > 3: # [s]
                        raise Exception("Timeout exceeded!")
        else:
            l.LOGGER.debug("[{}] Waiting stub because disabled: sleep {}s".format(type(self).__name__, self.STUB_WAIT))
            sleep(self.STUB_WAIT)

    def __recv__(self):
        """Receive data from the server FIFO."""
        with open(PATH_FIFO_S2C_DATA, "rb") as fifo:
            l.LOGGER.debug("[{}] Opened FIFO (rb): {}".format(type(self).__name__, PATH_FIFO_S2C_DATA))
            while True:
                data = fifo.read()
                if len(data) > 0:
                    l.LOGGER.debug("[{}] FIFO -> {} bytes".format(type(self).__name__, sys.getsizeof(data)))
                    return data
                sleep(POLLING_INTERVAL)

    def record(self):
        self.__cmd__("record")
        self.__wait__("record")

    def record_start(self):
        self.__cmd__("record_start")
        # NOTE: Don't need to wait because record_start is fast enough.

    def record_stop(self):
        self.__cmd__("record_stop")
        self.__wait__("record_stop")

    def accept(self):
        self.__cmd__("accept")
        self.__wait__("accept")

    def reinit(self):
        self.__cmd__("reinit")
        self.__wait__("reinit")

    def get(self):
        self.__cmd__("get")
        sig = np.frombuffer(self.__recv__(), dtype=np.complex64)
        l.LOGGER.debug("[{}:0] FIFO -> signal: len={}".format(type(self).__name__, len(sig)))
        l.LOGGER.debug("[{}:0] FIFO -> signal: head={}".format(type(self).__name__, sig[:3]))
        l.LOGGER.debug("[{}:0] FIFO -> signal: tail={}".format(type(self).__name__, sig[-3:]))
        self.__wait__("get")
        return sig

    def disable(self):
        self.__cmd__("disable")

    def stop(self):
        self.__cmd__("stop")
