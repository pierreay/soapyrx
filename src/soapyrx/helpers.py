"""Helpers functions wrapping classes or doing small computations."""

# * Importation

# Standard import.
from os import path

# External import.
import numpy as np
import SoapySDR

# Internal import.
from soapyrx import core
from soapyrx import plotters
from soapyrx import logger as l

# * Functions for command-line interface

def discover():
    """Helper for discovering SDRs."""
    results = SoapySDR.Device.enumerate()
    for idx, result in enumerate(results):
        l.LOGGER.info("{}".format(result))
        l.LOGGER.info("Index: {}".format(idx))
        sdr = SoapySDR.Device(result)
        # Query device info.
        l.LOGGER.info("Antennas: {}".format(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)))
        l.LOGGER.info("Gains: {}".format(sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)))
        for gainType in sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0):
            l.LOGGER.info("Gain range [{}]: {}".format(gainType, sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0, gainType)))
        freqRanges = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
        for freqRange in freqRanges:
            l.LOGGER.info("Frenquency range: {}".format(freqRange))
    if not results:
        l.LOGGER.error("No detected SDR!"); exit(1)

def record(freq, samp, duration, agc, gain):
    """Helper for recording functions.

    Return the recorded signal.

    """
    # Radio block.
    try:
        with core.SoapyRadio(fs=samp, freq=freq, idx=0, duration=duration, agc=agc, gain=gain) as rad:
            # Initialize the driver.
            rad.open()
            # Perform the recording.
            rad.record()
            # Save the radio capture in temporary buffer.
            rad.accept()
            sig = rad.get()
            # Reinit the radio in case it is reused.
            rad.reinit()
            # Return the radio capture.
            return sig
    except Exception as e:
        l.LOGGER.critical("Error during radio recording!")
        raise e    

def plot(sig, samp=None, freq=None, cut_flag=False, plot_flag=True, save_sig="", save_plot="", title=None):
    """Helper for plotting functions."""
    # Simple visualization and processing block.
    try:
        # Cut the signal as requested.
        if cut_flag is True:
            pltshrk = plotters.PlotShrink(sig, sr=samp, fc=freq)
            pltshrk.plot()
            sig = pltshrk.get_signal_from(sig)
        # Plot the signal as requested.
        if plot_flag:
            plotters.SignalQuadPlot(sig, sr=samp, fc=freq).plot(save=save_plot, title=title)
        # Save the signal as requested.
        if save_sig != "":
            l.LOGGER.info("Save recording: {}".format(save_sig))
            np.save(save_sig, sig)
    except Exception as e:
        l.LOGGER.critical("Error during signal processing!")
        raise e

def server_start(idx, freq, samp_rate, duration, agc, gain):
    """Helper for starting a server."""
    # Initialize the radio individually.
    try:
        rad = core.SoapyRadio(fs=samp_rate, freq=freq, idx=idx, duration=duration, agc=agc, gain=gain)
    except Exception as e:
        l.LOGGER.critical("Error during radio initialization!")
        raise e
    # Create a server.
    with core.SoapyServer() as server:
        # Add the radio.
        server.register(rad)
        # Initialize the driver.
        server.open()
        # Listen for commands from another process.
        server.start()

def client():
    """Helper for recording using a client.

    Return the recorded signal.

    """
    # Initialize the client.
    client = core.SoapyClient()
    # Perform the recording.
    client.record()
    # Save the radio capture in temporary buffer.
    client.accept()
    sig = client.get()
    # Reinit the radio in case it is reused.
    client.reinit()
    # Return the radio capture.
    return sig

# * Functions for computations

def phase_rot(trace):
    """Get the phase rotation of one or multiple traces."""
    dtype_in = np.complex64
    dtype_out = np.float32
    assert type(trace) == np.ndarray
    assert trace.dtype == dtype_in
    if trace.ndim == 1:
        # NOTE: Phase rotation from expe/240201/56msps.py without filter:
        # Compute unwraped (remove modulos) instantaneous phase.
        trace = np.unwrap(np.angle(trace))
        # Set the signal relative to 0.
        trace = [trace[i] - trace[0] for i in range(len(trace))]
        # Compute the phase rotation of instantenous phase.
        # NOTE: Manually add first [0] sample.
        trace = [0] + [trace[i] - trace[i - 1] for i in range(1, len(trace), 1)]
        # Convert back to np.ndarray.
        trace = np.array(trace, dtype=dtype_out)
        assert trace.dtype == dtype_out
        return trace
    elif trace.ndim == 2:
        trace_rot = np.empty_like(trace, dtype=dtype_out)
        for ti, tv in enumerate(trace):
            trace_rot[ti] = get_phase_rot(tv)
        return trace_rot
