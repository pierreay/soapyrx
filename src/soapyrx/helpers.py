"""Helpers functions wrapping classes."""

import numpy as np

from soapyrx import lib as soapysdr_lib
from soapyrx import plotters
from soapyrx import logger as l

def record(freq, samp, duration, gain, save_sig, save_plot, plot_flag, cut_flag, dir):
    """Helper for recording functions.

    Return the recorded signal.

    """
    # Radio block.
    try:
        with soapysdr_lib.SoapyRadio(fs=samp, freq=freq, idx=0, duration=duration, dir=dir, gain=gain) as rad:
            # Initialize the driver.
            rad.open()
            # Perform the recording.
            rad.record()
            # Save the radio capture in temporary buffer.
            rad.accept()
            # Return the radio capture.
            return rad.get_signal()
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
