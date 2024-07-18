"""Plotting functions."""

# * Importation

# Standard import.
from os import path
from functools import partial

# External import.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.widgets import Button, Slider
from scipy import signal

# Internal import.
from soapyrx import dsp
from soapyrx import config

# * Global configuration

# Use a standard bright Matplotlib style.
plt.style.use('bmh')

# Number of bins for the FFT.
NFFT = 256

# Flag indicated that LaTeX fonts have been enabled.
LATEX_FONT_ENABLED = False

# Matplotlib prefered color map [viridis | inferno].
COLOR_MAP = "inferno"

# * Functions

def enable_latex_fonts():
    """Use LaTeX for text rendering."""
    global LATEX_FONT_ENABLED
    # Use pdflatex to generate fonts.
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "font.size": 15
    })
    LATEX_FONT_ENABLED = True

# * Classes

class SignalQuadPlot():
    """Quad plot of a signal.

    Plot the amplitude and the phase of a signal in both time and frequency
    domains. The phase is only plotted if the signal is complex. If the
    sampling rate is specified, synchronized time-domain and frequency-domain
    will be enabled.

    If data is directly passed to __init__, the plot will be only show/saved
    once. If function to get the data is passed to __init__, the plot will be
    updating in live.

    """
    # Signal variables.

    # Data of the signal to plot [np.ndarray].
    sigdata = None
    # Function to generate data of the signal to plot [partial].
    sigfunc = None
    # Sampling rate of the plotted signal [Msps].
    sr = None
    # Center frequency of the plotted signal [Hz].
    fc = None
    # Duration of the plotted signal [s].
    duration = None

    # Plotting flags.

    # If we sould use a shared x axis accross plot [bool].
    sync = False
    # Initialized flag for plot_init() [bool].
    init_flag = False

    # Plotting objects.

    # Figure [Matplotlib Figure].
    fig = None
    # Time vector used for shared x axis [np.ndarray].
    t = None
    # Amplitude time-domain axe [Matplotlib Axes].
    ax_ampl_time = None
    # Amplitude frequency-domain axe [Matplotlib Axes].
    ax_ampl_freq = None
    # Phase time-domain axe [Matplotlib Axes].
    ax_phase_time = None
    # Phase frequency-domain axe [Matplotlib Axes].
    ax_phase_freq = None

    # Plotting parameters.

    # Number of rows [integer].
    nrows = None
    # Number of columns [integer].
    ncols = None
    # x-axis labels for all plots.
    xlabel = None
    # SOS filter that will be applied to the time-domain amplitude signal before plotting.
    sos_filter_ampl_time = None
    # SOS filter that will be applied to the time-domain phase rotation signal before plotting.
    sos_filter_phase_time = None

    def __init__(self, sigdata, sigfunc = None, sr = None, fc = None):
        # Save passed parameters.
        self.sigdata = sigdata
        self.sigfunc = sigfunc
        self.sr = sr
        self.fc = fc
        # Use a shared x-axis only if duration is computable for time vector
        # creation, hence, if sample rate is passed.
        if self.sr is not None:
            self.sync = True
        # Compute the number of columns and rows.
        # NOTE: ncols can be set to 1 if signal is amplitude only.
        self.nrows = 2
        self.ncols = 2
        # NOTE: Example of creating filters that will be applied before plotting.
        # self.sos_filter_phase_time = signal.butter(1, 2e6, 'low', fs=self.sr, output='sos')

    def __plot_amp__(self):
        """Plot the amplitude of the signal in time and frequency domains in a vertical way."""
        # Check needed parameters have been initialized.
        assert self.xlabel is not None
        # Apply a pre-configured filter if enabled.
        if config.loaded() is True:
            sig = dsp.LHPFilter(config.get()["PLOTTERS"]["amp_filter_type"],
                                    config.get()["PLOTTERS"]["amp_filter_cutoff"],
                                   order=config.get()["PLOTTERS"]["amp_filter_order"],
                                   enabled=config.get()["PLOTTERS"]["amp_filter_en"]).apply(
                                       self.sigdata, self.sr, force_dtype=True
                                   )
        else:
            sig = self.sigdata
        # Compute the amplitude.
        sig = np.abs(sig)
        # Filter the signal for better visualization if requested.
        if self.sos_filter_ampl_time is not None:
            sig_filt = np.array(signal.sosfilt(self.sos_filter_ampl_time, sig), dtype=sig.dtype)
        else:
            sig_filt = sig
        if self.sync is True:
            self.ax_ampl_time.plot(self.t, sig_filt)
            self.ax_ampl_freq.set_xlabel(self.xlabel)
        else:
            self.ax_ampl_time.plot(sig_filt)
            self.ax_ampl_time.set_xlabel(self.xlabel)
            self.ax_ampl_freq.set_xlabel(self.xlabel)
        self.ax_ampl_freq.specgram(self.sigdata, NFFT=NFFT, Fs=self.sr, Fc=self.fc, sides="twosided", mode="magnitude", cmap=COLOR_MAP)
        self.ax_ampl_time.set_ylabel("Amplitude [ADC value]")
        self.ax_ampl_freq.set_ylabel("Frequency [Hz]")

    def __plot_phase__(self):
        """Plot the phase of the signal in time and frequency domains in a vertical way."""
        # Check needed parameters have been initialized.
        assert self.xlabel is not None
        # Apply a pre-configured filter if enabled.
        if config.loaded() is True:
            sig = dsp.LHPFilter(config.get()["PLOTTERS"]["phr_filter_type"],
                                    config.get()["PLOTTERS"]["phr_filter_cutoff"],
                                    order=config.get()["PLOTTERS"]["phr_filter_order"],
                                    enabled=config.get()["PLOTTERS"]["phr_filter_en"]).apply(
                                       self.sigdata, self.sr, force_dtype=True
                                   )
        else:
            sig = self.sigdata
        # Compute phase rotation:
        sig = dsp.phase_rot(sig)
        # Filter the signal for better visualization if requested.
        if self.sos_filter_phase_time is not None:
            sig_filt = np.array(signal.sosfilt(self.sos_filter_phase_time, sig), dtype=sig.dtype)
        else:
            sig_filt = sig
        if self.sync is True:
            self.ax_phase_time.plot(self.t, sig_filt)
            self.ax_phase_freq.set_xlabel(self.xlabel)
        else:
            self.ax_phase_time.plot(sig_filt)
            self.ax_phase_time.set_xlabel(self.xlabel)
            self.ax_phase_freq.set_xlabel(self.xlabel)
        self.ax_phase_freq.specgram(sig, NFFT=NFFT, Fs=self.sr, Fc=self.fc, cmap=COLOR_MAP)
        self.ax_phase_time.set_ylabel("Phase rotation [Radian]")
        self.ax_phase_freq.set_ylabel("Frequency [Hz]")

    def __animate__(self, i):
        """Matplotlib animation function wrapper.

        1. Clear the axes of the figure.
        2. Get the data to plot if needed.
        3. Update plot initialization from data information
        4. Plot the data.

        """
        self.ax_ampl_time.clear()
        self.ax_ampl_freq.clear()
        self.ax_phase_time.clear()
        self.ax_phase_freq.clear()
        if self.sigdata is None:
            self.__sigdata_get__()
        self.__init_from_sigdata__()
        self.__sigdata_plot__()

    def __sigdata_get__(self):
        """Get data to plot from predefined function.

        Call self.sigfunc() to get the signal stored in self.sigdata. Then,
        compute additional variables related to sigdata.

        """
        assert self.sigdata is None, "Signal is already waiting to be plotted!"
        self.sigdata = self.sigfunc()
        assert type(self.sigdata) == np.ndarray, "sig should be a numpy array (np.ndarray)!"

    def __init_from_sigdata__(self):
        """Initialize figure with information from sigdata."""
        # Compute the duration of the signal if possible.
        if self.sr is not None:
            self.duration = len(self.sigdata) / self.sr
        # Generate a time vector if shared x-axis is required.
        if self.sync is True:
            self.t = np.linspace(0, self.duration, len(self.sigdata))
            assert len(self.t) == len(self.sigdata), "Bad length matching between time vector and signal!"
            # NOTE: For plt.specgram():
            # - If Fs is not set, it will generates an x-axis of "len(self.sigdata) / 2".
            # - If Fs is set, it will generates an x-axis of "duration * sampling rate".

    def __sigdata_plot__(self):
        """Plot currently registered data.

        Data should have been registered using either __init__ or
        __sigdata_get__ through self.sigfunc(). After plotting, data will be
        deleted from the memory.

        """
        assert self.sigdata is not None, "No data has been registered in __init__ nor using self.sigfunc()!"
        # Plotting.
        self.__plot_amp__()
        if self.ncols == 2:
            self.__plot_phase__()
        # Deletion.
        del self.sigdata
        self.sigdata = None

    def init(self, title=None):
        """Initialize figure without plotting, showing or saving.

        :param title: If set to a string, use it as plot title.

        """
        assert self.init_flag == False, "Plot initialized multiple times!"
        assert self.nrows == 2, "Bad nrows value"
        assert self.ncols == 1 or self.ncols == 2, "Bad ncols value"
        # Create the plot layout.
        sharex = "col" if self.sync is True else "none"
        if self.ncols == 1:
            self.fig, (self.ax_ampl_time, self.ax_ampl_freq) = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=sharex)
        elif self.ncols == 2:
            self.fig, ((self.ax_ampl_time, self.ax_phase_time), (self.ax_ampl_freq, self.ax_phase_freq)) = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=sharex)
        # Initialize the labels.
        if self.sync is True:
            self.xlabel = "Time [s]"
        else:
            self.xlabel = "Sample [#]" if LATEX_FONT_ENABLED is False else "Sample [\\#]"
        # Add the title if needed.
        if title is not None and title != "":
            self.fig.suptitle(title)
        # Enable tight_layout for larger plots.
        self.fig.set_tight_layout(True)
        # Set the initialized flag.
        self.init_flag = True
    
    def plot(self, block=True, save=None, title=None, show=True):
        """Plot the different components of a signal.

        :param block: If set to False, do not block the program execution while
        plotting.

        :param save: If set to a file path, use this to save the plot.

        :param title: If set to a string, use it as plot title.

        :param save: If set to True, show the interactive display.
        
        """
        # Initialize the plot if needed.
        if self.init_flag is False:
            self.init(title=title)
        # If no function for getting data is provided, only plot/save once.
        if self.sigfunc is None:
            self.__animate__(0)
            # Show it and/or save it.
            if save is not None and save != "":
                figure = plt.gcf()
                figure.set_size_inches(32, 18)
                plt.savefig(save, bbox_inches='tight', dpi=300)
            if show is True:
                plt.show(block=block)
        # If function for getting data is provided, enter lvie update.
        else:
            # NOTE: "interval" seems to be limited by the time Matplotlib takes
            # to render the data.
            anim = animation.FuncAnimation(self.fig, self.__animate__, interval=500, cache_frame_data=False)
            plt.show()
        plt.clf()
 
class PlotShrink():
    """Plot signal and visually skrink it.

    Plot the amplitude of a signal in temporal and frequency domain and interactively shrink the signal.

    Typical use case:

    pltshrk = PlotShrink(signal)
    # The user will visually shrink the signal.
    pltshrk.plot()
    # Get the result.
    signal = pltshrk.get_signal()

    """
    # Plotted signal (numpy ND array of complex numbers).
    signal = None
    # Sampling rate of the plotted signal [Msps].
    sr = None
    # Center frequency of the plotted signal [Hz].
    fc = None
    # Matplotlib main figure (from plt.subplots()).
    fig = None
    # Matplotlib axis for temporal plot (from plt.subplots()).
    axampl = None
    # Matplotlib axis for frequential plot (from plt.subplots()).
    axspec = None
    # Integer defining the current lower bound of the signal.
    lb = None
    # Integer defining the current uper bound of the signal.
    ub = None

    def __init__(self, signal, sr = None, fc = None):
        """Initialize the signal and the bounds.

        The signal must be complex.

        """
        # Sanity-check.
        assert type(signal) == np.ndarray
        assert signal.dtype == np.complex64
        # Initialization.
        self.signal = signal
        self.sr = sr
        self.fc = fc
        self.lb = 0
        self.ub = len(signal)

    def update(self):
        """Clear plots and redraw them using current bounds."""
        # Clear the axis from last plots.
        self.axampl.clear()
        self.axspec.clear()
        # Plot the signal using new lower bound.
        self.axampl.plot(np.abs(self.signal[self.lb:self.ub]))
        self.axspec.specgram(self.signal[self.lb:self.ub], NFFT=NFFT, Fs=self.sr, Fc=self.fc, sides="twosided", mode="magnitude", cmap=COLOR_MAP)
        # Redraw the figure.
        self.fig.canvas.draw()

    def update_lb(self, val):
        """Update plots with new lower bound."""
        # Save new bound.
        self.lb = int(val)
        # Update plots.
        self.update()

    def update_ub(self, val):
        """Update plots with new upper bound."""
        # Save new bound.
        self.ub = int(val)
        # Update plots.
        self.update()
    
    def plot(self):
        """Start the plot for interactive shrink."""
        # Create the figure and plot the signal.
        self.fig, (self.axampl, self.axspec) = plt.subplots(nrows=2, ncols=1)
        self.axampl.plot(np.abs(self.signal))
        self.axampl.set_xlabel('Sample [#]')
        self.axampl.set_ylabel('Amplitude')
        self.axspec.specgram(self.signal, NFFT=NFFT, Fs=self.sr, Fc=self.fc, sides="twosided", mode="magnitude", cmap=COLOR_MAP)
        self.axspec.set_xlabel('Sample [#]')
        self.axspec.set_ylabel('Frequency [Hz]')

        # Adjust the main plot to make room for the sliders.
        self.fig.subplots_adjust(bottom=0.25)

        # Make two horizontal sliders.
        # Dimensions: [left, bottom, width, height]
        axlb = self.fig.add_axes([0.25, 0.14, 0.65, 0.03])
        axub = self.fig.add_axes([0.25, 0.07, 0.65, 0.03])
        lb_slider = Slider(
            ax=axlb,
            label='Lower bound index',
            valmin=0,
            valmax=len(self.signal),
            valinit=0,
        )
        ub_slider = Slider(
            ax=axub,
            label='Upper bound index',
            valmin=0,
            valmax=len(self.signal),
            valinit=len(self.signal),
        )

        # Register the update function with each slider.
        lb_slider.on_changed(self.update_lb)
        ub_slider.on_changed(self.update_ub)

        # Start interactive plot.
        plt.show()

    def get_signal(self):
        """Get the shrinked signal from the object."""
        return self.get_signal_from(self.signal)
    
    def get_signal_from(self, signal):
        """Get the skrinked signal from external signal but with current bounds."""
        return signal[self.lb:self.ub]
