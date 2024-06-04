from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy import signal

from soapyrx import analyze

NFFT = 256
USER_SELECT = None

# Flag indicated that LaTeX fonts have been enabled.
LATEX_FONT_ENABLED = False

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

class SignalQuadPlot():
    """Quad plot of a signal.

    Plot the amplitude and the phase of a signal in both time and frequency
    domains. The phase is only plotted if the signal is complex. If the
    sampling rate is specified, synchronized time-domain and frequency-domain
    will be enabled.

    """
    # Signal variables.

    # Signal to plot [np.ndarray].
    sig = None
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
    plot_init_flag = False

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

    def __init__(self, sig, sr = None, fc = None):
        assert type(sig) == np.ndarray, "sig should be a numpy array (np.ndarray)!"
        self.sig = sig
        self.sr = sr
        self.fc = fc
        # Compute the duration of the signal if possible.
        if sr is not None:
            self.duration = len(sig) / sr
        # Compute the number of columns and rows depending on the signal type.
        self.nrows = 2
        self.ncols = 2 if analyze.is_iq(self.sig) else 1
        # Use a shared x-axis only if duration is available for time vector
        # creation.
        if self.duration is not None:
            self.sync = True
        # Example of creating filters that will be applied before plotting.
        # self.sos_filter_phase_time = signal.butter(1, 2e6, 'low', fs=self.sr, output='sos')

    def __plot_amp(self):
        """Plot the amplitude of the signal in time and frequency domains in a vertical way."""
        # Check needed parameters have been initialized.
        assert self.xlabel is not None
        # Compute the amplitude.
        sig = analyze.get_amplitude(self.sig)
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
        self.ax_ampl_freq.specgram(self.sig, NFFT=NFFT, Fs=self.sr, Fc=self.fc, sides="twosided", mode="magnitude")
        self.ax_ampl_time.set_ylabel("Amplitude [ADC value]")
        self.ax_ampl_freq.set_ylabel("Frequency [Hz]")

    def __plot_phase(self):
        """Plot the phase of the signal in time and frequency domains in a vertical way."""
        # Check needed parameters have been initialized.
        assert self.xlabel is not None
        # Compute phase rotation:
        sig = analyze.get_phase_rot(self.sig)
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
        self.ax_phase_freq.specgram(sig, NFFT=NFFT, Fs=self.sr, Fc=self.fc)
        self.ax_phase_time.set_ylabel("Phase rotation [Radian]")
        self.ax_phase_freq.set_ylabel("Frequency [Hz]")

    def __plot_init_labels(self):
        """Initialize the labels of the plot."""
        if self.sync is True:
            self.xlabel = "Time [s]"
        else:
            self.xlabel = "Sample [#]" if LATEX_FONT_ENABLED is False else "Sample [\\#]"

    def plot_init(self, title=None):
        """Initialize plot without showing or saving.

        :param title: If set to a string, use it as plot title.

        """
        assert self.plot_init_flag == False, "Plot initialized multiple times!"
        assert self.nrows == 2, "Bad nrows value"
        assert self.ncols == 1 or self.ncols == 2, "Bad ncols value"
        # Create the plot layout.
        sharex = "col" if self.sync is True else "none"
        if self.ncols == 1:
            self.fig, (self.ax_ampl_time, self.ax_ampl_freq) = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=sharex)
        elif self.ncols == 2:
            self.fig, ((self.ax_ampl_time, self.ax_phase_time), (self.ax_ampl_freq, self.ax_phase_freq)) = plt.subplots(nrows=self.nrows, ncols=self.ncols, sharex=sharex)
        # Generate a time vector if shared x-axis is required.
        if self.sync is True:
            self.t = np.linspace(0, self.duration, len(self.sig))
            assert len(self.t) == len(self.sig), "Bad length matching between time vector and signal!"
            # NOTE: For plt.specgram():
            # - If Fs is not set, it will generates an x-axis of "len(self.sig) / 2".
            # - If Fs is set, it will generates an x-axis of "duration * sampling rate".
        # Initialize the labels.
        self.__plot_init_labels()
        # Proceed to plots.
        self.__plot_amp()
        if self.ncols == 2:
            self.__plot_phase()
        # Add the title if needed.
        if title is not None and title != "":
            self.fig.suptitle(title)
        # Set the initialized flag.
        self.plot_init_flag = True
    
    def plot(self, block=True, save=None, title=None, show=True):
        """Plot the different components of a signal.

        :param block: If set to False, do not block the program execution while
        plotting.

        :param save: If set to a file path, use this to save the plot.

        :param title: If set to a string, use it as plot title.

        :param save: If set to True, show the interactive display.
        
        """
        # Initialize the plot if needed.
        if self.plot_init_flag is False:
            self.plot_init(title=title)
        # Enable tight_layout for larger plots.
        self.fig.set_tight_layout(True)
        # Show it and/or save it.
        if save is not None and save != "":
            figure = plt.gcf()
            figure.set_size_inches(32, 18)
            plt.savefig(save, bbox_inches='tight', dpi=300)
        if show is True:
            plt.show(block=block)
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
        self.axampl.plot(analyze.get_amplitude(self.signal[self.lb:self.ub]))
        self.axspec.specgram(self.signal[self.lb:self.ub], NFFT=NFFT, Fs=self.sr, Fc=self.fc, sides="twosided", mode="magnitude")
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
        self.axampl.plot(analyze.get_amplitude(self.signal))
        self.axampl.set_xlabel('Sample [#]')
        self.axampl.set_ylabel('Amplitude')
        self.axspec.specgram(self.signal, NFFT=NFFT, Fs=self.sr, Fc=self.fc, sides="twosided", mode="magnitude")
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
