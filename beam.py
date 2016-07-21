"""

This is a lightweight set of functions and classes for beam propagation in Python.

It's inspired by Johannes Courtial's "WaveTrace" library for LabVIEW, and does scalar propagation of complex beams.
Maybe if it proves useful I'll refine it and give it its own repo - for now, this is it.

(c) Richard Bowman 2016, released under GNU GPL v3 or later

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
import matplotlib.pyplot as plt

class Beam(np.ndarray):
    """Beams are represented as ndarrays with some extra properties.

    This is the main data-storage class we use.

    TODO: make this play nicely with nplab's data storage stuff.
    """

    attributes_to_copy = [  # attributes listed here will be copied/preserved
        "wavelength",  # wavelength of the beam
        "dx",  # spacing between pixels, in real-space beams
        "dk",  # spacing between pixels, in reciprocal-space beams
    ]

    def __new__(cls, input_array, attrs=None, **kwargs):
        """Make a new beam based on a numpy array."""
        # the input array should be a numpy array, then we cast it to this type
        obj = np.asarray(input_array).view(cls)
        # next, add the attributes, copying from keyword args if possible.
        for attr in cls.attributes_to_copy:
            setattr(obj, attr, kwargs.get(attr, None))
        if attrs is not None:  # copy metadata from another object
            obj.copy_attrs(attrs)
        return obj

    def __array_finalize__(self, obj):
        # this is called by numpy when the object is created (__new__ may or
        # may not get called)
        if obj is None: return  # if obj is None, __new__ was called - do nothing
        # if we didn't create the object with __new__,  we must add the attrs.
        # We copy these from the source object if possible or create
        # new ones and set to None if not.
        for attr in self.attributes_to_copy:
            setattr(self, attr, getattr(obj, attr, None))

    def copy_attrs(self, obj, exclude=[]):
        """Copy the non-array data from another beam.

        obj: the object to copy from
        exclude: a list of attribute names not to copy

        NB this will only copy attributes that are named in
        the class as ones to copy.
        """
        for attr in self.attributes_to_copy:
            if attr not in exclude:
                setattr(self, attr, getattr(obj, attr, None))

    @property
    def k(self):
        """Magnitude of the wavevector"""
        return 2 * np.pi / self.wavelength

    @property
    def kx(self):
        """Return the x component of the wavevector for each column"""
        dim = 0 if self.ndim == 1 else -2
        fractions = fftshift(fftfreq(self.shape[dim]))
        return (fractions * self.dk * self.shape[dim])[:, np.newaxis]

    @property
    def ky(self):
        """Return the y component of the wavevector for each row"""
        assert self.ndim > 1, "This property needs 2D beams"
        fractions = fftshift(fftfreq(self.shape[-1]))
        return (fractions * self.dk * self.shape[-1])[np.newaxis, :]

    @property
    def kz(self):
        """Z component of the wavevector for each pixel."""
        return np.sqrt(self.k ** 2 - self.kx ** 2 - self.ky ** 2)

    @property
    def x(self):
        """Return the x position for each column"""
        N = self.shape[0 if self.ndim == 1 else -2]
        return (np.linspace(-(N - 1) / 2.0, (N - 1) / 2.0, N) * self.dx)[:, np.newaxis]

    @property
    def y(self):
        """Return the y position for each row"""
        assert self.ndim > 1, "This property needs 2D beams"
        N = self.shape[-1]
        return (np.linspace(-(N - 1) / 2.0, (N - 1) / 2.0, N) * self.dx)[np.newaxis, :]

def tophat_beam(N, dx, wavelength, r):
    """Create a top-hat beam"""
    beam = Beam(np.zeros((N,N), dtype=np.complex),
                dx=dx,
                wavelength=wavelength)
    beam[beam.x**2+beam.y**2 < r**2] = 1
    return beam

def gaussian_beam(N, dx, wavelength, r):
    """Create a Gaussian beam with 1/e radius r"""
    beam = Beam(np.zeros((N,N), dtype=np.complex),
                dx=dx,
                wavelength=wavelength)
    # the [:,:] means we just replace the data, we
    # don't make a whole new object
    beam[:,:] = np.exp(-(beam.x**2 + beam.y**2)/(2 * r**2))
    return beam

def FFT(beam):
    """Take the FFT of a beam"""
    beam_ft = Beam(fftshift(fftn(beam)), attrs=beam)
    beam_ft.dk = 2*np.pi/(beam.dx*beam.shape[0])
    beam_ft.dx = None # Fourier-space beams shouldn't have dx
    return beam_ft

def IFFT(beam_ft):
    """Take the IFFT of a beam"""
    beam = Beam(ifftn(ifftshift(beam_ft)), attrs=beam_ft)
    beam.dx = 2*np.pi/(beam_ft.dk*beam_ft.shape[0])
    beam.dk = None # Real-space beams shouldn't have dk
    return beam


def propagate_incremental(beam, dz, N=1):
    """Propagate a beam by phase-shifting its FT by kz*dz

    The propagation is performed N times, and the beam is
    returned as a 3D array, where the first index is the
    propagation number.  The returned array will have size
    N+1, as the original beam is included.

    In the future this should include absorbing edges."""
    # Make a new beam to store the result
    propagation = Beam(
        np.empty((N + 1,) + beam.shape, dtype=np.complex),
        attrs=beam)
    propagation[0, :, :] = beam
    propagator = np.exp(1j * FFT(beam).kz * dz)
    for i in range(N):
        beam_ft = FFT(propagation[i, :, :])
        beam_ft *= propagator
        propagation[i + 1, :, :] = IFFT(beam_ft)
    return propagation


def propagate_fast(beam, dz):
    """Quickly propagate a beam by a given dz (in one step).

    This corresponds to a single iteration of
    propagate_incremental."""
    beam_ft = FFT(beam)
    return IFFT(beam_ft * np.exp(1j * beam_ft.kz * dz))


def phase_and_intensity_image(beam, vmin=0, vmax=None):
    """Return an RGB array where brightness=intensity and hue=phase.

    vmin and vmax set the max and min intensity values.
    """
    if vmax is None:
        vmax = float(np.max(np.abs(beam) ** 2))
    normalised_phase = (np.angle(beam) + np.pi) / (2 * np.pi / 3)
    # we use a 3-segment colour map: cyan -> magenta -> yellow -> cyan
    colours = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]).astype(np.float)
    segment = np.floor(normalised_phase).astype(np.int)  # 1, 2, or 3
    remainder = normalised_phase - segment
    # find the starting (A) and ending (B) colours for each pixel
    A = np.array([colours[:, i][segment] for i in range(3)])
    B = np.array([colours[:, i][segment + 1] for i in range(3)])
    hued_image = A * (1 - remainder) + B * remainder
    return (255.99 * hued_image.transpose(1, 2, 0)
            * ((np.abs(beam) ** 2 - vmin) / (vmax - vmin))[:, :, np.newaxis]
            ).astype(np.uint8)

def show_beam(beam, axes=None, length_units=1000, **kwargs):
    """display a beam, by plotting a phase/intensity image.

    beam: the beam to be plotted
    axes: a matplotlib axes object to plot in (or None)
    length_units: divisor for the X/Y axes (e.g. 1000 gives mm)
    Extra keyword arguments are passed to axes.imshow()
    """
    if axes is None:
        axes = plt
    plot_args = {"aspect": 1}
    try:
        u = length_units
        plot_args['extent'] = (np.min(beam.x) / u, np.max(beam.x) / u,
                               np.min(beam.y) / u, np.max(beam.y) / u)
    except:
        pass  # ignore errors if we're in units of k
    plot_args.update(kwargs)
    axes.imshow(phase_and_intensity_image(beam), **plot_args)

