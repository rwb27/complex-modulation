"""

This is a collection of methods to turn complex holograms into real-valued phase holograms.

A flattener function should take a complex Beam as an argument, and return a real-valued one with the same
dimensions in pixels.

(c) Richard Bowman 2016, released under GNU GPL v3.0 or later

"""

from beam import Beam
import numpy as np


def nearest(hologram):
    """Ignore amplitude information and just return phase"""
    return Beam(np.angle(hologram), attrs=hologram)


def normalised_amplitude(hologram):
    """Ensure the maximum amplitude in a hologram is 1"""
    amplitude = np.abs(hologram)
    amplitude /= np.max(amplitude)  # do it in-place for efficiency
    return amplitude


def linear_blaze(hologram):
    """Multiply phase with amplitude."""
    return nearest(hologram) * normalised_amplitude(hologram)


def arcsinc(y):
    """Slow way of calculating inverse sinc of y

    NB y must be between 0 and 1, and may be an ndarray :)
    """
    assert 0 <= np.min(y) and np.max(y) <= 1, "y must be between 0 and 1"
    x = np.pi / 2
    dx = np.pi / 4
    for i in range(11):
        x += dx * np.sign(np.sinc(x) - y)
        dx /= 2
    return x


def arcsinc_blaze(hologram):
    """Multiply phase with the inverse sinc of the amplitude ("""
    return nearest(hologram) * (1 - arcsinc(normalised_amplitude(hologram)) / np.pi)


def binary_noise(hologram):
    """Add binary phase noise to dissipate unwanted light"""
    # Generate random +/-1 array:
    random_sign = np.sign(np.random.choice([-1.0, 1.0], size=hologram.shape))
    return nearest(hologram) + random_sign * np.acos(normalised_amplitude(hologram))


def random_noise_powerlaw(p):
    """Generate a flattener that draws the noise from a probability distribution.

    The distribution makes the probability proportional to the invers of the squared distance
    between the chosen value and the desired one, raised to some power.  NB for high
    powers, we reduce this to the binary noise method, but much slower."""
    def random_noise(hologram):
        """Add phase noise to dissipate unwanted light.

        The probability of adding phase noise p to the hologram is given by the
        inverse of the squared distance between the complext value we'd get with e^ip and
        the desired complex value.
        """
        #  p=1/((cos(p)-A)^2+sin(p)^2)
        iterating = np.ones(hologram.shape, dtype=bool)
        phase = nearest(hologram)
        amplitude = normalised_amplitude(hologram)
        noisyphase = phase.copy()
        iterating[amplitude == 1] = False # Don't add noise where it's not needed
        # NB the less noise that's needed, the more samples we'll need to find a
        # good value; the amplitude==1 pixels may iterate forever...
        while(np.any(iterating)):
            # pick a new phase for the pixels that are still going
            noisyphase[iterating] = (np.random.random(hologram.shape)-0.5)*2*np.pi
            # calculate the probability distribution value for that position
            # NB I don't know if this is normalised!  Should really do that...
            # That said, it's a moot point so long as p<1; then we're just using
            # up more random numbers, but the normalisation is OK.
            p = (1 + amplitude**2 - 2*amplitude*np.cos(noisyphase-phase))**(-p)/2*np.pi
            # accept that value with probability p
            iterating &= np.random.random(hologram.shape) > p
        return noisyphase
    return random_noise

random_noise = random_noise_powerlaw(1)


def higher_order(order):
    """Add noise that's a harmonic of the hologram's phase, to send noise to higher orders.

    `order` is the order to dump into.  NB `order=1` reduces to the "nearest" algorithm.
    """
    def higher_order_n(hologram):
        """Dump power into the %dth order""" % order
        signs = np.sign(np.real(hologram ** order))
        return nearest(hologram) + signs * np.acos(normalised_amplitude(hologram))
    return higher_order_n


def chessboard_binary_noise(hologram):
    """Dump power into a binary chessboard grating"""
    signs = np.zeros(hologram.shape, dtype=np.float)
    signs[0::2, 1::2] = 1
    signs[1::2, 0::2] = 1
    # signs is a chessboard grating
    return nearest(hologram) + signs * np.acos(normalised_amplitude(hologram))

def binary_grating_noise(k):
    """TODO: pick signs to make a binary grating with pitch k"""
    raise NotImplementedError

