# encoding: utf-8


"""
A numpy.ndarray with axis names

Access as mpnum.named_ndarray.
"""


import numpy as np


class named_ndarray(object):

    """Associate names to the axes of a ndarray.

    :property axisnames: The names of the axes.

    All methods which return arrays return named_ndarray instances.

    :method axispos(axisname): Return the position of the named axis
    :method rename(translate): Rename axes
    :method conj(): Return the complex conjugate array
    :method to_array(name_order): Return a ndarray with axis order
        specified by name_order.
    :method tensordot(other, axes): numpy.tensordot() with axis names
        instead of axis indices

    """

    def __init__(self, array, axisnames):
        """
        :param numpy.ndarray array: A numpy.ndarray instance
        :param axisnames: A iterable with a name for each axis
        """
        assert(len(array.shape) == len(axisnames)), \
            'number of names does not match number of dimensions'
        assert len(axisnames) == len(set(axisnames)), \
            'axisnames contains duplicates: {}'.format(axisnames)
        self._array = array
        self._axisnames = tuple(axisnames)

    def axispos(self, axisname):
        """Return the position of an axis.
        """
        return self._axisnames.index(axisname)

    def rename(self, translate):
        """Rename axes.

        An error will be raised if the resulting list of names
        contains duplicates.

        :param translate: List of (old_name, new_name) axis name pairs.

        """
        new_names = list(self._axisnames)
        for oldname, newname in translate:
            new_names[self.axispos(oldname)] = newname
        return named_ndarray(self._array, new_names)

    def conj(self):
        """Complex conjugate as named_ndarray.
        """
        return named_ndarray(self._array.conj(), self._axisnames)

    def to_array(self, name_order):
        """Convert to a normal ndarray with given axes ordering.

        :param name_order: Order of axes in the array
        """
        name_pos = [self.axispos(name) for name in name_order]
        array = self._array.transpose(name_pos)
        return array

    def tensordot(self, other, axes):
        """Compute tensor dot product along named axes.

        An error will be raised if the remaining axes of self and
        other contain duplicate names.

        :param other: Another named_ndarray instance
        :param axes: List of axis name pairs (self_name, other_name)
            to be contracted
        :returns: Result as named_ndarray
        """
        axes_self = [names[0] for names in axes]
        axes_other = [names[1] for names in axes]
        axespos_self = [self.axispos(name) for name in axes_self]
        axespos_other = [other.axispos(name) for name in axes_other]
        new_names = [name for name in self._axisnames if name not in axes_self]
        new_names += (name for name in other._axisnames if name not in axes_other)
        array = np.tensordot(self._array, other._array, (axespos_self, axespos_other))
        return named_ndarray(array, new_names)

    @property
    def axisnames(self):
        """The names of the array"""
        return self._axisnames
