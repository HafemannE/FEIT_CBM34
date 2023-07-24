"""
.. module:: useful_1
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Andrew Carter <andrew@invalid.com>


"""

def public_fn_with_googley_docstring(name, state=None):
    """This function does something.

    Args:
       name (str):  The name to use.

    Kwargs:
       state (bool): Current state to be in.

    Returns:
       int.  The return code::

          0 -- Success!
          1 -- No good.
          2 -- Try again.

    Raises:
       AttributeError, KeyError

    A really great idea.  A way you might use me is

    >>> print public_fn_with_googley_docstring(name='foo', state=None)
    0

    BTW, this always returns 0.  **NEVER** use with :class:`MyPublicClass`.

    """
    return 0

def public_fn_with_sphinxy_docstring(name, state=None):
    """This function does something.

    :param name: The name to use.
    :type name: str.
    :param state: Current state to be in.
    :type state: bool.
    :returns:  int -- the return code.
    :raises: AttributeError, KeyError

    """
    return 0

def public_fn_without_docstring():
    return True

def _private_fn_with_docstring(foo, bar='baz', foobarbas=None):
    """I have a docstring, but won't be imported if you just use ``:members:``.
    """
    return None


class MyPublicClass(object):
    """We use this as a public class example class.

    You never call this class before calling :func:`public_fn_with_sphinxy_docstring`.

    .. note::

       An example of intersphinx is this: you **cannot** use :mod:`pickle` on this class.

    """

    def __init__(self, foo, bar='baz'):
        """A really simple class.

        Args:
           foo (str): We all know what foo does.

        Kwargs:
           bar (str): Really, same as foo.

        """
        self._foo = foo
        self._bar = bar

    def get_foobar(self, foo, bar=True):
        """This gets the foobar

        This really should have a full function definition, but I am too lazy.

        >>> print get_foobar(10, 20)
        30
        >>> print get_foobar('a', 'b')
        ab

        Isn't that what you want?

        """
        return foo + bar

    def _get_baz(self, baz=None):
        """A private function to get baz.

        This really should have a full function definition, but I am too lazy.

        """
        return baz


"""This module illustrates how to write your docstring in OpenAlea
and other projects related to OpenAlea."""

__license__ = "Cecill-C"
__revision__ = " $Id: actor.py 1586 2009-01-30 15:56:25Z cokelaer $ "
__docformat__ = 'reStructuredText'


class MainClass1(object):
    """This class docstring shows how to use sphinx and rst syntax

    The first line is brief explanation, which may be completed with 
    a longer one. For instance to discuss about its methods. The only
    method here is :func:`function1`'s. The main idea is to document
    the class and methods's arguments with 

    - **parameters**, **types**, **return** and **return types**::

          :param arg1: description
          :param arg2: description
          :type arg1: type description
          :type arg1: type description
          :return: return description
          :rtype: the return type description

    - and to provide sections such as **Example** using the double commas syntax::

          :Example:

          followed by a blank line !

      which appears as follow:

      :Example:

      followed by a blank line

    - Finally special sections such as **See Also**, **Warnings**, **Notes**
      use the sphinx syntax (*paragraph directives*)::

          .. seealso:: blabla
          .. warnings also:: blabla
          .. note:: blabla
          .. todo:: blabla

    .. note::
        There are many other Info fields but they may be redundant:
            * param, parameter, arg, argument, key, keyword: Description of a
              parameter.
            * type: Type of a parameter.
            * raises, raise, except, exception: That (and when) a specific
              exception is raised.
            * var, ivar, cvar: Description of a variable.
            * returns, return: Description of the return value.
            * rtype: Return type.

    .. note::
        There are many other directives such as versionadded, versionchanged,
        rubric, centered, ... See the sphinx documentation for more details.

    Here below is the results of the :func:`function1` docstring.

    """

    def function1(self, arg1, arg2, arg3):
        """returns (arg1 / arg2) + arg3

        This is a longer explanation, which may include math with latex syntax
        :math:`\\alpha`.
        Then, you need to provide optional subsection in this order (just to be
        consistent and have a uniform documentation. Nothing prevent you to
        switch the order):

          - parameters using ``:param <name>: <description>``
          - type of the parameters ``:type <name>: <description>``
          - returns using ``:returns: <description>``
          - examples (doctest)
          - seealso using ``.. seealso:: text``
          - notes using ``.. note:: text``
          - warning using ``.. warning:: text``
          - todo ``.. todo:: text``

        **Advantages**:
         - Uses sphinx markups, which will certainly be improved in future
           version
         - Nice HTML output with the See Also, Note, Warnings directives


        **Drawbacks**:
         - Just looking at the docstring, the parameter, type and  return
           sections do not appear nicely

        :param arg1: the first value
        :param arg2: the first value
        :param arg3: the first value
        :type arg1: int, float,...
        :type arg2: int, float,...
        :type arg3: int, float,...
        :returns: arg1/arg2 +arg3
        :rtype: int, float

        :Example:

        >>> import template
        >>> a = template.MainClass1()
        >>> a.function1(1,1,1)
        2

        .. note:: can be useful to emphasize
            important feature
        .. seealso:: :class:`MainClass2`
        .. warning:: arg2 must be non-zero.
        .. todo:: check that arg2 is non zero.
        """
        return arg1/arg2 + arg3




if __name__ == "__main__":
    import doctest
    doctest.testmod()