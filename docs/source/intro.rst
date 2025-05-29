Pearson Distributions
=========================

For some distributions, it is easier to get their moments than their probability density
functions. However, probability densities have more applications than moments. Therefore,
it is desirable to construct the densities from the moments. Pearson distributions provide
an effective way for the density construction.

:abbr:`PDE(Partial Differential Equation)`
-------------------------------------------

The density :math:`p(x)` of a Pearson distribution satisfies the following
:abbr:`PDE(Partial Differential Equation)`:

.. math::

   \frac{dp(x)/dx}{p(x)} = -\frac{a + x}{\sum_{i=0}^n c_i x^i},

where parameters :math:`a`, :math:`c_0,\cdots,c_n` are determined by the moments of the
distribution.

Moments
---------

The number of moments, employed to determining the coefficients, increases along with the
number :math:`n`:

- :math:`n = 2`: requires the first four moments,

- :math:`n = 3`: requires the first six moments,

- :math:`n = 4`: requires the first eight moments,

- :math:`n = 5`: requires the first ten moments,

so on and so forth.

Usually, matching the first eight moments would produce an accurate enough density
approximation. Therefore, I implement the Pearson distribution for this case within the
`pearsondist` Python pacakge.





