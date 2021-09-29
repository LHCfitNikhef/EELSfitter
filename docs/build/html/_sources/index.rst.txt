.. EELSfitter documentation master file, created by
   sphinx-quickstart on Fri Mar 19 16:49:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: ../_assets/nikhef_logo.png
     :width: 18 %
     :alt: Nikhef logo
.. image:: ../_assets/kavli_logo.png
     :width: 18 %
     :alt: Kavli logo
.. image:: ../_assets/delft_logo.jpg
     :width: 18 %
     :alt: TU Delft logo
.. image:: ../_assets/vu_logo.png
     :width: 18 %
     :alt: VU logo
.. image:: ../_assets/qn_logo.png
     :width: 18 %
     :alt: QN logo

Welcome to the EELSfitter website!
=======================================

EELSfiter is an open-source Python-based framework developed for the analysis and interpretation of Electron Energy Loss Spectroscopy (EELS) measurements in Transmission Electron Microscopy (TEM). EELSfitter is based on the machine learning techniques developed by the NNPDF Collaboration in the context of applications in high energy physics, in particular feed-forward neural networks for unbiased regression in multidimensional problems.

.. figure:: ../_assets/ws2_bandgap_energy.png
    :width: 90%
    :class: align-center
    :figwidth: 90%
    :figclass: align-center

    *Spatially-resolved map of the bandgap for the* :math:`WS_2` *nanoflower specimen, where a mask has been applied to remove the vacuum and pure substrate pixels.*


Project description
~~~~~~~~~~~~~~~~~~~

EELSfitter has been used in the following **scientific publications**:

- *Charting the low-loss region in Electron Energy Loss Spectroscopy with machine learning*, Roest, Laurien I. and van Heijst, Sabrya E. and Maduro, Luigi and Rojo, Juan and Conesa-Boj, Sonia :cite:p:`Roest:2020kqy`.
- *Illuminating the Electronic Properties of WS2 Polytypism with Electron Microscopy*, van Heijst, Sabrya E. and Mukai, Masaki and Okunishi, Eiji and Hashiguchi, Hiroki and Roest, Laurien I. and Maduro, Luigi and Rojo, Juan and Conesa-Boj, Sonia :cite:p:`WS2_nanoflowers`.
- *Spatially-resolved bandgap and dielectric function in 2D materials from Electron Energy Loss Spectroscopy*, Brokkelkamp, Abel, and ter Hoeve, Jaco, and Postmes, Isabel, and van Heijst, Sabrya E. and Maduro, Luigi, and Davydov, Albert, and Krylyuk, Sergiy, and Rojo, Juan and Conesa-Boj, Sonia, *in preparation*.

The code from the these publications is available in the :ref:`Code<code>` section.

Team description
~~~~~~~~~~~~~~~~

The **EELSfitter collaboration** is currently composed by the following members:

- Abel Brokkelkamp, *Kavli Institiute of Nanoscience, Delft University of Technology*
- Jaco ter Hoeve, *VU Amsterdam and Nikhef Theory Group*
- Isabel Postmes, *Kavli Institiute of Nanoscience, Delft University of Technology*
- Luigi Maduro, *Kavli Institiute of Nanoscience, Delft University of Technology*
- Juan Rojo, *VU Amsterdam and Nikhef Theory Group*
- Sonia Conesa Boj, *Kavli Institiute of Nanoscience, Delft University of Technology*

Former members of the EELSfitter project include

- Laurien Roest, now data scientist at *Picnic*

.. toctree::
   :maxdepth: 1
   :caption: Theory
   :hidden:

   theory/clustering_pooling
   theory/nn_training
   theory/kk_analysis
   theory/bandgap_analysis

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :hidden:

   installation/instructions
   installation/tutorial

.. toctree::
   :maxdepth: 2
   :caption: Code
   :hidden:

   modules/EELSFitter.rst

.. toctree::
   :maxdepth: 1
   :caption: Bibliography
   :hidden:

   bibliography.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


