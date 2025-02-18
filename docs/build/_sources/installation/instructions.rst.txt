Instructions
============

To install EELSfitter, all one need to do is open up a terminal and run

.. code-block::

   pip install EELSFitter

This installs all the missing dependencies and enables easy import statements in Python, for example:

.. code-block:: python

   import EELSFitter

   # example of loading spectral image .dm4 file
   path_to_image = "path_to_image"
   image = EELSFitter.SpectralImage.load_data(dm4_path)


Alternatively, one can choose to clone our public `github repository <https://github.com/LHCfitNikhef/EELSfitter>`_ instead:

.. code-block::

   git clone https://github.com/LHCfitNikhef/EELSfitter.git

