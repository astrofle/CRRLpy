Examples
========

Line frequencies
----------------

To get the frequency of all the alpha transitions for carbon between 1.15 and 1.73 GHz (L-band at the Green Bank Telescope)

.. code-block:: python
   
   from crrlpy import crrls
   n,f = crrls.find_lines_sb([1150, 1730], "RRL_CIalpha")

Line broadening
---------------
   
To compute the line broadening due to collisions with ions as computed by `Salgado et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017ApJ...837..142S/abstract>`_

.. code-block:: python
   
   import numpy as np
   from crrlpy import crrls
   
   n = np.arange(100, 200, 1)
   df_p = crrls.pressure_broad_salgado(n, 100., 1)
   
The above will return the line broadening in Hz.
