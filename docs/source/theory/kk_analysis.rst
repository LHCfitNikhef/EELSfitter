Kramers-Kronig analysis of EEL spectra
======================================


Here we provide an overview of the theoretical formalism adopted in this 
work :cite:p:`Egerton2007` to evaluate the single-scattering distribution,
local thickness, band gap energy and type and complex dielectric function from  EELS spectra.
Electron energy loss spectra  measured experimentally are composed by three contributions:
the one from inelastic single scatterings off the electrons in the specimen, 
the one  associated to multiple inelastic scatterings, and then the Zero-Loss Peak (ZLP) 
arising from both elastic scatterings and from instrumental broadening.
Hence, a given spectrum :math:`I_{\rm EELS}(E)` can be decomposed as

.. math:: :label: eq:EELSmaster

    I_{\rm EELS}(E) = I_{\rm ZLP}(E) + I_{\rm inel}(E) =
    I_{\rm ZLP}(E) + \sum_{n=1}^\infty
    I^{(n)}_{\rm inel}(E) \, ,


where :math:`E` is the energy loss experienced by the electrons upon traversing the specimen [#f1]_,
:math:`I_{\rm ZLP}(E)` is the ZLP contribution, and :math:`I^{(n)}_{\rm inel}(E)` indicates
the contribution from :math:`n` inelastic scatterings. The ZLP intensity can be further
expressed as

.. math:: :label: eq:ZLP_norm

    I_{\rm ZLP}(E) = N_0 R(E) \, ,\qquad \int_{-\infty}^\infty dE \,R(E)=1 \, ,


where :math:`R(E)` is known as the (normalised) resolution or instrumental response function,
whose full width at half-maximum (FWHM) indicates the resolution of the instrument.
The normalisation factor :math:`N_0` thus corresponds to the integrated intensity under 
the zero-loss peak. In the following we assume that the ZLP intensity :math:`I_{\rm ZLP}(E)`
is known and can be reliably subtracted from :math:`I_{\rm EELS}(E)` in order to isolate
the inelastic scattering contribution :math:`I_{\rm inel}(E)`. In EELSfitter, the ZLP
subtraction is achieved using a machine learning method developed in :cite:p:`Roest2021`
and extended to spectral images.


The single-scattering distribution
----------------------------------


If one denotes by :math:`t` the local thickness of the specimen and by :math:`\lambda` the 
mean free path of the electrons, then assuming that inelastic scatterings are 
uncorrelated it follows that the integral over the :math:`n`-scatterings distribution
:math:`I^{(n)}_{\rm inel}` follow a Poisson distribution [#f2]_ 

.. math:: :label: eq:N_n

    N_n \equiv \int_{-\infty}^{\infty} dE\, I^{(n)}_{\rm inel}(E) =
    B \frac{\left( t/\lambda \right)^n}{n!}e^{-t/\lambda} \, , \qquad n=1,2,\ldots,\infty \, ,


with :math:`B` a normalisation constant. From the combination of Eqns. :eq:`eq:EELSmaster`
and :eq:`eq:N_n` it follows that

.. math:: :label: eq:norm_inelastic

    N_{\rm inel}\equiv \int_{-\infty}^{\infty} dE\,
    I_{\rm inel}(E) = \sum_{n=1}^\infty N_n = B \sum_{n=1}^\infty
    \frac{\left( t/\lambda \right)^n}{n!}e^{-t/\lambda}=B\left( 1-e^{-t/\lambda}\right) \, ,


and hence one finds that the integral over the :math:`n`-scatterings distribution 
is such that

.. math::

   N_n = \frac{N_{\rm inel}}{\left( 1-e^{-t/\lambda} \right)}\frac{\left( t/\lambda \right)^n}{n!}e^{-t/\lambda} \, ,


in terms of the normalisation :math:`N_{\rm inel}` of the full inelastic scattering distribution,
the sample thickness :math:`t` and the mean free path :math:`\lambda`.
Note also that the ZLP normalisation factor :math:`N_0` is then given in terms of the inelastic one as

.. math::

    N_0 = \frac{N_{\rm inel}}{e^{t/\lambda}-1} \, ,


and hence one has the following relations between integrated inelastic scattering intensities

.. math::

    \frac{N_1^n}{N_k}=n!N_0^{n-1} \, ,\qquad \forall ~n \ge 1 \, .


In order to evaluate the local thickness of the specimen and the corresponding dielectric function,
it is necessary to perform a deconvoluation of the measured spectra and
extract from them the single-scattering distribution (SSD), :math:`I_{\rm SSD}(E)`.
The SSD is related to the experimentally measured :math:`n=1` distribution,
:math:`I^{(1)}_{\rm inel}(E)` by the finite resolution of our measurement apparatus:

.. math:: :label: eq:def_convolution

    I^{(1)}_{\rm inel}(E) = R(E)\otimes
    I_{\rm SSD}(E) \equiv \int_{-\infty}^{\infty} dE'\, R(E-E')
    I_{\rm SSD}(E') \, ,


where in the following :math:`\otimes` denotes the convolution operation. It can
be shown, again treating individual scatterings as uncorrelated, that the
experimentally measured :math:`n=2` and :math:`n=3` multiple scattering
distributions can be expressed in terms of the SSD as

.. math::

    I^{(2)}_{\rm inel}(E) &=&  R(E)\otimes I_{\rm SSD}(E)\otimes I_{\rm SSD}(E)/\left( 2! N_0\right) \ ,
    \\
    I^{(3)}_{\rm inel}(E) &=&
    R(E)\otimes I_{\rm SSD}(E)\otimes I_{\rm SSD}(E)\otimes I_{\rm SSD}(E)/\left( 3! N^2_0\right) \ ,


and likewise for :math:`n\ge 4`. Combining this information, one observes that an
experimentally measured EELS spectrum, Eq. :eq:`eq:EELSmaster` can be expressed in terms
of the resolution function :math:`R`, the ZLP normalisation :math:`N_0`, and the single-scattering
distribution :math:`I_{\rm SSD}` as follows

.. math:: :label: eq:EELSmaster_2

    && I_{\rm EELS}(E) \nonumber = N_0 R(E) + R(E)\otimes I_{\rm SSD}(E) + R(E)\otimes
    I_{\rm SSD}(E)\otimes I_{\rm SSD}(E)/\left( 2! N_0\right) + \ldots\\ \nonumber
    && =R(E) \otimes \left( N_0\delta(E) + I_{\rm SSD}(E) + I_{\rm SSD}(E)\otimes
    I_{\rm SSD}(E)/\left( 2! N_0\right) +\ldots \right) \\
    && =N_0 R(E) \otimes \left( \delta(E) +\sum_{n=1}^{\infty} \left[
    I_{\rm SSD}(E)\otimes\right]^n \delta(E)/\left( n! N_0^{n}\right)  \right) \, ,


where :math:`\delta(E)` is the Dirac delta function. If the ZLP normalisation factor
:math:`N_0` and resolution function :math:`R(E)` are known, then one can use Eq. :eq:`eq:EELSmaster_2`
to extract the SSD from the measured spectra by means of a deconvolution procedure.


SSD deconvolution
-----------------


The structure of Eq. :eq:`eq:EELSmaster_2` suggests that transforming to Fourier 
space will lead to an algebraic equation which can then be solved for the SSD.
Here we define the Fourier transform :math:`\widetilde{f}(\nu)` of a function :math:`f(E)`
as follows

.. math:: :label: eq:continuous_fourier_transform

    \mathcal{F}\left[ f(E) \right](\nu)\equiv \widetilde{f}(\nu)\equiv \int_{-\infty}^\infty
    dE\,f(E) e^{-2\pi i E\nu}\, ,


whose inverse is given by

.. math:: :label: eq:continuous_fourier_transform_inverse

    \mathcal{F}^{-1}\left[ \widetilde{f}(\nu) \right](E) = f(E)\equiv \int_{-\infty}^\infty
    d\nu\,\widetilde{f}(\nu) e^{2\pi i E\nu}\, ,


which has the useful property that convolutions such as  Eq. :eq:`eq:def_convolution`
are transformed into products,

.. math:: :label: eq:fourier_convolutions

    {\rm if~}f(E)=g(E)\otimes h(E)\quad{\rm then}\quad \mathcal{F}\left[ f(E) \right] =
    \widetilde{f}(\nu) = \widetilde{g}(\nu)\widetilde{h}(\nu) \, .


The Fourier transform of Eq. :eq:`eq:EELSmaster_2` leads to the Taylor expansion 
of the exponential and hence

.. math::

    \widetilde{I}_{\rm EELS}(\nu)=N_0\widetilde{R}(\nu)\exp\left(  \frac{\widetilde{I}_{\rm SSD}(\nu)}{N_0}\right) \, ,


which can be solved for the Fourier transform of the single scattering distribution

.. math::

    \widetilde{I}_{\rm SSD}(\nu)=N_0 \ln \frac{\widetilde{I}_{\rm EELS}(\nu)}{N_0\widetilde{R}(\nu)}
    = N_0 \ln \frac{\mathcal{F}\left[ I_{\rm EELS}(E)\right] (\nu)}{N_0 \mathcal{F}\left[ R(E)\right] (\nu)  } \, .


By taking the  inverse Fourier transform, one obtains the sought-for expression
for the single scattering distribution as a function of the electron energy loss

.. math:: :label: eq:deconvolution_procedure

    I_{\rm SSD}(E)=N_0 \mathcal{F}^{-1}\left[ \ln \frac{\mathcal{F}\left[
    I_{\rm EELS}\right] }{N_0 \mathcal{F}\left[  R\right]}\right] \, ,


where the only required inputs are the experimentally measured EELS spectra,
Eq. :eq:`eq:EELSmaster`, with the corresponding ZLP.


Discrete Fourier transforms
---------------------------


In EELSfitter, Eq. :eq:`eq:deconvolution_procedure` is evaluated numerically by
approximating the continuous transform Eq. :eq:`eq:continuous_fourier_transform`
by its discrete Fourier transform equivalent. The same method will be used for the
implementation of the Kramers-Kronig analysis. The discrete Fourier transform  of a
discretised function :math:`f(E)` defined at :math:`E_n \in \{E_0, ..., E_{N-1}\}`
is given by:

.. math:: :label: eq_def_DFT

    \mathcal{F}_D \left[ f(E) \right] (\nu_k) = \widetilde{f}(\nu_k) = \sum^{N-1}_{n=0}
    \operatorname{e}^{-i2\pi kn/N} f(E_n), \qquad \forall\, k \in \{0, ..., N-1\} \, ,


with the corresponding inverse transformation 

.. math:: :label: eq_def_DFT_inverse

    \mathcal{F}_D^{-1} \left[ \widetilde{f}(\nu) \right] (E_n) = f(E_n) =\frac{1}{N}
    \sum^{N-1}_{k=0} \operatorname{e}^{i2\pi kn/N}  \widetilde{f}(\nu_k) \qquad
    \forall\, n \in \{0, ..., N-1\} \, .


If one approximates the continuous function :math:`f(E)` by its discretised version
:math:`f(E_0 + n\Delta E)` and likewise :math:`\widetilde{f}(\nu)` by :math:`\widetilde{f}(k\Delta \nu)` 
where :math:`\Delta x \Delta \nu = N^{-1}` one finds that

.. math:: :label: eq_approx_CFT

	\widetilde{f}(\nu) \approx \Delta x e^{-i 2\pi k \Delta \nu E_0}\mathcal{F}_D \left[ f(E)\right] \, ,


and likewise for the inverse transform

.. math::

    f(E) \approx \frac{1}{\Delta x} \mathcal{F}_D^{-1} \left[ \widetilde{g}(k\Delta\nu)
    \right] \, ,\qquad \widetilde{g}(k\Delta\nu) \equiv e^{i2\pi k \Delta\nu E_0}
    \widetilde{f}(k\Delta\nu) \, .


In practice, the EELS spectra considered are characterised by a fine spacing in 
:math:`E` and the discrete approximation for the Fourier transform produces results 
very close to the exact one.


Thickness calculation
---------------------


Once the SSD has been determined by means of the deconvolution procedure summarised 
by Eq. :eq:`eq:deconvolution_procedure`, it can be used as input in order to 
evaluate the local sample thickness :math:`t` from the experimentally measured spectra.
Kramers-Kronig analysis  provides the following relation between the thickness :math:`t`,
the ZLP normalisation :math:`N_0`, and the single-scattering distribution,

.. math:: :label: eq:thickness_calculation

    t = \frac{4a_0 F E_0}{N_0\left(  1-{\rm Re}\left[ 1/\epsilon(0)\right]\right)} \int_0^\infty
    dE\frac{I_{\rm SSD}(E)}{E\ln \left( 1+\beta^2/\theta_E^2\right)} \, ,


where we have assumed that the effects of surface scatterings can be neglected.
In Eq. :eq:`eq:thickness_calculation`, :math:`a_0=0.0529` nm is Bohr's radius, :math:`F`
is a relativistic correction factor,

.. math::

    F = \frac{  1+E_0/(1022~{\rm keV})  }{\left[ 1+E_0/(511~{\rm keV})\right]^2  } \, ,


with :math:`E_0` being the incident electron energy, :math:`\epsilon(E)` is the
complex dielectric function, and :math:`\theta_E` is the characteristic angle defined by

.. math:: :label: eq:characteristic_angle

    \theta_E = \frac{E}{\gamma m_0v^2} = \frac{E}{\left( E_0 + m_0c^2\right) (v/c)^2}


with :math:`\gamma` being the usual relativistic dilation factor, :math:`\gamma=\left( 1-v^2/c^2\right)^{-1/2}`,
and :math:`\beta` the collection semi-angle of the microscope. [#f3]_




For either an  insulator or a semiconductor material
with refractive index :math:`n`, one has that

.. math:: :label: eq:refractive_index

    {\rm Re}\left[ 1/\epsilon(0)\right] = n^{-2} \, ,


while :math:`{\rm Re}\left[ 1/\epsilon(0)\right]=0` for a metal or semi-metal.

Hence, the determination of the dielectric function is not a pre-requisite to 
evaluate the specimen thickness, and for given microscope operation conditions 
we can express Eq. :eq:`eq:thickness_calculation` as

.. math:: :label: eq:thickness_calculation_v2

    t = \frac{A}{N_0} \int_0^\infty dE\frac{I_{\rm SSD}(E)}{E\ln \left( 1+\beta^2/\theta_E^2\right)} \, ,


with :math:`A`  constant across the specimen.

If the thickness of the specimen is already known at some location, then
Eq. :eq:`eq:thickness_calculation_v2`  can be  used to calibrate :math:`A`
and evaluate this thickness elsewhere. Furthermore, if the thickness of the material
has already been determined by means of an independent experimental technique, then
Eq. :eq:`eq:thickness_calculation` can be inverted to determine the refractive index :math:`n`
of an insulator or semi-conducting material using

.. math::

    n = \left[ 1-\frac{4a_0 FE_0}{N_0 t} \left( \int_0^\infty
    dE\frac{I_{\rm SSD}(E)}{E\ln \left( 1+\beta^2/\theta_E^2\right)} \right) \right]^{-1/2} \, .


The dielectric function from Kramers-Kronig analysis
----------------------------------------------------


The dielectric function of a material, also known as permittivity, is a measure 
of how easy or difficult it is to polarise a dielectric material such an insulator 
upon the application of an external electric field. In the case of oscillating electric
fields such as those that constitute electromagnetic radiation, the dielectric response
will have both a real and a complex part and will depend on the oscillation frequency :math:`\omega`,

.. math::

    \epsilon(\omega)={\rm Re}\left[ \omega\right]+i{\rm Im}\left[ \omega\right] \, ,


which can also be expressed in terms of the energy :math:`E=\hbar \omega` of the photons
that constitute this electromagnetic radiation,

.. math:: :label: eq:dielectric_function_def

    \epsilon(E)={\rm Re}\left[ \epsilon(E)\right]+i{\rm Im}\left[ \epsilon(E)\right] \, .


In the vacuum, the real and imaginary parts of the dielectric function reduce to
:math:`{\rm Re}\left[ \epsilon(E)\right]=1` and :math:`{\rm Im}\left[ \epsilon(E)\right]=0`.
Furthermore, the dielectric function is related to the susceptibility :math:`\chi` by

.. math::

    \epsilon(E)=1-\nu\chi(E) \, ,


where :math:`\nu` is the so-called Coulomb matrix. The single scattering distribution
:math:`I_{\rm SSD}(E)` is related to the imaginary part of the  complex dielectric function
:math:`\epsilon(E)` by means the following relation

.. math::

    I_{\rm SSD}(E) = \frac{N_0 t}{\pi a_0 m_0 v^2}{\rm Im}\left[ \frac{-1}{\epsilon(E)}\right]
    \ln \left[ 1+\left( \frac{\beta}{\theta_E}\right)^2\right] \, ,


in terms of the sample thickness :math:`t`, the ZLP normalisation :math:`N_0`, and
the microscope operation parameters defined in Sect. :ref:`theory/kk_analysis:Thickness calculation`.
We can invert this relation to obtain

.. math:: :label: eq:im_diel_fun

    {\rm Im}\left[ \frac{-1}{\epsilon(E)}\right] = \frac{\pi a_0 m_0 v^2}{N_0 t}\frac{
    I_{\rm SSD}(E)}{\ln \left[ 1+\left( \frac{\beta}{\theta_E}\right)^2\right]} \, .


Since the prefactor in Eq. :eq:`eq:im_diel_fun` does not depend on the energy loss :math:`E`,
we see that :math:`{\rm Im}[-1/\epsilon(E)]` will be proportional to the single scattering
distribution :math:`I_{\rm SSD}(E)` with a denominator that decreases with the energy
(since :math:`\theta_E\propto E`) and hence weights more higher energy losses. Given that
the dielectric response function is causal, the real part of the dielectric function
can be obtained from the imaginary one by using a Kramers-Kronig relation of the form

.. math:: :label: eq:kramerskronig

    {\rm Re}\left[ \frac{1}{\epsilon(E)}\right] = 1-\frac{2}{\pi}\mathcal{P}\int_0^{\infty}  dE'\, {\rm Im}
    \left[ \frac{-1}{\epsilon(E')}\right] \frac{E'}{E'^2-E^2} \, ,


where :math:`\mathcal{P}` stands for Cauchy's prescription to evaluate the principal
part of the integral.

A particularly important application of this relation is the :math:`E=0` case,

.. math:: :label: eq:normalisation_im_deltaEim

    {\rm Re}\left[ \frac{1}{\epsilon(0)}\right] = 1-\frac{2}{\pi}\mathcal{P}\int_0^{\infty}  dE\, {\rm Im}
    \left[ \frac{-1}{\epsilon(E)}\right] \frac{1}{E} \, ,


which is known as the Kramers-Kronig sum rule. Eq. :eq:`eq:normalisation_im_deltaEim`
can be used to determine the overall normalisation of :math:`{\rm Im}\left[ -1/\epsilon(E)\right]`,
since :math:`{\rm Re}\left[ 1/\epsilon(0)\right]` is known for most materials.
For instance, as mentioned in Eq. :eq:`eq:refractive_index`, for an insulator
or semiconductor material it is given in terms of its refractive index :math:`n`.
Once the imaginary part of the dielectric function has been determined from the
single-scattering distribution, Eq. :eq:`eq:im_diel_fun`, then one can obtain 
the corresponding real part by means of the Kramers-Kronig relation, Eq. :eq:`eq:kramerskronig`.
Afterwards, the full complex dielectric function can be reconstructed by combining
the calculation of the real and imaginary parts, since

.. math::

    \epsilon(E)={\rm Re}\left[ \epsilon(E)\right]+i{\rm Im}\left[ \epsilon(E)\right] \equiv
    \epsilon_1(E)+i\epsilon_2(E) \, ,


implies that

.. math::

    {\rm Re}\left[ \frac{1}{\epsilon(E)}\right] = \frac{\epsilon_1(E)}{\epsilon_1^2(E) + \epsilon_2^2(E)}\,,\qquad
    {\rm Im}\left[ \frac{-1}{\epsilon(E)}\right] = \frac{\epsilon_2(E)}{\epsilon_1^2(E) + \epsilon_2^2(E)}\,,


and hence one can express the dielectric function in terms of the quantities that
one has just evaluated as follows

.. math:: :label: eq:final_dielectric_function

    \epsilon(E) = \frac{{\rm Re}\left[ \frac{1}{\epsilon(E)}\right]+ i{\rm Im}\left[ \frac{-1}{\epsilon(E)}\right]}{\left( {\rm Re}\left[ \frac{1}{\epsilon(E)}\right]\right)^2+\left( {\rm Im}\left[ \frac{-1}{\epsilon(E)}\right]\right)^2} \, .


Once the complex dielectric function of a material has been determined, it is 
possible to evaluate related quantities that also provide information about the 
opto-electronic properties of a material.

One example of this would be the optical absorption coefficient, given by

.. math::

    \mu(E) = \frac{E}{\hbar c}\left[ 2\left( \epsilon_1^2(E)+\epsilon_2^2(E)\right)^{1/2}-2\epsilon_1(E)\right]^{1/2} \, ,


which represents a measure of how far light of a given wavelength :math:`\lambda=hc/E`
can penetrate into a material before it is fully extinguished via absorption processes.
The complex dielectric function :math:`\epsilon(E)` provides direct information on
the opto-electronic properties of a material, for example those associated to plasmonic
resonances. Specifically, a collective plasmonic excitation should be indicated by the
condition that the real part of the dielectric function crosses the :math:`x` axis,
:math:`\epsilon_1(E)=0`, with a positive slope. These plasmonic excitations typically
are also translated by a well-defined peak in the energy loss spectra. Hence,
verifying that a plasmonic transition indicated by :math:`\epsilon_1(E)=0`
corresponds to specific energy-loss features provides a valuable handle to 
pinpoint the nature of local electronic excitations present in the analysed specimen.


The role of surface scatterings
-------------------------------


The previous derivations assume that the specimen is thick enough such that the 
bulk of the measured energy loss distributions arises from volume inelastic 
scatterings, while edge- and surface-specific contributions can be neglected.
However, for relatively thin samples with thickness :math:`t` below a few tens of nm,
this approximation is not necessarily suitable. Assuming a locally flat specimen with
two surfaces, in this case  Eq. :eq:`eq:EELSmaster` must be generalised to

.. math:: :label: eq:EELSmaster_v3

    I_{\rm EELS}(E) = I_{\rm ZLP}(E) + I_{\rm inel}(E) +  I_{S}(E)


with :math:`I_{S}(E)` representing the  contribution from surface-specific inelastic 
scattering. This surface contribution can be evaluated in terms of the real :math:`\epsilon_1` and
imaginary :math:`\epsilon_2` components of the complex dielectric function,

.. math:: :label: eq:surface_intensity

    I_{S}(E) = \frac{N_0}{\pi a_0 k_0 T} \left[ \frac{\tan^{-1}(\beta/\theta_E)}{\theta_E} -
    \frac{\beta}{\beta^2+\theta_E^2} \right] \left( \frac{4\epsilon_2}{\left( \epsilon_1 + 1 \right)^2
    +\epsilon_2^2} - {\rm Im}\left[\frac{-1}{\epsilon(E)} \right] \right) \, ,


where the electron kinetic energy is :math:`T=m_ev^2/2`. The main challenge to evaluate
the surface component from Eq. :eq:`eq:surface_intensity` is that it depends on the
complex dielectric function :math:`\epsilon(E)`, which in turn is a function of the
single scattering distribution obtained from the deconvolution of :math:`I_{\rm inel}(E)`
obtained assuming that :math:`I_S(E)` vanishes. For not too thin specimens, the best approach
is then an iterative procedure, whereby one starts by assuming that :math:`I_{S}(E)\simeq 0`,
evaluates :math:`\epsilon(E)`, and uses it to evaluate a first approximation to :math:`I_S(E)`
using Eq. :eq:`eq:surface_intensity`. This approximation is then subtracted from Eq.
:eq:`eq:EELSmaster_v3` and hence provides a better estimate of the bulk contribution
:math:`I_{\rm inel}(E)`. One can go back to the first step and iterate the procedure
until some convergence criterion is met. Whether or not this procedure converges will
depend on the specimen under consideration, and specifically on the features of the EELS
spectra at low energy losses, :math:`E \lesssim 10` eV.

For the specimens considered in this study, it is found that this iterative procedure
to determine the surface contributions converges provided that the local sample
thickness satisfies :math:`t \gtrsim 20` nm. For thinner samples the iterative approach
fails to converge and another strategy would be needed. Hence in this work we disentangle the
bulk from the surface contributions to the EELS spectra only when the thickness is above this threshold.


The role of relativistic contributions
--------------------------------------


When an incoming charged particle moves inside a medium, while having a higher velocity than the phase velocity of light
inside that medium, Cerenkov radiation can be emitted. The magnetic and electric fields that come with the charge
excite the atoms of the medium. These excited atoms then radiate their energy at a fixed angle with respect to the
trajectory of the particle. The influence of cerenkov radiation is not to be understated as it contributes most in
the low loss region between 0 and 5 eV depending on the medium, therefore impacting the dielectric function from
Kramers-Kronig analysis and band gap determination
:cite:p:`VonFestenberg1968, Moreau1997, StogerPollach2006, StogerPollach2008, Erni2008, StogerPollach2014, Horak2015, Erni2016, Sakaguchi2016`.



.. [#f1] Here :math:`E` indicates the energy lost by EELS electrons, with :math:`E` for elastic scatterings.
.. [#f2] The following derivation assumes that the specimen is not too thin, that is, :math:`t \gtrsim \lambda`.
.. [#f3] Which should not be confused with the normalised velocity often used in relativity, :math:`\beta=v/c`.