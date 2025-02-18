Band gap analysis
=================

One important  application of ZLP-subtracted EELS spectra is the determination
of the band gap energy and type (direct or indirect) in semiconductor materials.
The reason is that the onset of the inelastic scattering intensity provides
information on the value of the band gap energy :math:`E_{\rm bg}`, while its shape
for :math:`E \gtrsim E_{\rm bg}` is determined by the underlying band structure.
Different approaches  have been put proposed to evaluate :math:`E_{\rm bg}` from
subtracted EEL spectra, such as by means of the inflection point of the rising
intensity or a linear fit to the maximum positive slope :cite:p:`Schamm2003`.
Following :cite:p:`Roest2021`, here we adopt the method of :cite:p:`Rafferty1998, Rafferty2000`,
where the behaviour of :math:`I_{\rm inel}(E)` in the region close to the onset of
the inelastic scatterings is described by

.. math:: :label: eq:I1

    I_{\rm inel}(E) \simeq  A \left( E-E_{\rm bg} \right)^{b} \, , \quad E \gtrsim E_{\rm bg} \, ,


and vanishes for :math:`E < E_{\rm bg}`. Here :math:`A` is a normalisation constant, while
the exponent :math:`b` provides information on the type of band gap: it is expected
to be :math:`b\simeq0.5~(1.5)` for a semiconductor material characterised by a
direct (indirect) band gap. While :eq:`eq:I1` requires as input the complete
inelastic distribution, in practice the onset region is dominated by the
single-scattering distribution, since multiple scatterings contribute only
at higher energy losses.

The band gap energy :math:`E_{\rm bg}`, the overall normalisation factor :math:`A`, and
the band gap exponent :math:`b` can be determined from a least-squares fit to the
experimental data on the ZLP-subtracted spectra. This polynomial fit is carried
out in the energy loss region around the band gap energy, :math:`[ E^{(\rm fit)}_{\rm min}, E^{(\rm fit)}_{\rm max}]`.
A judicious choice of this interval is necessary to achieve stable results:
a too wide energy range will bias the fit by probing regions where Eq. :eq:`eq:I1`
is not necessarily valid, while a too narrow fit range might not contain
sufficient information to stabilize the results and be dominated by statistical
fluctuation.

.. _Bandgapfitsfig:

.. figure:: figures/Fig-SI5.png
    :width: 95%
    :class: align-center
    :figwidth: 90%
    :figclass: align-center

    *Representative examples of bandgap fits to the onset of inelastic
    spectra in the InSe (a) and WS* \ :sub:`2`\  *(b) specimens. The red shaded
    areas indicate the polynomial fitting range, the blue curve and
    band corresponds to the median and 68\% CL intervals of the
    ZLP-subtracted intensity* :math:`I_{\rm inel}(E)` *, and the outcome of
    the bandgap fits based on Eq.* :eq:`eq:I1` *is indicated by the
    green dashed curve (median) and band (68\% CL intervals).*


:numref:`Bandgapfitsfig` (a,b) displays representative examples of bandgap
fits to the onset of the inelastic spectra in the InSe (WS\ :sub:`2`\ ) specimens
respectively. The red shaded areas indicate the fitting range, bracketed
by :math:`E^{(\rm fit)}_{\rm min}` and :math:`E^{(\rm fit)}_{\rm max}`. The blue
curve and band corresponds to the median and 68\% CL intervals of the
ZLP-subtracted intensity :math:`I_{\rm inel}(E)`, and the outcome of the band gap
fits based on Eq. :eq:`eq:I1` is indicated by the green dashed curve (median)
and band (68\% CL intervals). Here the onset exponents :math:`b` have been kept
fixed to :math:`b=0.5~(1/5)` for the InSe (WS\ :sub:`2`\ ) specimen given the direct (indirect)
nature of the underlying band gaps. One observes how the fitted model describes
well the behaviour of :math:`I_{\rm inel}(E)` in the onset region for both specimens,
further confirming the reliability of our strategy to determine the band gap energy
:math:`E_{\rm bg}`. As mentioned in :cite:p:`Rafferty2000`, it is important to avoid
taking a too large interval for :math:`[ E^{(\rm fit)}_{\rm min}, E^{(\rm fit)}_{\rm max}]`,
else the polynomial approximation ceases to be valid, as one can also see directly
from these plots.


Retardation losses
------------------


When attempting to measure the band gap using EELS, it is important to consider the effects of retardation losses
(or Cerenkov losses) :cite:p:`StogerPollach2006, StogerPollach2008, Horak2015, Erni2016`. For more details see:
:ref:`theory/kk_analysis:The role of relativistic contributions`. In short, the retardation losses can pose a limit on
the determination of electronic properties in the low loss region. The best course of action is to reduce the influence
of retardation effects (e.g. lower beam voltage, sample thickness considerations) when taking measurements.
