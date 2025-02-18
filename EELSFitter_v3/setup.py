import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EELSFitter",
    version="3.1.0",
    author="J.M. Sangers, A. Brokkelkamp, J.J. ter Hoeve, I. Postmes, L. Maduro, J. Rojo, S. Conesa Boj",
    author_email="j.j.ter.hoeve@vu.nl",
    description="Electron Energy Loss Spectroscopy Fitter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LHCfitNikhef/EELSfitter",
    project_urls={
        "Documentation": "https://lhcfitnikhef.github.io/EELSfitter",
    },
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "ncempy>=1.8.0",
        "matplotlib>=3.3.2",
        "numpy>=1.19.2",
        "natsort>=7.1.0",
        "scipy>=1.5.2",
        "torch>=1.6.0",
        "scikit_learn>=1.0",
        "kneed>=0.8",
        ],
    python_requires=">=3.7",
)
