# Hexatic_Phase_Analysis
This repo is a Python pipeline for processing SEM images of nanoparticle arrays. It enhances images, extracts particle centers, computes the bond–orientational order parameter (ψ₆), and calculates the correlation function G₆(D) to check for hexatic phase order. 

## Background

Two-dimensional nanoparticle arrays may form a hexatic phase—characterized by short-range positional order but quasi–long-range orientational order. Following the approach in [1] and the theoretical summary from, the analysis proceeds by:
- **Extracting Particle Positions:** After image adjustment and edge detection, particle centers are found.
- **Calculating ψ₆:** For each particle, ψ₆ = (1/N) Σ₍ⱼ₎ exp(6iθ₍ᵢⱼ₎) is computed, where the sum is over nearest neighbors.
- **Computing G₆(D):** The correlation function G₆(D) = ⟨ψ₆*(0) ψ₆(D)⟩ is estimated by binning pairwise correlations over distance.

![Hexatic Phase Diagram](main/Hexatic.png)


## Features

- **Image Preprocessing:**  
  Uses median filtering, Gaussian blur, and Laplacian edge detection to enhance SEM images for automated particle detection.

- **Particle Extraction:**  
  Detects nanoparticle centers via peak detection and converts pixel coordinates to nanometers.

- **Order Parameter Calculation:**  
  Computes ψ₆ using Delaunay triangulation for nearest neighbor detection.

- **Correlation Analysis:**  
  Bins the pairwise correlations to produce G₆(D), providing insight into quasi–long-range orientational order.


  [1] - Mohtasebzadeh, A. R., Davidson, J. C., Livesey, K. L., & Crawford, T. M. (2022). Tunability and ordering in 2D arrays of magnetic nanoparticles assembled via extreme field gradients. Advanced Materials Interfaces, 9(26), 2201056.


