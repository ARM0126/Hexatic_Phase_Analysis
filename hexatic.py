
---

## 2. Python Code: Image Preprocessing & Analysis Pipeline

Below is the complete Python script (`hexatic_analysis.py`) incorporating image preprocessing along with the bond–orientational analysis. The code is extensively commented to explain its purpose and parameters.

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, img_as_float
from scipy.spatial import distance_matrix, Delaunay
import csv

# ======================================
# 1. Image Preprocessing & Particle Extraction
# ======================================

def preprocess_image(image_path):
    """
    Loads and preprocesses an SEM image.
    
    Steps:
    - Convert to float and grayscale if needed.
    - Apply median filtering to reduce noise.
    - Apply Gaussian Blur to smooth the image.
    - Compute Laplacian (edge detection) to enhance particle boundaries.
    
    Parameters:
        image_path (str): Path to the SEM image.
        
    Returns:
        image_processed (ndarray): The processed image for peak detection.
    """
    # Read image and convert to float
    image = img_as_float(io.imread(image_path, as_gray=True))
    
    # Apply median filter (radius can be tuned)
    image_med = filters.median(image, behavior='ndimage')
    
    # Apply Gaussian Blur (sigma parameter may need adjustment)
    image_blur = filters.gaussian(image_med, sigma=5)
    
    # Apply Laplacian filter for edge enhancement
    image_lap = filters.laplace(image_blur)
    
    # Combine blurred image and Laplacian to enhance features
    # (A simple combination; feel free to modify this strategy.)
    image_processed = image_blur - image_lap
    
    return image_processed

def extract_particle_coords(image, min_distance=5, threshold_rel=0.1):
    """
    Detects particle centers using a peak detection algorithm.
    
    Parameters:
        image (ndarray): Preprocessed SEM image.
        min_distance (int): Minimum separation between peaks (in pixels).
        threshold_rel (float): Relative threshold for peak detection.
        
    Returns:
        coords (ndarray): Array of (row, col) coordinates of detected particles.
    """
    # Using peak_local_max from skimage.feature to find local maxima
    coords = feature.peak_local_max(image, min_distance=min_distance, threshold_rel=threshold_rel)
    return coords

# ======================================
# 2. Bond–Orientational Order Calculation
# ======================================

def compute_psi6(coordinates):
    """
    Computes the local bond–orientational order parameter ψ₆ using Delaunay triangulation.
    
    Parameters:
        coordinates (ndarray): Array of (row, col) positions (assumed in nm after conversion).
        
    Returns:
        psi6 (ndarray): Array of complex ψ₆ values for each particle.
    """
    tri = Delaunay(coordinates)
    n_points = len(coordinates)
    
    # Build neighbor list from Delaunay triangles
    neighbors = {i: set() for i in range(n_points)}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                neighbors[simplex[i]].add(simplex[j])
                neighbors[simplex[j]].add(simplex[i])
    
    psi6 = np.zeros(n_points, dtype=complex)
    for i, coord in enumerate(coordinates):
        nb_indices = np.array(list(neighbors[i]))
        if nb_indices.size == 0:
            psi6[i] = 0
        else:
            # Compute the angle (θ) relative to x-axis for each neighbor
            dy = coordinates[nb_indices, 0] - coord[0]
            dx = coordinates[nb_indices, 1] - coord[1]
            angles = np.arctan2(dy, dx)
            psi6[i] = np.mean(np.exp(1j * 6 * angles))
    return psi6

def bond_orientation_correlation(psi6, coordinates, bin_width=27):
    """
    Calculates the bond–orientational correlation function G₆(D) by:
      - Computing pairwise distances between particles.
      - Forming the real part of the pairwise correlation: Re[conjugate(ψ₆[i]) * ψ₆[j]].
      - Binning the values over distance intervals of width bin_width (nm).
    
    Parameters:
        psi6 (ndarray): The complex local bond–orientational order parameter.
        coordinates (ndarray): Array of (x,y) positions (in nm).
        bin_width (float): Width of the bins (in nm) for grouping particle separations.
    
    Returns:
        bin_centers (ndarray): Centers of the distance bins.
        g6_bin (ndarray): Mean G₆ values in each bin.
    """
    dm = distance_matrix(coordinates, coordinates)
    n_points = len(coordinates)
    
    # Compute pairwise correlation matrix (real part)
    corr_matrix = np.real(np.outer(np.conjugate(psi6), psi6))
    
    # Use only upper triangle to avoid duplicate pairs
    iu = np.triu_indices(n_points, k=1)
    distances = dm[iu]
    correlations = corr_matrix[iu]
    
    # Define bins based on bin_width
    max_distance = distances.max()
    bins = np.arange(0, max_distance + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the pair correlations
    g6_bin = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if np.any(mask):
            g6_bin[i] = np.mean(correlations[mask])
        else:
            g6_bin[i] = np.nan
            
    return bin_centers, g6_bin

# ======================================
# 3. Complete Processing Pipeline
# ======================================

def process_sem_images(image_folder, conversion_factor=2.335766423, min_distance=5,
                       threshold_rel=0.1, bin_width=27):
    """
    Processes all SEM images in the specified folder:
      - Preprocesses each image.
      - Extracts particle coordinates.
      - Converts coordinates from pixels to nanometers.
      - Computes ψ₆ and G₆(D).
      - Exports binned data as CSV and plots G₆(D).
    
    Parameters:
        image_folder (str): Folder containing SEM image files.
        conversion_factor (float): Factor to convert pixels to nanometers.
        min_distance (int): Minimum distance (in pixels) between particle centers.
        threshold_rel (float): Relative threshold for detecting peaks.
        bin_width (float): Binning width (nm) for correlation function.
    
    Returns:
        avg_bin_centers (ndarray): Distance bins (nm).
        avg_g6 (ndarray): Averaged G₆ values across all images.
    """
    all_g6 = []
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.tif', '.tiff')):
            image_path = os.path.join(image_folder, file)
            print(f"Processing image: {file}")
            
            # Load and preprocess the image
            proc_image = preprocess_image(image_path)
            
            # Extract particle coordinates in pixel space
            coords_pixels = extract_particle_coords(proc_image, min_distance, threshold_rel)
            if coords_pixels.shape[0] < 3:
                print(f"Insufficient particles detected in {file}. Skipping.")
                continue
            
            # Convert pixel coordinates to nanometers
            coords_nm = coords_pixels * conversion_factor
            
            # Compute the local bond–orientational order parameter ψ₆
            psi6 = compute_psi6(coords_nm)
            
            # Compute bond–orientational correlation function G₆(D)
            bin_centers, g6_vals = bond_orientation_correlation(psi6, coords_nm, bin_width)
            all_g6.append(g6_vals)
    
    if not all_g6:
        print("No images processed successfully.")
        return None, None
    
    # Average the correlation function over all processed images
    all_g6 = np.array(all_g6)
    avg_g6 = np.nanmean(all_g6, axis=0)
    
    # Export the averaged data to a CSV file
    export_results(bin_centers, avg_g6, output_filename='G6_averaged_results.csv')
    
    # Plot the averaged G₆(D)
    plt.figure()
    plt.plot(bin_centers, avg_g6, 'o-')
    plt.xlabel('Distance r (nm)')
    plt.ylabel('G₆(r)')
    plt.title('Averaged Bond–Orientational Correlation Function')
    plt.show()
    
    return bin_centers, avg_g6

def export_results(bin_centers, g6_values, output_filename='G6_results.csv'):
    """
    Exports the binned correlation data to a CSV file.
    """
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance (nm)", "G6"])
        for r, g6 in zip(bin_centers, g6_values):
            writer.writerow([r, g6])
    print(f"Results exported to {output_filename}")

# ======================================
# 4. Main Execution
# ======================================
if __name__ == '__main__':
    # Folder containing SEM images (update the path accordingly)
    image_folder = 'examples/SEM_images'
    
    # Example parameters (adjust as needed)
    conversion_factor = 2.335766423   # Pixels to nm conversion factor determined via scale bar
    min_distance = 5                  # Minimum separation in pixels for particle detection
    threshold_rel = 0.1               # Relative threshold for peak detection
    bin_width = 27                    # Bin width in nm for computing G₆(D)
    
    # Process all SEM images in the folder
    process_sem_images(image_folder, conversion_factor, min_distance, threshold_rel, bin_width)
