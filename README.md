# DICOM DVH Extraction Pipeline

- A Python pipeline for extracting dosimetric metrics from DICOM 
  radiotherapy files (CT, RT Structure, RT Dose, RT Plan).

# Features
- Reads and auto-matches DICOM files across modalities by FrameOfReferenceUID
- Reconstructs 3D ROI masks from RT Structure contour data
- Resamples RT Dose grids to CT geometry using SimpleITK
- Computes DVH, D90, D2cc for target and OAR structures

# Dependencies
- pydicom
- SimpleITK
- numpy
- scikit-image
