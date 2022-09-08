# VSI_automated_disease_detection_public

 Code for "Statistical and Topological Summaries Aid Disease Detection for Segmented Retinal Vascular Images," by John T. Nardini, Charles W. Pugh, and Helen M. Byrne. Article available at https://arxiv.org/abs/2202.09708
 
 All code is implemented using Python 3. All data used in the study is located in **Data/**, code and implementation of the Network Extraction From Images (NEFI) software is located in **NEFI/**, all code used for analysis is in **Notebooks/**, and all results are located in **Results/**.
 
The **Data/** folder contains vessel segmentation image (VSI) data that we compiled from multiple datasets. The VSIs from the STARE datasets are in Data/Dataset_1/Provided_masks and Data/Dataset_1/Provided_masks_VK, the data from the HRF dataset is located in Data/HRF_Dataset_1/Provided_masks, the data from the DRIVE dataset is located in Data/DRIVE/Provided_masks, and data from the CHASE dataset is located in Data/CHASEDB1/Provided_masks. The code Create_All_dataset.ipynb was used to generate the All dataset.

The **Notebooks/** folder contains jupyter notebook files to perform the analyses of our study. TDA_pipeline.ipynb performs the computation of all topological descriptor vectors, Fractal\ dimension.ipynb performs computation of the standard descriptors and performs disease classification for these descriptors, binary_TDA_classification.ipynb performs disease classification for the topological descriptor vectors, PCA\ interpretation\ plot.ipynb depicts the PCA plot (Figure 6),and Investigating_loop_size_num.ipynb investigates the number of loops and the size of loops in the All dataset (Figure 7).

The **Results/** folder contains the saved descriptor vectors from each dataset. Dataset_1 is for the STARE expert 1 dataset, Dataset_1_VK is for the STARE expert 2 dataset, HRF_Dataset_1 is for the HRF dataset, all for the All dataset, 
 
 Please contact John Nardini at nardinij@tcnj.edu if you have any questions, thank you.
