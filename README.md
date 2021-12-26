1. Environment:

    Python --version >= 3.7
        Third Party: numpy, matplotlib, math, scipy.io, tensorflow.keras
        
    Matlab_R2021a
        Third Party: libsvm-3.25

2. Preparation

    Download CMU PIE face dataset from the link in Assignment Description.
    And put the 25 classes folders into `./face/` and your 10 photos after transformed to 32x32 (grey scale) as a folder named 'myself' into `./face/69/`.
    Run the `dataloader.m`  to get `facedata.mat`.

3. Content

    facedata.mat
    
    PCA
        pca_calculation.m
        pca_visualize.m
    LDA
        lda_calculate.m
        lda_visualize.m
    GMM
        GMM.py
    SVM
        libsvm-3.25
        svm.m
    CNN
        CNN.py





