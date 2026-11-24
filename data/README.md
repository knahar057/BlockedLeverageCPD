# Data

The datasets used in the numerical experiments are **not included in this repository**.

Download the datasets from their original sources and place the required files in this directory using the filenames and directory structure specified below.

The data files are excluded from version control and should not be committed to this repository.

---

## Miranda Turbulent Flow Dataset

The Miranda Turbulent Flow Dataset is available from:

https://gitlab.com/tensors/tensor_data_miranda_sim

The experiments use the density tensor contained in

```text
density.mat
```

Download the file and place it at

```text
data/density.mat
```
---

## Chicago Crime Tensor

The Chicago Crime tensor is available from the FROSTT tensor collection:

https://frostt.io/tensors/chicago-crime/

Download

```text
chicago-crime-comm.tns.gz
```

and extract the compressed file.

Place the extracted file at

```text
data/chicago-crime-comm.tns
```

---

## COIL-100

The Columbia Object Image Library (COIL-100) is available from the Columbia University Computer Vision Laboratory:

https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php

COIL-100 contains **7,200 color images of 100 objects**. Each object is photographed at 72 different viewing angles, corresponding to rotations from 0 to 355 degrees in 5-degree increments.

The images have resolution

```text
128 × 128
```

and are provided as RGB color images.

Download and extract the COIL-100 dataset.

Place the extracted image directory inside the `data/` directory so that the images are available at

```text
data/coil-100/
```

The expected directory structure is approximately

```text
data/
└── coil-100/
    ├── obj1__0.png
    ├── obj1__5.png
    ├── obj1__10.png
    ├── ...
    ├── obj100__345.png
    ├── obj100__350.png
    └── obj100__355.png
```

---

## Expected Data Directory

After downloading the datasets, the local `data/` directory should have the following general structure:

```text
data/
├── README.md
├── density.mat
├── chicago-crime-comm.tns
└── coil-100/
    ├── obj1__0.png
    ├── obj1__5.png
    ├── ...
    └── obj100__355.png
```
