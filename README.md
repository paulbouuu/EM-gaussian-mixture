# EM Algorithm for 2D Mixture of Gaussians

This project implements the Expectation-Maximization (EM) algorithm to fit a 2D Mixture of Gaussians (MoG). It includes dynamic visualization of the algorithm's progress over iterations and can generate an animated GIF showing how the clusters evolve.

The class `MoG` implements the EM algorithm for the MoG. It is inside [MoG.py](MoG.py). An example of how to use it is provided in the Jupyter notebook [EM_algorithm.ipynb](EM_algorithm.ipynb).

All the theory behind the EM algorithm is explained in detail in the pdf [EM_algorithm_for_MoG.pdf](resources/EM_algorithm_for_MoG.pdf).

---

## 🚀 Features

- Implements EM for Gaussian mixtures in 2D
- Visualizes Gaussian components using ellipses
- Optional creation of a GIF animation
- Minimal dependencies

---

## 📁 Repository Structure

```
.
├── MoG.py                # EM algorithm for MoG
├── utils.py              # GIF generation & data generation
├── images/               # Saved plots for each iteration (will be created during the first run)
├── resources/            # GIF examples and pdf on the theory
├── gaussian_mixture.gif  # Output animation (will be created during the first run)
├── EM_algorithm.ipynb    # Example usage in Jupyter
└── README.md
```

---

## 📦 Requirements

- Numpy
- Matplotlib
- Scipy
- Pillow  (if you set `create_gif=True`)

---

## 🧪 How to Run

You can test the algorithm and generate visualizations with the included notebook `EM_algorithm.ipynb`. This repo has been designed to be simple to undersand and modify.

## Documentation:

**_class_ `MoG`:**
- **Parameters:**
    - `k` (`int`): number of clusters, default is `3`
    - `domain` (`list`): domain of the data, default is `[-5, 5]`

**Other parameters:**
- `iterations` : number of iterations of the EM algorithm
- `plotting` : whether to display the plot interactively
- `create_gif` : whether to create a GIF animation of the results (needs pillow package)

## 📈 Output

The EM algorithm saves visualizations of each iteration in the `images/` folder. Once training completes, it generates an animated GIF (`gaussian_mixture.gif`) illustrating how the Gaussians fit the data over time.

![Gaussian Mixture Animation](https://github.com/paulbouuu/EM_gaussian_mixture/raw/main/resources/optimal_gaussian_mixture.gif)

### License
This project is free to use and modify under the MIT License. See the [LICENSE](LICENSE) file for details.