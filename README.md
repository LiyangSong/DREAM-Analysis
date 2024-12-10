# DREAM Dataset Analysis

## How to reproduce


1. Set up the environment
```bash
conda env create -f environment.yml
conda activate dream-analysis-env
```

2. Download the raw dataset using `Make` command
```bash
make download
```

3. Execute symmetry analysis and output a csv format result in `results/anslysis` folder
```bash
make analysis
```

4. Execute skeleton visualization and output mp4 format videos in `results/visualization` folder
```bash
make visualization
```

