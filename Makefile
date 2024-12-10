.PHONY: all download analysis visualization

DATASET_URL=https://snd.se/en/catalogue/download/dataset/snd1156-1/1?principal=his.se&filename=DREAM2020_SND1156-001-V1.0.zip
DATASET_ZIP_FILE=dataset.zip
DATASET_DIR_PATH=dataset
SYMMETRY_ANALYSIS_FILE=src/symmetry_analysis.py
VISUALIZATION_FILE=src/skeleton_visualization.py

all: download analysis visualization

download:
	wget --progress=bar:force $(DATASET_URL) -O $(DATASET_ZIP_FILE)
	mkdir -p $(DATASET_DIR_PATH)
	unzip -q $(DATASET_ZIP_FILE) -d $(DATASET_DIR_PATH)
	rm -f $(DATASET_ZIP_FILE)

analysis:
	python $(SYMMETRY_ANALYSIS_FILE)

visualization:
	python $(VISUALIZATION_FILE)
