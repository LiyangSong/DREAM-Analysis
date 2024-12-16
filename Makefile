.PHONY: all download analysis visualization

DATASET_URL=https://snd.se/en/catalogue/download/dataset/snd1156-1/1?principal=his.se&filename=DREAM2020_SND1156-001-V1.0.zip
DATASET_ZIP_FILE=dataset.zip
DATASET_INNER_ZIP_FILE=DREAMdataset.zip
DATASET_DIR_PATH=dataset
SYMMETRY_ANALYSIS_FILE=src/symmetry_analysis.py
VISUALIZATION_FILE=src/skeleton_visualization.py

all: download analysis visualization

download:
	@echo "Downloading dataset from $(DATASET_URL)..."
	@if wget --progress=bar:force "$(DATASET_URL)" -O $(DATASET_ZIP_FILE); then \
		echo "Download successful."; \
	else \
		echo "Download failed. Please check the URL."; \
		rm -f $(DATASET_ZIP_FILE); \
		exit 1; \
	fi

	@echo "Unzipping dataset to $(DATASET_DIR_PATH)..."
	@if unzip -q $(DATASET_ZIP_FILE) -d $(DATASET_DIR_PATH); then \
		echo "Unzip successful."; \
		rm -f $(DATASET_ZIP_FILE); \
	else \
		echo "Unzip failed."; \
		exit 1; \
	fi

	@echo "Unzipping any nested zip files..."
	@find $(DATASET_DIR_PATH) -name "*.zip" -exec bash -c "unzip -q {} -d $(DATASET_DIR_PATH) && echo {} && rm -f {}" \;
	@echo "Nested zip files have been unzipped."

analysis:
	python $(SYMMETRY_ANALYSIS_FILE)

visualization:
	python $(VISUALIZATION_FILE)
