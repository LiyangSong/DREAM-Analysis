.PHONY: all install download

DATASET_URL=https://snd.se/en/catalogue/download/dataset/snd1156-1/1?principal=his.se&filename=DREAM2020_SND1156-001-V1.0.zip
DATASET_ZIP_FILE=dataset.zip
DATASET_DIR_PATH=dataset

all: download analysis visualization clean

download:
	wget --progress=bar:force $(DATASET_URL) -O $(DATASET_ZIP_FILE)
	mkdir -p $(DATASET_DIR_PATH)
	unzip -q $(DATASET_ZIP_FILE) -d $(DATASET_DIR_PATH)
	rm -f $(DATASET_ZIP_FILE)

analysis:

visualization:

clean:



