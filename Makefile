.PHONY: setup run ci test clean

CORES ?= 2

setup:
	pip install -r requirements.txt

run:
	snakemake --cores $(CORES) --configfile config/config.yaml --config _config_path=config/config.yaml

ci:
	snakemake --cores $(CORES) --configfile config/ci.yaml --config _config_path=config/ci.yaml

test:
	pytest -v tests/

clean:
	rm -rf results/*.tsv figures/*.png data/raw/*.gz data/raw/*.tsv data/raw/*.txt data/raw/*.json
