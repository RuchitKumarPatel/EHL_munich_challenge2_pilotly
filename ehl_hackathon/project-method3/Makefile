SOURCE ?= ../dataset1/export

.PHONY: test reproduce ablation clean

test:
	python -m pytest tests/ -v

reproduce:
	python scripts/reproduce_all.py $(SOURCE)

ablation:
	python scripts/ablation.py $(SOURCE) results/ablation.json

clean:
	rm -rf models/*.json models/*.metadata.json results/*.json results/*.csv results/*.png
