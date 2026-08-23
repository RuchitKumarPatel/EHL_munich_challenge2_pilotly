SOURCE ?= ../dataset2/export
M3 ?= ../project-method3/models/dataset2_20260823_045709

.PHONY: test reproduce clean
test:
	python -m pytest tests/ -v
reproduce:
	python scripts/reproduce_all.py $(SOURCE) $(M3)
clean:
	rm -rf models/*/ results/*/
