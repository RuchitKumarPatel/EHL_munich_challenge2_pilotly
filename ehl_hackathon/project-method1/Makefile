PYTHON ?= python
EXPORT ?= ../viktor-tumai-starter/viktor-tumai-starter/export

prepare:
	$(PYTHON) scripts/prepare_data.py $(EXPORT)

train:
	$(PYTHON) scripts/train_router.py $(EXPORT)

evaluate:
	$(PYTHON) scripts/evaluate_router.py $(EXPORT)

pareto:
	$(PYTHON) scripts/generate_pareto.py $(EXPORT)

test:
	$(PYTHON) -m unittest discover -s tests -v

reproduce:
	$(PYTHON) scripts/reproduce_all.py $(EXPORT)

