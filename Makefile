.PHONY: verify paper smoke reproduce

verify:
	python3 scripts/reproduce.py verify

paper:
	python3 scripts/reproduce.py paper

smoke:
	python3 scripts/reproduce.py smoke

reproduce: verify smoke paper

