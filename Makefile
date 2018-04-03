default: quicktest

init:
	pip install pipenv --upgrade
	pipenv install --dev --skip-lock

flake8:
	time pipenv run flake8 --ignore E741 lib tests setup.py

test:
	time detox

quicktest:
	time pipenv run pytest -v

publish:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build dist *.egg derplog.egg-info derplog/.__pycache__ checkpoints/*

clean-eggs:
	rm -rf .eggs

.PHONY: clean clean-eggs init flake8 test publish quicktest
