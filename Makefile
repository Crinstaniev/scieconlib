clean:
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./scieconlib.egg-info

start:
	python setup.py sdist bdist_wheel

upload:
	twine upload dist/*

freeze:
	pip freeze > requirements.txt

docker:
	cp ./common/Dockerfile ./docs/build/html/Dockerfile