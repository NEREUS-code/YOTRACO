test-cov-unit:
	pytest --cov=YOTRACO --cov-report=html tests/ 

# run the command pytest...tests/ then run open htmlcov/index.html to open the htmlcov 
# it's just an html page to show coverage report

uploadtest:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release:
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*