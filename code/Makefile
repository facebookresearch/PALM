.PHONY: clean fix imports sort cache

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find exp/* -delete
fix:
	black */*.py *.py
sort:
	isort src *.py --wrap-length=1 --combine-as --trailing-comma --use-parentheses
imports:
	find datasets -type f -name "*.py" -not -name "__init__.py" | xargs autoflake -i --remove-all-unused-imports
	find models -type f -name "*.py" -not -name "__init__.py" | xargs autoflake -i --remove-all-unused-imports
	find systems -type f -name "*.py" -not -name "__init__.py" | xargs autoflake -i --remove-all-unused-imports
	find utils -type f -name "*.py" -not -name "__init__.py" | xargs autoflake -i --remove-all-unused-imports
cache:
	find datasets -type d -name "__pycache__" -exec rm -r {} +
	find systems -type d -name "__pycache__" -exec rm -r {} +
	find utils -type d -name "__pycache__" -exec rm -r {} +
	find configs -type d -name "__pycache__" -exec rm -r {} +
	find models -type d -name "__pycache__" -exec rm -r {} +
	find ~/.cache/torch_extensions/ -delete
