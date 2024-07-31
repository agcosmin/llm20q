.PHONY: setup_lint lint format test

setup_lint:
	lintrunner init

lint:
	lintrunner

format:
	lintrunner format

test:
	pytest --verbose ./tests

submission:
	tar --use-compress-program='pigz --fast --recursive' \
		-cf submission.tar.gz \
		--transform "s|agents.py|main.py|" \
		-C ./llm20q/ agents.py \
		-C ../gemma/models/ transformers/2b-it/3

