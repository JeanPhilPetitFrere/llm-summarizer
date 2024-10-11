# llm_summarizer

<Add short description of the project>


## Installation

### Getting Started

Run the following command from the root of the folder to get your container setup.
It'll start the container and install all the necessary dependencies.

```
docker build . -t llm_summarizer
```

Then use the following command to start the container:

```
docker compose up
```


### Pre-commit

`pre-commit` should be installed thanks to the file `pre-commit` in the `.git/hooks` folder. If pre-commit does not work, you can run `pre-commit install` at the root of the project to create your own `pre-commit` file.
