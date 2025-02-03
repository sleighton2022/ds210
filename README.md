## Tool setup

### tl:dr

1. Install Git
2. Install poetry
3. Install Docker
4. Install minikube 

### Install Git 

Use the following to install Git[Install Git](https://github.com/git-guides/install-git)

### Python Dependency Management: Poetry

We will follow the opinionated structure that is put together by the [python poetry](https://python-poetry.org/) package. You will need to install this tool on your machine by [following the instructions](https://python-poetry.org/docs/#installation). Make sure to **follow all instructions, including adding poetry to your PATH**.

Please refer to the `pyproject.toml` file provided as a baseline. 

### Install Docker 

Use the following to install [Docker](https://www.docker.com/get-started/) 

### Install minikube 

nstall minikube [Docs](https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary+download)

### API development

`FASTApi` will be used as the basis for API development. Please refer to the `main.py` file provided as a baseline.

## Setting up your working directory

```
cd ~
mkdir ds210
cd ds210
git clone git@github.com:sleighton2022/ds210.git
```

## Creating your environment with Poetry

Once you have installed `poetry`, you can use the following command from within your `ds210` folder to create a boilerplate project structure.

```{bash}
poetry new lab3 --name src
```

Poetry will create the following folder structure; your repository should look like the following:

```text
.   ├── Dockerfile
    ├── README.md
    ├── model_pipeline.pkl
    ├── poetry.lock
    ├── pyproject.toml
    ├── infra
    │   ├── deployment-pythonapi.yaml
    │   ├── deployment-redis.yaml
    │   ├── service-prediction.yaml
    │   └── service-redis.yaml
    ├── src
    │   ├── __init__.py
    │   ├── housing_predict.py
    │   └── main.py
    ├── tests
    │   ├── __init__.py
    │   ├── conftest.py
    │   └── test_src.py
    └── trainer
        ├── predict.py
        └── train.py
```

**Do not delete the `.github/` directory,
**

## Working in the environment 

### Running poetry

In the directory with the pyproject.toml
```
poetry shell
```

### Running python

```cd <path_to_main.py>
poetry run uvicorn main:app --reload
```

### Testing the application with python

```
cd <path_to_main>
poetry shell
pytest <path_to_testfile>
```

A baseline pytest file is provided 

### Building docker container

```
cd <path_to_dockerfile>
docker build -t <container_name> .
```

### Running docker

docker run -p 8000:8000 <container_name>

### Running minikube

To run the application in Minikube, first ensure that the Minikube has been installed. You will need to build the docker container in the Minikube environment, so that Minikube can pull the docker image. You will then need to deploy the application container, the redis container, create the redis service, and create the prediction service. To accomplish this, run the following commands:

```
minikube start
eval $(minikube docker-env)
cd ~/ds210/ds210 # replace with your path to docker file
docker build -t lab3 .  
cd ~/ds210/dsl210/infra # replace with your path to infra directory
kubectl apply -f deployment-redis.yaml
kubectl apply -f service-redis.yaml
kubectl apply -f deployment-pythonapi.yaml
kubectl apply -f service-prediction.yaml
```

### Ensuring the application is running 

Once the application has been installed use the following command to determine if the application is up and running

```
kubectl get all 
````

### Cleaning up Minikube

To clean up, use the following commands

``` 
minikube stop
minikube delete
```




