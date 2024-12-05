# start from python base image
FROM python:3.10-slim

# change working directory on docker image
WORKDIR /code

# add requirements file to docker image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir -r /code/requirements.txt

# add python code to docker image
COPY ./ /code/

# specify default commands
CMD ["fastapi", "run", "main.py", "--port", "8080"]