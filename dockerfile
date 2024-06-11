from python:3.12

# copy dependencies and install

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# copy the rest of the code
COPY . /app
