# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container first to leverage Docker layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's source code and MLflow runs
COPY ./src /app/src
COPY ./mlruns /app/mlruns

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Command to run the app when the container launches
# Assumes your FastAPI app instance is named 'app' in 'src/app.py'
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
