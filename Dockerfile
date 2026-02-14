# Use a lightweight Python Linux image

FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies (pandas, xgboost, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Command to run when the container starts
CMD ["python", "main.py"]