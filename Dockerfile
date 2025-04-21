# Use a light Python base image
FROM python:3.10-slim

# Set a directory inside the container for our app
WORKDIR /app

# Copy requirements into container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code into the container
COPY . .

# Expose the port Flask runs on (adjust if needed)
EXPOSE 5000

# Set environment variable so Flask knows what to run
ENV FLASK_APP=app.py

# Start the Flask app with Waitress (production WSGI server)
CMD ["waitress-serve", "--host", "0.0.0.0", "--port=5000", "app:app"]
