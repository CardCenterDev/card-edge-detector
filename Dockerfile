# Use Python 3.10 base image
FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
