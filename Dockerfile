FROM python:3.10-slim

# Install system-level dependencies (including libGL)
RUN apt-get update && apt-get install -y libgl1

# Set working directory
WORKDIR /app

# Copy code into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 5000

# Start the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
