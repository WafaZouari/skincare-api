FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies globally
RUN pip install chromadb langchain langchain-community fastapi uvicorn openpyxl sentence-transformers

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
