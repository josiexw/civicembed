FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run road encoder
CMD [ "python", "src/embedding/road_encoder.py" ]
