FROM node:20-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev --legacy-peer-deps

# Copy application code
COPY . .

EXPOSE 5000

CMD ["node", "server.js"]
