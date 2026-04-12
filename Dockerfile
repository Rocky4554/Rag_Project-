FROM node:20-slim

# Install CA certificates + OpenSSL for native TLS (@livekit/rtc-node Rust binary)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates openssl && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Point native Rust TLS to system CA certs
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
ENV NODE_TLS_REJECT_UNAUTHORIZED=0

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev --legacy-peer-deps

# Copy application code
COPY . .

EXPOSE 5000

CMD ["node", "server.js"]
