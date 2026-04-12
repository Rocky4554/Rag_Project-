# LiveKit Connectivity Fix: "Failed to Retrieve Region Info"

## Problem
In production (Render), both the Conversational AI and Interview agents failed to connect to LiveKit with the following error:
`engine: signal failure: failed to retrieve region info: error sending request for url (https://raunak-b7fpljji.livekit.cloud/settings/regions)`

This error prevented the agents from joining the room, resulting in no voice interactions.

## Root Cause
The agents use the `@livekit/rtc-node` package, which relies on a **native Rust binary** for WebRTC and signaling. This binary has its own TLS (SSL) stack separate from Node.js.

The production Docker image (`node:20-slim`) is a minimal image that **does not include system CA certificates** by default. Consequently, the native Rust binary could not verify the SSL certificate of the LiveKit Cloud endpoint when trying to fetch region settings, leading to a "signal failure."

Note: `NODE_TLS_REJECT_UNAUTHORIZED=0` only affects the Node.js HTTPS module and is ignored by the native Rust binary.

## Solution
The fix involved updating the `Dockerfile` to install the necessary SSL certificates and configure the environment for the native binary.

### 1. Install CA Certificates
Added the following to the `Dockerfile`:
```dockerfile
# Install CA certificates + OpenSSL for native TLS (@livekit/rtc-node Rust binary)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates openssl && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*
```

### 2. Configure Environment Variables
Added environment variables to point the native Rust TLS stack to the system certificates:
```dockerfile
# Point native Rust TLS to system CA certs
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
```

## Verification
After deploying these changes to Render:
1. The container successfully initialized the SSL bundle.
2. The `Room.connect()` call succeeded.
3. Logs verified active audio transmission (`AudioPublisher playback start`) and user-to-agent transcriptions (`🎤 User said`).
