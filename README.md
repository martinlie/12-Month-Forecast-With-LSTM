# Build image

    docker build -f Dockerfile -t python-science:latest .

# Test image (optional)

    docker run --rm -i -t python-science:latest bash