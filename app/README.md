# Assignment 3 web application

Predicts classes 0–3 using the saved A3 preprocessing/softmax pipeline. Run commands from the repository root.

## Local

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

This uses http://localhost:8501. Unknown values are imputed from the fitted training pipeline. The application displays a price band and four class probabilities, not an exact rupee estimate.

## Docker

```bash
docker build -f app/Dockerfile -t car-price-a3:local .
docker run --rm -p 8501:8501 car-price-a3:local
```

The container uses a base path: open http://localhost:8501/a3/. Its health endpoint is `/a3/_stcore/health`.

## Manual course-server deployment

After tests pass and Docker is running, publish a versioned image you control:

```bash
python -m unittest discover -s tests -v
docker login
docker buildx build --platform linux/amd64 -f app/Dockerfile \
  -t lha007/car-price-a3:a3-v1 --push .
ssh st127132@ml.brain.cs.ait.ac.th 'mkdir -p ~/car-price-a3'
scp app/docker-compose.yaml st127132@ml.brain.cs.ait.ac.th:~/car-price-a3/docker-compose.yaml
ssh st127132@ml.brain.cs.ait.ac.th \
  'cd ~/car-price-a3 && A3_IMAGE=lha007/car-price-a3:a3-v1 docker compose pull && A3_IMAGE=lha007/car-price-a3:a3-v1 docker compose up -d --wait --wait-timeout 120'
```

The course VM must have the existing `web` Docker network, Traefik and Docker Compose with `--wait` support. The image must be public or the VM must have Docker Hub pull credentials.

The A3 route is https://st127132.ml.brain.cs.ait.ac.th/a3/. It uses a higher-priority `/a3` path rule while retaining the existing A1/A2 root service. The repository's root `Dockerfile` and `docker-compose.yaml` remain the A2 configuration.

For automated deployment, configure the four `course-vm` GitHub secrets described in the root README, then push to the repository's default branch. Tests must pass before deployment starts. A failed deployment produces a failing job; the workflow does not implement an automatic rollback. Redeploy a previous known-good image tag if needed.
