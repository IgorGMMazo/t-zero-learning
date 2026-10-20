"""Verify project readability without sending credentials."""
import json
from pathlib import Path
import requests

query = """query Project($project: String!, $entity: String!) {
  project(name: $project, entityName: $entity) { name }
}"""
session = requests.Session()
session.trust_env = False  # Do not silently authenticate using a local .netrc.
response = session.post(
    "https://api.wandb.ai/graphql",
    json={"query": query, "variables": {
        "project": "sac-assignment",
        "entity": "igor_mazo-federal-university-of-goi-s",
    }},
    timeout=30,
)
result = {"status_code": response.status_code, "response": response.json()}
print(json.dumps(result, indent=2))
path = Path(__file__).resolve().parents[1] / "results" / "public_access.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
