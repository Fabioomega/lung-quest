import requests
import base64


def img_to_b64(img_path: str) -> str:
    with open(img_path, "rb") as file:
        return base64.b64encode(file.read())


img = "test.jpg"

print(
    "Output is:",
    requests.post("localhost:8000/evaluate", json={"img": img_to_b64(img)}),
)
