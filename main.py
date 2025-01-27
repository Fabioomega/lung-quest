from fastapi import FastAPI
from pydantic import BaseModel
from disease_classifier import XRayClassifier, ModelOutput
from risk_assessor import *
from typing import List, Dict

ClassificationOutput = ModelOutput | Dict[str, float]


class ImageRequest(BaseModel):
    img: str


class AssessmentResponse(BaseModel):
    biological_age: float


class EvaluationResponse(BaseModel):
    classification: ClassificationOutput
    biological_age: float


app = FastAPI()
disease_model = XRayClassifier(device="cuda")
risk_model = XRayMortalityAssessor(device="cuda")


@app.post("/classify")
def no_opt_disease_classifier(
    img: ImageRequest, raw: bool = False
) -> ClassificationOutput:
    if raw:
        return disease_model.raw_batch_all_classify_by_b64([img.img])[0]
    return disease_model.batch_all_classify_by_b64([img.img])[0]


@app.get("/pathologies")
def pathologies() -> List[str]:
    return disease_model.pathologies


@app.post("/assess-risk")
def no_opt_risk_assessment(img: ImageRequest) -> AssessmentResponse:
    return {"biological_age": risk_model.batch_all_classify_by_b64([img.img])[0]}


@app.post("/evaluate")
def no_opt_general_evaluation(
    img: ImageRequest, raw: bool = False
) -> EvaluationResponse:
    output = {}
    if raw:
        output["classification"] = disease_model.raw_batch_all_classify_by_b64(
            [img.img]
        )[0]
    else:
        output["classification"] = disease_model.batch_all_classify_by_b64([img.img])[0]

    output["biological_age"] = risk_model.batch_all_classify_by_b64([img.img])[0]
    return output
