import os

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from dotenv import load_dotenv
from app.database import engine
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from fastapi import Depends
from app.database import get_db
from app.models import Violation
load_dotenv()

app = FastAPI(
    title=os.getenv("APP_NAME", "Helmet Detection API"),
    description="Backend for helmet detection and traffic compliance monitoring.",
    version="0.1.0",
)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "helmet-detection-api",
    }


@app.get("/api/health/database", tags=["Health"])
def database_health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "database": "connected",
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "database": "unavailable",
            },
        )

class DetectionResult(BaseModel):
    number_plate: str | None = Field(default=None, max_length=20)
    helmet_detected: bool
    detection_source: str = Field(default="mock", max_length=30)
    evidence_path: str | None = None


@app.post("/api/detections", tags=["Detections"])
def create_detection(
    detection: DetectionResult,
    db: Session = Depends(get_db),
):
    violation = Violation(
        number_plate=detection.number_plate,
        helmet_detected=detection.helmet_detected,
        detection_source=detection.detection_source,
        review_status="pending",
        evidence_path=detection.evidence_path,
    )

    db.add(violation)
    db.commit()
    db.refresh(violation)

    return {
        "id": violation.id,
        "number_plate": violation.number_plate,
        "helmet_detected": violation.helmet_detected,
        "detection_source": violation.detection_source,
        "review_status": violation.review_status,
        "evidence_path": violation.evidence_path,
        "created_at": violation.created_at,
    }
