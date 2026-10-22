from sqlalchemy import select

from app.database import SessionLocal
from app.models import Violation


def main():
    with SessionLocal() as db:
        test_violation = Violation(
            number_plate="TEST-1234",
            helmet_detected=False,
            detection_source="mock",
            review_status="pending",
            evidence_path=None,
        )

        db.add(test_violation)
        db.commit()
        db.refresh(test_violation)

        print("Inserted test record:")
        print(f"ID: {test_violation.id}")
        print(f"Number plate: {test_violation.number_plate}")
        print(f"Detection source: {test_violation.detection_source}")
        print(f"Review status: {test_violation.review_status}")

        saved_record = db.scalar(
            select(Violation).where(
                Violation.id == test_violation.id
            )
        )

        print("\nRetrieved record:")
        print(saved_record.number_plate)
        print(saved_record.review_status)


if __name__ == "__main__":
    main()