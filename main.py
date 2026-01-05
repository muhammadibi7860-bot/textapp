from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import ssl
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime

# ---------- NLTK SETUP ----------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("vader_lexicon")

# ---------- TRAINING EXAMPLE ----------
texts = [
    "This is a very good product. Very much liked it.",
    "This is a terrible product. I'm very disappointed.",
    "Spam message: win a prize!",
    "I absolutely love this new feature! It's fantastic.",
    "Worst experience ever. I regret buying this.",
    "Get 50% off on our latest products! Limited time offer!",
]

labels = [0, 1, 2, 0, 1, 2]  # 0=positive, 1=negative, 2=spam

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

sia = SentimentIntensityAnalyzer()
threshold = 0.2

vader_scores = [sia.polarity_scores(text)["compound"] for text in X_test]
vader_classifications = [0 if score > threshold else 1 for score in vader_scores]

accuracy = accuracy_score(y_test, vader_classifications)
report_vader = classification_report(
    y_test, vader_classifications, zero_division=0
)

# ---------- FASTAPI APP ----------
app = FastAPI()

# ---------- DATABASE SETUP ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@db:5432/nlp_db",  # docker-compose service name
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SentimentLog(Base):
    __tablename__ = "sentiment_logs"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    predicted_sentiment = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# ---------- Pydantic MODELS ----------
class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    id: int
    text: str
    predicted_sentiment: str
    accuracy: float
    classification_report: str
    created_at: datetime


class SentimentLogResponse(BaseModel):
    id: int
    text: str
    predicted_sentiment: str
    accuracy: float
    created_at: datetime

    class Config:
        orm_mode = True


class SentimentUpdateRequest(BaseModel):
    text: str | None = None
    predicted_sentiment: str | None = None


# ---------- ROUTES ----------

@app.get("/")
def home():
    return {"message": "FastAPI server is running!"}


@app.get("/hello")
def say_hello():
    return {"message": "Hello from FastAPI!"}


# ---- POST: create & analyze text (already writing to DB) ----
@app.post("/text-check", response_model=TextResponse)
async def text_check(payload: TextRequest, db: Session = Depends(get_db)):
    text = payload.text

    score = sia.polarity_scores(text)["compound"]
    input_text_classification = 0 if score > threshold else 1
    predicted_sentiment = "Positive" if input_text_classification == 0 else "Negative"

    log = SentimentLog(
        text=text,
        predicted_sentiment=predicted_sentiment,
        accuracy=float(accuracy),
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return TextResponse(
        id=log.id,
        text=text,
        predicted_sentiment=predicted_sentiment,
        accuracy=round(float(accuracy), 2),
        classification_report=report_vader,
        created_at=log.created_at,
    )


# ---- GET: all logs ----
@app.get("/logs", response_model=list[SentimentLogResponse])
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(SentimentLog).order_by(SentimentLog.id.desc()).all()
    return logs


# ---- GET: single log by id ----
@app.get("/logs/{log_id}", response_model=SentimentLogResponse)
def get_log(log_id: int, db: Session = Depends(get_db)):
    log = db.query(SentimentLog).filter(SentimentLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    return log


# ---- PUT: update log text/sentiment ----
@app.put("/logs/{log_id}", response_model=SentimentLogResponse)
def update_log(
    log_id: int,
    payload: SentimentUpdateRequest,
    db: Session = Depends(get_db),
):
    log = db.query(SentimentLog).filter(SentimentLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    # if text changed, recompute sentiment
    if payload.text is not None:
        log.text = payload.text
        score = sia.polarity_scores(payload.text)["compound"]
        cls = 0 if score > threshold else 1
        log.predicted_sentiment = (
            "Positive" if cls == 0 else "Negative"
        )

    if payload.predicted_sentiment is not None:
        log.predicted_sentiment = payload.predicted_sentiment

    db.commit()
    db.refresh(log)
    return log


# ---- DELETE: delete log ----
@app.delete("/logs/{log_id}")
def delete_log(log_id: int, db: Session = Depends(get_db)):
    log = db.query(SentimentLog).filter(SentimentLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    db.delete(log)
    db.commit()
    return {"detail": f"log {log_id} deleted"}