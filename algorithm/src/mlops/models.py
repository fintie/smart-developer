from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RetrievalRequestLog(Base):
    __tablename__ = "retrieval_requests"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True)

    strategy: Mapped[str] = mapped_column(String, nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)

    locality: Mapped[str | None] = mapped_column(String, nullable=True)
    address_contains: Mapped[str | None] = mapped_column(String, nullable=True)

    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    recall_k: Mapped[int] = mapped_column(Integer, nullable=False)

    retrieval_model: Mapped[str] = mapped_column(String, nullable=False)
    reranking_model: Mapped[str | None] = mapped_column(String, nullable=True)
    use_dcn_reranker: Mapped[bool] = mapped_column(Boolean, nullable=False)
    with_explanations: Mapped[bool] = mapped_column(Boolean, nullable=False)
    dedupe_by_address: Mapped[bool] = mapped_column(Boolean, nullable=False)

    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    request_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    planner_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metadata_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class RetrievalResultLog(Base):
    __tablename__ = "retrieval_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    request_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("retrieval_requests.request_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rid: Mapped[str] = mapped_column(String, nullable=False, index=True)
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)

    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_site_address: Mapped[str | None] = mapped_column(Text, nullable=True)

    primary_zoning_code: Mapped[str | None] = mapped_column(String, nullable=True)
    zoning_band: Mapped[str | None] = mapped_column(String, nullable=True)
    lot_size_band: Mapped[str | None] = mapped_column(String, nullable=True)
    constraint_severity_band: Mapped[str | None] = mapped_column(String, nullable=True)
    station_distance_band: Mapped[str | None] = mapped_column(String, nullable=True)

    strategy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    fusion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    dcn_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_rank_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    result_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "rid",
            "rank_position",
            name="uq_retrieval_result_request_rid_rank",
        ),
    )


class UserFeedbackLog(Base):
    __tablename__ = "user_feedback"

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True)

    request_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("retrieval_requests.request_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rid: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    rank_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ReportJobLog(Base):
    __tablename__ = "report_jobs"

    report_id: Mapped[str] = mapped_column(String, primary_key=True)

    request_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("retrieval_requests.request_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(String, nullable=False, index=True)

    explanation_mode: Mapped[str] = mapped_column(String, nullable=False)
    output_markdown_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_pdf_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelRegistryLog(Base):
    __tablename__ = "model_registry"

    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False)

    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    preprocessing_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False, index=True)

    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_card: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)