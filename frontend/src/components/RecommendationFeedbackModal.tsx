type Props = {
  title: string;
  rating: number | null;
  note: string;
  submitting: boolean;
  onRatingChange: (rating: number) => void;
  onNoteChange: (note: string) => void;
  onSubmit: () => void;
  onClose: () => void;
};

export function RecommendationFeedbackModal({
  title,
  rating,
  note,
  submitting,
  onRatingChange,
  onNoteChange,
  onSubmit,
  onClose,
}: Props) {
  return (
    <div className="feedback-modal-backdrop" role="presentation">
      <section
        className="feedback-modal"
        aria-labelledby="recommendation-feedback-title"
        role="dialog"
        aria-modal="true"
      >
        <div className="feedback-modal-header">
          <div>
            <p className="eyebrow">Feedback</p>
            <h2 id="recommendation-feedback-title">{title}</h2>
          </div>
          <button
            className="icon-button"
            aria-label="Close feedback dialog"
            onClick={onClose}
          >
            x
          </button>
        </div>

        <div className="rating-row" aria-label="Recommendation rating">
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              key={value}
              className={
                rating !== null && value <= rating
                  ? "star-button selected"
                  : "star-button"
              }
              aria-label={`Rate ${value} out of 5`}
              onClick={() => onRatingChange(value)}
              type="button"
            >
              ★
            </button>
          ))}
        </div>

        <div className="rating-scale-labels">
          <span>Very unsatisfied</span>
          <span>Very satisfied</span>
        </div>

        <label className="feedback-note-label">
          Note
          <textarea
            value={note}
            onChange={(event) => onNoteChange(event.target.value)}
            rows={3}
            placeholder="Optional"
          />
        </label>

        <div className="feedback-actions">
          <button
            className="feedback-secondary"
            onClick={onClose}
            type="button"
          >
            Not now
          </button>
          <button
            className="primary-button feedback-submit"
            disabled={rating === null || submitting}
            onClick={onSubmit}
            type="button"
          >
            {submitting ? "Submitting..." : "Submit"}
          </button>
        </div>
      </section>
    </div>
  );
}
