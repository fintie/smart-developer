import { RANKING_PROFILES, STRATEGIES, type RankingProfile } from "../lib/strategies";

type Props = {
  strategy: string;
  queryText: string;
  locality: string;
  topK: number;
  rankingProfile: RankingProfile;
  loading: boolean;
  onStrategyChange: (value: string) => void;
  onQueryTextChange: (value: string) => void;
  onLocalityChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onRankingProfileChange: (value: RankingProfile) => void;
  onSearch: () => void;
};

export function SearchPanel({
  strategy,
  queryText,
  locality,
  topK,
  rankingProfile,
  loading,
  onStrategyChange,
  onQueryTextChange,
  onLocalityChange,
  onTopKChange,
  onRankingProfileChange,
  onSearch,
}: Props) {
  const selectedRankingProfile =
    RANKING_PROFILES.find((profile) => profile.value === rankingProfile) ??
    RANKING_PROFILES[0];

  return (
    <aside className="panel search-panel">
      <h2>Search Criteria</h2>

      <label>
        Strategy
        <select
          value={strategy}
          onChange={(event) => onStrategyChange(event.target.value)}
        >
          {STRATEGIES.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
      </label>

      <label>
        Locality filter
        <input
          value={locality}
          onChange={(event) => onLocalityChange(event.target.value)}
          placeholder="e.g. WOLLI CREEK, WAITARA, GYMEA BAY"
        />
      </label>

      <label>
        Ranking Profile
        <select
          value={rankingProfile}
          onChange={(event) =>
            onRankingProfileChange(event.target.value as RankingProfile)
          }
        >
          {RANKING_PROFILES.map((profile) => (
            <option key={profile.value} value={profile.value}>
              {profile.label}
            </option>
          ))}
        </select>
      </label>

      <div className="profile-note">
        <strong>{selectedRankingProfile.label}:</strong>{" "}
        {selectedRankingProfile.description}
      </div>

      <label>
        Top K
        <input
          type="number"
          min={1}
          max={20}
          value={topK}
          onChange={(event) => onTopKChange(Number(event.target.value))}
        />
      </label>

      <label>
        Query text
        <textarea
          value={queryText}
          onChange={(event) => onQueryTextChange(event.target.value)}
          rows={7}
        />
      </label>

      <button className="primary-button" onClick={onSearch} disabled={loading}>
        {loading ? "Searching..." : "Find Sites"}
      </button>

      <div className="demo-note">
        <strong>Current mode:</strong>
        <br />
        Two-tower retrieval + DCN reranking + NSW policy RAG + ML market value
        model + development cost and cost-efficiency scoring.
      </div>
    </aside>
  );
}
