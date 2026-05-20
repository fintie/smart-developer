import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import type { SiteResult } from "../api";

type ResultsMapProps = {
  results: SiteResult[];
};

export function ResultsMap({ results }: ResultsMapProps) {
  const points = results.filter(
    (site) =>
      typeof site.latitude === "number" &&
      typeof site.longitude === "number"
  );

  if (points.length === 0) {
    return (
      <div className="empty-state">
        No map-ready coordinates available for these results.
      </div>
    );
  }

  const center: [number, number] = [
    points[0].latitude as number,
    points[0].longitude as number,
  ];

  return (
    <div className="map-card">
      <MapContainer
        center={center}
        zoom={13}
        scrollWheelZoom={false}
        style={{ height: "360px", width: "100%", borderRadius: "18px" }}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {points.map((site, index) => (
          <Marker
            key={`${site.RID ?? index}-${index}`}
            position={[site.latitude as number, site.longitude as number]}
          >
            <Popup>
              <strong>{site.base_site_address || site.address}</strong>
              <br />
              Score: {site.strategy_score ?? "N/A"}
              <br />
              Zoning: {site.primary_zoning_code ?? "N/A"}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}