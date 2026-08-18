import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import {
  dashboardApi,
  predictionApi,
  favoritesApi,
} from "../api/client";
import {
  Alert,
  EmptyState,
  PageHeader,
  Spinner,
} from "../components/ui";

const formatINR = (value) => {
  const n = Number(value);

  if (!Number.isFinite(n) || n === 0) {
    return "₹0";
  }

  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(n);
};

const formatCompactINR = (value) => {
  const n = Number(value);

  if (!Number.isFinite(n)) return "₹0";

  if (n >= 10000000) {
    return `₹${(n / 10000000).toFixed(2)}Cr`;
  }

  if (n >= 100000) {
    return `₹${(n / 100000).toFixed(2)}L`;
  }

  return formatINR(n);
};

export default function Dashboard() {
  const [stats, setStats] = useState({
    predictions: 0,
    favorites: 0,
    average_price: 0,
  });

  const [history, setHistory] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadDashboard = async () => {
      setLoading(true);
      setError("");

      try {
        const [dashboardResult, historyResult, favoritesResult] =
          await Promise.all([
            dashboardApi.summary(),
            predictionApi.history(),
            favoritesApi.list(),
          ]);

        const dashboardStats = dashboardResult?.stats || {};

        const historyItems = Array.isArray(historyResult?.history)
          ? historyResult.history
          : Array.isArray(historyResult?.items)
          ? historyResult.items
          : Array.isArray(historyResult)
          ? historyResult
          : [];

        const favoriteItems = Array.isArray(favoritesResult?.favorites)
          ? favoritesResult.favorites
          : Array.isArray(favoritesResult?.items)
          ? favoritesResult.items
          : Array.isArray(favoritesResult)
          ? favoritesResult
          : [];

        setStats({
          predictions: Number(
            dashboardStats.predictions ?? historyItems.length ?? 0
          ),
          favorites: Number(
            dashboardStats.favorites ?? favoriteItems.length ?? 0
          ),
          average_price: Number(
            dashboardStats.average_price ?? 0
          ),
        });

        setHistory(historyItems);
        setFavorites(favoriteItems);
      } catch (err) {
        console.error("Dashboard API error:", err);

        setError(
          err?.response?.data?.message ||
            "Unable to load your dashboard data."
        );
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

  const locations = useMemo(() => {
    const unique = new Set();

    history.forEach((item) => {
      const location =
        item.location ||
        item.input?.location;

      if (location) {
        unique.add(location);
      }
    });

    return Array.from(unique);
  }, [history]);

  const locationStats = useMemo(() => {
    const map = {};

    history.forEach((item) => {
      const location =
        item.location ||
        item.input?.location;

      if (!location) return;

      if (!map[location]) {
        map[location] = {
          location,
          count: 0,
          totalPrice: 0,
        };
      }

      const price = Number(
        item.price_inr ??
          item.predicted_price ??
          item.prediction ??
          item.price ??
          0
      );

      map[location].count += 1;

      if (Number.isFinite(price)) {
        map[location].totalPrice += price;
      }
    });

    return Object.values(map)
      .map((item) => ({
        ...item,
        averagePrice:
          item.count > 0
            ? item.totalPrice / item.count
            : 0,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [history]);

  const monthlyTrend = useMemo(() => {
    const map = {};

    history.forEach((item) => {
      const rawDate =
        item.created_at ||
        item.createdAt ||
        item.date ||
        item.timestamp;

      if (!rawDate) return;

      const date = new Date(rawDate);

      if (Number.isNaN(date.getTime())) return;

      const key = `${date.getFullYear()}-${String(
        date.getMonth() + 1
      ).padStart(2, "0")}`;

      if (!map[key]) {
        map[key] = {
          key,
          label: date.toLocaleDateString("en-IN", {
            month: "short",
            year: "numeric",
          }),
          count: 0,
          total: 0,
        };
      }

      const price = Number(
        item.price_inr ??
          item.predicted_price ??
          item.prediction ??
          item.price ??
          0
      );

      map[key].count += 1;

      if (Number.isFinite(price)) {
        map[key].total += price;
      }
    });

    return Object.values(map)
      .map((item) => ({
        ...item,
        average:
          item.count > 0
            ? item.total / item.count
            : 0,
      }))
      .sort((a, b) => a.key.localeCompare(b.key))
      .slice(-6);
  }, [history]);

  const maxLocationCount = Math.max(
    ...locationStats.map((item) => item.count),
    1
  );

  const maxTrendPrice = Math.max(
    ...monthlyTrend.map((item) => item.average),
    1
  );

  if (loading) {
    return (
      <div className="container-x py-14">
        <PageHeader
          eyebrow="Overview"
          title="Your dashboard"
          sub="A live snapshot of your predictions, trends and saved properties."
        />

        <div className="grid place-items-center py-24">
          <Spinner className="h-10 w-10" />
        </div>
      </div>
    );
  }

  return (
    <div className="container-x py-14">
      <PageHeader
        eyebrow="Overview"
        title="Your dashboard"
        sub="A live snapshot of your predictions, trends and saved properties."
      />

      {error && (
        <div className="mb-6">
          <Alert>{error}</Alert>
        </div>
      )}

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card">
          <p className="text-xs font-bold uppercase tracking-widest text-ink-600">
            Total Predictions
          </p>

          <p className="mt-4 font-display text-4xl font-bold text-ink-950">
            {stats.predictions}
          </p>

          <p className="mt-2 text-sm text-ink-600">
            Estimates you've run
          </p>
        </div>

        <div className="card">
          <p className="text-xs font-bold uppercase tracking-widest text-ink-600">
            Average Estimate
          </p>

          <p className="mt-4 font-display text-3xl font-bold text-ink-950">
            {stats.average_price > 0
              ? formatCompactINR(stats.average_price)
              : "₹0"}
          </p>

          <p className="mt-2 text-sm text-ink-600">
            Average predicted property value
          </p>
        </div>

        <div className="card">
          <p className="text-xs font-bold uppercase tracking-widest text-ink-600">
            Saved Favorites
          </p>

          <p className="mt-4 font-display text-4xl font-bold text-ink-950">
            {stats.favorites}
          </p>

          <p className="mt-2 text-sm text-ink-600">
            Properties saved
          </p>
        </div>

        <div className="card">
          <p className="text-xs font-bold uppercase tracking-widest text-ink-600">
            Locations Covered
          </p>

          <p className="mt-4 font-display text-4xl font-bold text-ink-950">
            {locations.length}
          </p>

          <p className="mt-2 text-sm text-ink-600">
            Locations you've predicted
          </p>
        </div>
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <div className="card">
          <div className="mb-8">
            <h2 className="text-xl font-bold text-ink-950">
              Average price trend
            </h2>

            <p className="mt-1 text-sm text-ink-600">
              Your actual prediction history
            </p>
          </div>

          {monthlyTrend.length === 0 ? (
            <div className="py-16 text-center text-sm text-ink-600">
              No prediction data available yet.
            </div>
          ) : (
            <div className="space-y-5">
              {monthlyTrend.map((item) => (
                <div key={item.key}>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="font-semibold text-ink-950">
                      {item.label}
                    </span>

                    <span className="font-semibold text-ink-950">
                      {formatCompactINR(item.average)}
                    </span>
                  </div>

                  <div className="h-3 overflow-hidden rounded-full bg-ink-950/10">
                    <div
                      className="h-full rounded-full bg-gold-500"
                      style={{
                        width: `${Math.max(
                          4,
                          (item.average / maxTrendPrice) * 100
                        )}%`,
                      }}
                    />
                  </div>

                  <p className="mt-1 text-xs text-ink-600">
                    {item.count} prediction
                    {item.count === 1 ? "" : "s"}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="mb-8">
            <h2 className="text-xl font-bold text-ink-950">
              Predictions by location
            </h2>

            <p className="mt-1 text-sm text-ink-600">
              Real locations from your prediction history
            </p>
          </div>

          {locationStats.length === 0 ? (
            <div className="py-16 text-center text-sm text-ink-600">
              No location data available yet.
            </div>
          ) : (
            <div className="space-y-5">
              {locationStats.map((item) => (
                <div key={item.location}>
                  <div className="mb-2 flex items-center justify-between gap-4">
                    <span className="truncate text-sm font-semibold text-ink-950">
                      {item.location}
                    </span>

                    <span className="shrink-0 text-sm font-bold text-ink-950">
                      {item.count}
                    </span>
                  </div>

                  <div className="h-3 overflow-hidden rounded-full bg-ink-950/10">
                    <div
                      className="h-full rounded-full bg-ink-950"
                      style={{
                        width: `${Math.max(
                          5,
                          (item.count / maxLocationCount) * 100
                        )}%`,
                      }}
                    />
                  </div>

                  <p className="mt-1 text-xs text-ink-600">
                    Average: {formatCompactINR(item.averagePrice)}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-ink-950">
                Recent predictions
              </h2>

              <p className="mt-1 text-sm text-ink-600">
                Your latest property estimates
              </p>
            </div>

            <NavLink
              to="/history"
              className="text-sm font-bold text-gold-600 hover:text-gold-700"
            >
              View all →
            </NavLink>
          </div>

          {history.length === 0 ? (
            <EmptyState
              title="No predictions yet"
              body="Run your first property prediction to see it here."
              action={
                <NavLink
                  to="/predict"
                  className="btn-primary mt-2"
                >
                  Run Prediction
                </NavLink>
              }
            />
          ) : (
            <div className="mt-6 space-y-3">
              {history.slice(0, 5).map((item, index) => {
                const price = Number(
                  item.price_inr ??
                    item.predicted_price ??
                    item.prediction ??
                    item.price ??
                    0
                );

                return (
                  <div
                    key={item.id ?? item._id ?? index}
                    className="flex items-center justify-between rounded-xl border border-ink-950/10 p-4"
                  >
                    <div>
                      <p className="font-semibold text-ink-950">
                        {item.location || "Unknown location"}
                      </p>

                      <p className="mt-1 text-xs text-ink-600">
                        {item.total_sqft
                          ? `${Number(item.total_sqft).toLocaleString(
                              "en-IN"
                            )} sq.ft.`
                          : "—"}{" "}
                        ·{" "}
                        {item.bhk ?? "—"} BHK
                      </p>
                    </div>

                    <p className="font-bold text-ink-950">
                      {formatCompactINR(price)}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-ink-950">
                Saved favorites
              </h2>

              <p className="mt-1 text-sm text-ink-600">
                Properties you've saved for later
              </p>
            </div>

            <NavLink
              to="/favorites"
              className="text-sm font-bold text-gold-600 hover:text-gold-700"
            >
              View all →
            </NavLink>
          </div>

          {favorites.length === 0 ? (
            <div className="py-16 text-center">
              <p className="text-sm text-ink-600">
                No saved favorites yet.
              </p>

              <NavLink
                to="/predict"
                className="btn-primary mt-5 inline-flex"
              >
                Make a Prediction
              </NavLink>
            </div>
          ) : (
            <div className="mt-6 space-y-3">
              {favorites.slice(0, 5).map((item, index) => {
                const price = Number(
                  item.predicted_price ??
                    item.price_inr ??
                    item.prediction ??
                    item.price ??
                    0
                );

                return (
                  <div
                    key={item.id ?? item._id ?? index}
                    className="rounded-xl border border-ink-950/10 p-4"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <p className="font-semibold text-ink-950">
                        {item.location || "Unknown location"}
                      </p>

                      <p className="font-bold text-ink-950">
                        {formatCompactINR(price)}
                      </p>
                    </div>

                    <p className="mt-1 text-xs text-ink-600">
                      {item.total_sqft
                        ? `${Number(item.total_sqft).toLocaleString(
                            "en-IN"
                          )} sq.ft.`
                        : "—"}{" "}
                      ·{" "}
                      {item.bhk ?? "—"} BHK
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
