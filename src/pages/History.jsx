import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { predictionApi } from "../api/client";
import { Alert, EmptyState, PageHeader, Spinner } from "../components/ui";

const inr = (value) => {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "—";
  }

  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(n);
};

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState(null);

  const load = async () => {
    setLoading(true);
    setError("");

    try {
      const data = await predictionApi.history();

      const historyItems =
        data?.history ||
        data?.items ||
        data?.predictions ||
        data ||
        [];

      setItems(Array.isArray(historyItems) ? historyItems : []);
    } catch (err) {
      console.error("History API error:", err);

      setError(
        "Couldn't load your prediction history. Confirm your backend is connected."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleDelete = async (id) => {
    setDeletingId(id);

    try {
      await predictionApi.deleteHistoryItem(id);

      setItems((prev) =>
        prev.filter((item) => (item.id ?? item._id) !== id)
      );
    } catch (err) {
      console.error("Delete history error:", err);
      setError("Couldn't remove that entry. Please try again.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="container-x py-14">
      <PageHeader
        eyebrow="Your Activity"
        title="Prediction history"
        sub="Every estimate you've run with HouseAI, most recent first."
      />

      {error && (
        <div className="mb-6">
          <Alert>{error}</Alert>
        </div>
      )}

      {loading ? (
        <div className="grid place-items-center py-24">
          <Spinner className="h-8 w-8" />
        </div>
      ) : items.length === 0 ? (
        <EmptyState
          title="No predictions yet"
          body="Once you run a prediction, it'll show up here so you can revisit it any time."
          action={
            <NavLink to="/predict" className="btn-primary mt-2">
              Run your first prediction →
            </NavLink>
          }
        />
      ) : (
        <div className="overflow-hidden rounded-2xl border border-ink-950/10">
          <table className="w-full text-left text-sm">
            <thead className="bg-ink-950 text-white">
              <tr>
                <th className="px-5 py-3.5 font-semibold">Location</th>
                <th className="px-5 py-3.5 font-semibold">Area</th>
                <th className="px-5 py-3.5 font-semibold">BHK</th>
                <th className="px-5 py-3.5 font-semibold">
                  Predicted Value
                </th>
                <th className="px-5 py-3.5 font-semibold">Date</th>
                <th className="px-5 py-3.5 font-semibold text-right">
                  Action
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-ink-950/10 bg-white">
              {items.map((item, index) => {
                const id =
                  item.id ??
                  item._id ??
                  item.prediction_id ??
                  `history-${index}`;

                // Support the actual backend fields as well as
                // the older frontend field names.
                const location =
                  item.location ??
                  item.input?.location ??
                  "—";

                const area =
                  item.total_sqft ??
                  item.area ??
                  item.input?.total_sqft ??
                  item.input?.area;

                const bhk =
                  item.bhk ??
                  item.bedrooms ??
                  item.input?.bhk ??
                  item.input?.bedrooms;

                const predictedPrice =
                  item.price_inr ??
                  item.prediction ??
                  item.predicted_price ??
                  item.price ??
                  item.input?.price_inr ??
                  item.input?.prediction;

                const date =
                  item.created_at ??
                  item.createdAt ??
                  item.date ??
                  item.timestamp;

                return (
                  <tr key={id}>
                    <td className="px-5 py-4 font-medium text-ink-950">
                      {location}
                    </td>

                    <td className="px-5 py-4 text-ink-700">
                      {area !== undefined && area !== null && area !== ""
                        ? `${Number(area).toLocaleString("en-IN")} sq.ft.`
                        : "—"}
                    </td>

                    <td className="px-5 py-4 text-ink-700">
                      {bhk !== undefined && bhk !== null && bhk !== ""
                        ? bhk
                        : "—"}
                    </td>

                    <td className="px-5 py-4 font-semibold text-ink-950">
                      {inr(predictedPrice)}
                    </td>

                    <td className="px-5 py-4 text-ink-600">
                      {date
                        ? new Date(date).toLocaleDateString("en-IN")
                        : "—"}
                    </td>

                    <td className="px-5 py-4 text-right">
                      <button
                        onClick={() => handleDelete(id)}
                        disabled={deletingId === id}
                        className="text-sm font-semibold text-red-600 hover:text-red-700 disabled:opacity-50"
                      >
                        {deletingId === id ? "Removing…" : "Remove"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
