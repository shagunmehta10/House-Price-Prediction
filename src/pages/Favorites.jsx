import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { favoritesApi } from "../api/client";
import { Alert, EmptyState, PageHeader, Spinner } from "../components/ui";

const inr = (v) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(v);

export default function Favorites() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [removingId, setRemovingId] = useState(null);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await favoritesApi.list();
      setItems(data?.favorites || data?.items || data || []);
    } catch {
      setError("Couldn't load your favorites. Confirm your backend is connected.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleRemove = async (id) => {
    setRemovingId(id);
    try {
      await favoritesApi.remove(id);
      setItems((prev) => prev.filter((it) => (it.id ?? it._id) !== id));
    } catch {
      setError("Couldn't remove that favorite. Please try again.");
    } finally {
      setRemovingId(null);
    }
  };

  return (
    <div className="container-x py-14">
      <PageHeader
        eyebrow="Saved Properties"
        title="Your favorites"
        sub="Properties and estimates you've bookmarked to compare later."
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
          title="No favorites yet"
          body="Save a prediction to compare it against others later â€” it'll show up here."
          action={
            <NavLink to="/predict" className="btn-primary mt-2">
              Make a prediction â†’
            </NavLink>
          }
        />
      ) : (
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => {
            const id = item.id ?? item._id;
            return (
              <div key={id} className="card flex flex-col border-t-4 border-gold-500">
                <p className="text-xs font-bold uppercase tracking-widest text-ink-600">
                  {item.property_type || "Property"}
                </p>
                <h3 className="mt-2 text-lg font-bold text-ink-950">{item.location}</h3>
                <dl className="mt-4 space-y-1.5 text-sm text-ink-700">
                  <div className="flex justify-between">
                    <dt>Area</dt>
                    <dd className="font-medium text-ink-950">{item.area} sq.ft.</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Bedrooms</dt>
                    <dd className="font-medium text-ink-950">{item.bedrooms} BHK</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt>Bathrooms</dt>
                    <dd className="font-medium text-ink-950">{item.bathrooms}</dd>
                  </div>
                </dl>
                <p className="mt-5 font-display text-2xl text-ink-950">
                  {inr(item.predicted_price ?? item.price)}
                </p>
                <button
                  onClick={() => handleRemove(id)}
                  disabled={removingId === id}
                  className="btn-outline-dark mt-6 disabled:opacity-50"
                >
                  {removingId === id ? "Removingâ€¦" : "Remove from Favorites"}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

