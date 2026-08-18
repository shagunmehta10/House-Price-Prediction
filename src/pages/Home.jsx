import { NavLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const steps = [
  {
    n: "01",
    title: "Predict",
    body: "Enter a property's area, bedrooms, bathrooms and location to get an instant ML-backed price estimate.",
  },
  {
    n: "02",
    title: "Dashboard",
    body: "See trends, averages and a live snapshot of every estimate you've run across India's housing market.",
  },
  {
    n: "03",
    title: "Save & Compare",
    body: "Bookmark listings as favorites, revisit your history, and compare properties side by side.",
  },
];

const fields = [
  { label: "Property Area", unit: "Sq. Ft." },
  { label: "Bedrooms", unit: "BHK" },
  { label: "Bathrooms", unit: "Bath" },
  { label: "Location", unit: "India" },
];

export default function Home() {
  const { isAuthenticated } = useAuth();

  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden bg-ink-950">
        <div
          className="pointer-events-none absolute inset-0 opacity-60"
          style={{
            background:
              "radial-gradient(700px 400px at 85% 30%, rgba(245,183,0,0.18), transparent 70%)",
          }}
        />
        <div className="container-x relative grid gap-14 py-20 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:py-28">
          <div>
            <span className="eyebrow">AI Powered Real Estate</span>
            <h1 className="mt-6 font-display text-4xl leading-[1.08] text-white sm:text-5xl lg:text-6xl">
              Predict your dream home&apos;s <span className="text-gold-500">value.</span>
            </h1>
            <p className="mt-6 max-w-xl text-base text-white/70 sm:text-lg">
              Welcome to HouseAI — use machine learning to estimate property prices and explore
              housing opportunities across India.
            </p>
            <div className="mt-9 flex flex-wrap gap-4">
              <NavLink to={isAuthenticated ? "/predict" : "/signup"} className="btn-primary">
                Start Prediction <span aria-hidden>→</span>
              </NavLink>
              {!isAuthenticated && (
                <NavLink to="/signup" className="btn-outline">
                  Create Account
                </NavLink>
              )}
            </div>
          </div>

          <div className="rounded-2xl border-t-4 border-gold-500 bg-white p-7 shadow-gold">
            <p className="text-xs font-bold uppercase tracking-widest text-ink-600">
              Smart Property Intelligence
            </p>
            <div className="mt-4 flex items-center gap-2">
              <span className="text-2xl font-display text-ink-950">₹</span>
              <span className="text-2xl font-display text-ink-950">AI</span>
            </div>
            <p className="mt-2 text-sm text-ink-600">Machine-learning powered property estimation</p>
            <div className="mt-5 divide-y divide-ink-950/10 border-t border-ink-950/10">
              {fields.map((f) => (
                <div key={f.label} className="flex items-center justify-between py-3.5 text-sm">
                  <span className="text-ink-800">{f.label}</span>
                  <span className="font-semibold text-ink-950">{f.unit}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="bg-white py-24">
        <div className="container-x">
          <div className="mx-auto max-w-2xl text-center">
            <span className="eyebrow border-gold-600 text-gold-600">HouseAI Platform</span>
            <h2 className="mt-4 font-display text-3xl text-ink-950 sm:text-4xl">
              Everything in one place.
            </h2>
            <p className="mt-4 text-ink-600">
              Your complete HouseAI property intelligence platform for predicting, saving and
              managing property estimates.
            </p>
          </div>

          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {steps.map((s) => (
              <div key={s.n} className="card">
                <span className="inline-flex h-9 items-center rounded-md bg-gold-500 px-3 text-sm font-bold text-ink-950">
                  {s.n}
                </span>
                <h3 className="mt-5 text-xl font-bold text-ink-950">{s.title}</h3>
                <p className="mt-2 text-sm text-ink-600">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-ink-950 py-20">
        <div className="container-x flex flex-col items-center gap-6 text-center">
          <h2 className="max-w-xl font-display text-3xl text-white sm:text-4xl">
            Ready to see what your property is really worth?
          </h2>
          <NavLink to={isAuthenticated ? "/predict" : "/signup"} className="btn-primary">
            Start Prediction <span aria-hidden>→</span>
          </NavLink>
        </div>
      </section>
    </div>
  );
}
