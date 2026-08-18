import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Alert } from "../components/ui";

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => setForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await signup(form);
      navigate("/dashboard", { replace: true });
    } catch (err) {
      setError(err?.response?.data?.message || "Couldn't create your account. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid min-h-[calc(100vh-72px)] bg-white lg:grid-cols-2">
      <div className="hidden flex-col justify-between bg-ink-950 p-12 lg:flex">
        <NavLink to="/" className="text-2xl font-display text-white">
          House<span className="text-gold-500">AI</span>
        </NavLink>
        <div>
          <h2 className="max-w-sm font-display text-3xl leading-tight text-white">
            Join HouseAI and start predicting property value in seconds.
          </h2>
          <p className="mt-4 max-w-sm text-white/60">
            Free to use. Save predictions, track history and build a shortlist of favorites.
          </p>
        </div>
        <p className="text-sm text-white/40">&copy; {new Date().getFullYear()} HouseAI</p>
      </div>

      <div className="flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-sm">
          <h1 className="font-display text-3xl text-ink-950">Create account</h1>
          <p className="mt-2 text-sm text-ink-600">
            Already have an account?{" "}
            <NavLink to="/login" className="font-semibold text-gold-600 hover:text-gold-700">
              Log in
            </NavLink>
          </p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-5">
            {error && <Alert>{error}</Alert>}
            <div>
              <label className="field-label" htmlFor="name">
                Full name
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                autoComplete="name"
                className="field-input"
                placeholder="Aditi Sharma"
                value={form.name}
                onChange={handleChange}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="email">
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                autoComplete="email"
                className="field-input"
                placeholder="you@example.com"
                value={form.email}
                onChange={handleChange}
              />
            </div>
            <div>
              <label className="field-label" htmlFor="password">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                minLength={6}
                autoComplete="new-password"
                className="field-input"
                placeholder="At least 6 characters"
                value={form.password}
                onChange={handleChange}
              />
            </div>
            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? "Creating account…" : "Create Account"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
