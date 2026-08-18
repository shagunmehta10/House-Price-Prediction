import { useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Alert } from "../components/ui";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [form, setForm] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const from = location.state?.from?.pathname || "/dashboard";

  const handleChange = (e) => setForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(form);
      navigate(from, { replace: true });
    } catch (err) {
      setError(err?.response?.data?.error || err?.response?.data?.message || "Couldn't sign you in. Check your details and try again.");
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
            Welcome back to smarter property decisions.
          </h2>
          <p className="mt-4 max-w-sm text-white/60">
            Sign in to run new predictions, revisit your history and manage your favorites.
          </p>
        </div>
        <p className="text-sm text-white/40">&copy; {new Date().getFullYear()} HouseAI</p>
      </div>

      <div className="flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-sm">
          <h1 className="font-display text-3xl text-ink-950">Log in</h1>
          <p className="mt-2 text-sm text-ink-600">
            New to HouseAI?{" "}
            <NavLink to="/signup" className="font-semibold text-gold-600 hover:text-gold-700">
              Create an account
            </NavLink>
          </p>

          <form onSubmit={handleSubmit} className="mt-8 space-y-5">
            {error && <Alert>{error}</Alert>}
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
                autoComplete="current-password"
                className="field-input"
                placeholder="â€¢â€¢â€¢â€¢â€¢â€¢â€¢â€¢"
                value={form.password}
                onChange={handleChange}
              />
            </div>
            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? "Signing inâ€¦" : "Log in"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

