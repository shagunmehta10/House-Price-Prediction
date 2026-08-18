import { NavLink } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="container-x flex min-h-[60vh] flex-col items-center justify-center text-center">
      <p className="font-display text-6xl text-gold-500">404</p>
      <h1 className="mt-4 text-2xl font-bold text-ink-950">Page not found</h1>
      <p className="mt-2 max-w-sm text-ink-600">
        The page you're looking for doesn't exist or may have moved.
      </p>
      <NavLink to="/" className="btn-primary mt-8">
        Back to Home
      </NavLink>
    </div>
  );
}
