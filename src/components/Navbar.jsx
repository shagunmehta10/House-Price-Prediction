import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const loggedOutLinks = [{ to: "/", label: "Home" }];

const loggedInLinks = [
  { to: "/", label: "Home" },
  { to: "/predict", label: "Predict" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/history", label: "History" },
  { to: "/favorites", label: "Favorites" },
  { to: "/profile", label: "Profile" },
];

export default function Navbar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const links = isAuthenticated ? loggedInLinks : loggedOutLinks;

  const handleLogout = async () => {
    await logout();
    setOpen(false);
    navigate("/login");
  };

  return (
    <header className="sticky top-0 z-50 border-b-2 border-gold-500 bg-ink-950">
      <div className="container-x flex h-[72px] items-center justify-between">
        <NavLink to="/" className="text-2xl font-display tracking-tight text-white">
          House<span className="text-gold-500">AI</span>
        </NavLink>

        <nav className="hidden items-center gap-8 lg:flex">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `text-sm font-semibold transition ${
                  isActive ? "text-gold-500" : "text-white/85 hover:text-gold-400"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="hidden items-center gap-4 lg:flex">
          {isAuthenticated ? (
            <button onClick={handleLogout} className="text-sm font-semibold text-white/85 hover:text-gold-400">
              Logout
            </button>
          ) : (
            <>
              <NavLink to="/login" className="text-sm font-semibold text-white/85 hover:text-gold-400">
                Login
              </NavLink>
              <NavLink to="/signup" className="btn-primary !px-5 !py-2.5">
                Create Account
              </NavLink>
            </>
          )}
        </div>

        <button
          className="grid h-10 w-10 place-items-center rounded-lg border border-white/20 text-white lg:hidden"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M3 6h18M3 12h18M3 18h18" />}
          </svg>
        </button>
      </div>

      {open && (
        <div className="border-t border-white/10 bg-ink-950 lg:hidden">
          <div className="container-x flex flex-col gap-1 py-4">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === "/"}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2.5 text-sm font-semibold ${
                    isActive ? "bg-white/5 text-gold-500" : "text-white/85"
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
            <div className="mt-2 flex flex-col gap-2 border-t border-white/10 pt-3">
              {isAuthenticated ? (
                <button onClick={handleLogout} className="btn-outline">
                  Logout
                </button>
              ) : (
                <>
                  <NavLink to="/login" onClick={() => setOpen(false)} className="btn-outline">
                    Login
                  </NavLink>
                  <NavLink to="/signup" onClick={() => setOpen(false)} className="btn-primary">
                    Create Account
                  </NavLink>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
