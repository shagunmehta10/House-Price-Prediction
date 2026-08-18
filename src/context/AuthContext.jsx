import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { authApi } from "../api/client";

const AuthContext = createContext(null);


export function AuthProvider({ children }) {

  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);


  // ============================================================
  // CHECK EXISTING FLASK SESSION
  // ============================================================

  useEffect(() => {

    let cancelled = false;

    async function bootstrap() {

      try {

        const data = await authApi.me();

        if (cancelled) return;

        const nextUser =
          data?.user ||
          (data?.authenticated ? data : null);

        setUser(nextUser || null);

      } catch (error) {

        if (!cancelled) {
          setUser(null);
        }

      } finally {

        if (!cancelled) {
          setLoading(false);
        }

      }
    }

    bootstrap();

    return () => {
      cancelled = true;
    };

  }, []);


  // ============================================================
  // LOGIN
  // ============================================================

  const login = async (credentials) => {

    const data = await authApi.login(credentials);

    const nextUser =
      data?.user ||
      data?.data?.user ||
      data?.user_info ||
      null;

    setUser(nextUser);

    return nextUser || data;
  };


  // ============================================================
  // SIGNUP
  // ============================================================

  const signup = async (payload) => {

    const data = await authApi.signup(payload);

    const nextUser =
      data?.user ||
      data?.data?.user ||
      null;

    setUser(nextUser);

    return nextUser || data;
  };


  // ============================================================
  // LOGOUT
  // ============================================================

  const logout = async () => {

    try {
      await authApi.logout();
    } catch (error) {
      console.error("Logout error:", error);
    }

    setUser(null);
  };


  // ============================================================
  // UPDATE USER
  // ============================================================

  const updateUser = (patch) => {

    setUser((previous) => ({
      ...(previous || {}),
      ...patch,
    }));

  };


  const value = useMemo(
    () => ({
      user,

      token: null,

      isAuthenticated: Boolean(user),

      loading,

      login,

      signup,

      logout,

      updateUser,
    }),
    [user, loading]
  );


  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}


export function useAuth() {

  const context = useContext(AuthContext);

  if (!context) {
    throw new Error(
      "useAuth must be used within an AuthProvider"
    );
  }

  return context;
}
