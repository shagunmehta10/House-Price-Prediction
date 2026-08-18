import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5001";

const client = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error(
      "HouseAI API error:",
      error?.response?.status,
      error?.response?.data || error.message
    );

    return Promise.reject(error);
  }
);

// ============================================================
// AUTH
// ============================================================

export const authApi = {
  login: (payload) =>
    client
      .post("/auth/login", payload)
      .then((response) => response.data),

  signup: (payload) =>
    client
      .post("/auth/register", payload)
      .then((response) => response.data),

  logout: () =>
    client
      .post("/auth/logout")
      .then((response) => response.data),

  me: () =>
    client
      .get("/auth/me")
      .then((response) => response.data),
};

// ============================================================
// LOCATIONS
// ============================================================

export const locationsApi = {
  getAll: () =>
    client
      .get("/locations")
      .then((response) => response.data),
};

// ============================================================
// PREDICTION
// ============================================================

export const predictionApi = {
  predict: (payload) =>
    client
      .post("/predict", payload)
      .then((response) => response.data),

  history: () =>
    client
      .get("/history")
      .then((response) => response.data),

  deleteHistoryItem: (id) =>
    client
      .delete(`/history/${id}`)
      .then((response) => response.data),
};

// ============================================================
// FAVORITES
// ============================================================

export const favoritesApi = {
  list: () =>
    client
      .get("/favorites")
      .then((response) => response.data),

  add: (payload) =>
    client
      .post("/favorites", payload)
      .then((response) => response.data),

  remove: (id) =>
    client
      .delete(`/favorites/${id}`)
      .then((response) => response.data),
};

// ============================================================
// DASHBOARD
// ============================================================

export const dashboardApi = {
  summary: () =>
    client
      .get("/dashboard")
      .then((response) => response.data),
};

// ============================================================
// PROFILE
// ============================================================

export const profileApi = {
  get: () =>
    client
      .get("/profile")
      .then((response) => response.data),

  update: (payload) =>
    client
      .put("/profile", payload)
      .then((response) => response.data),
};

export default client;