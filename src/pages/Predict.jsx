import { useEffect, useState } from "react";
import { locationsApi, predictionApi } from "../api/client";
import { Alert, PageHeader, Spinner } from "../components/ui";

const initialForm = {
  location: "",
  area: "",
  bedrooms: "",
  bathrooms: "",
  propertyType: "Apartment",
  furnishing: "Semi-Furnished",
  parking: "0",
  age: "",
};

const formatINR = (value) => {
  const n = Number(value);
  if (Number.isNaN(n)) return value;
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(n);
};

export default function Predict() {
  const [form, setForm] = useState(initialForm);
  const [locations, setLocations] = useState([]);
  const [locationsLoading, setLocationsLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const loadLocations = async () => {
      try {
        const data = await locationsApi.getAll();
        const supported = Array.isArray(data.locations)
          ? data.locations
          : [];

        setLocations(supported);
      } catch {
        setError("Couldn't load the available locations. Confirm the backend is running.");
      } finally {
        setLocationsLoading(false);
      }
    };

    loadLocations();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setError("");
    setResult(null);
    setSaved(false);
    setLoading(true);

    try {
      const data = await predictionApi.predict({
        location: form.location,
        total_sqft: Number(form.area),
        bhk: Number(form.bedrooms),
        bath: Number(form.bathrooms),
      });

      console.log("HOUSEAI PREDICTION RESPONSE:", data);

      if (data.success === false) {
        throw new Error(
          data.message || data.error || "Prediction failed"
        );
      }

      const predictedPrice = Number(data.prediction);

      if (!Number.isFinite(predictedPrice)) {
        throw new Error("Backend returned an invalid prediction");
      }

      setResult({
        price: predictedPrice,
      });
    } catch (err) {
      console.error("HOUSEAI PREDICTION ERROR:", err);

      setError(
        err?.message ||
          "Unable to calculate the property price."
      );
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="container-x py-14">
      <PageHeader
        eyebrow="Run a Prediction"
        title="Estimate a property's value"
        sub="Fill in the details below and HouseAI's model will return an instant price estimate."
      />

      <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
        <form onSubmit={handleSubmit} className="card space-y-5">
          <div className="grid gap-5 sm:grid-cols-2">

            <div className="sm:col-span-2">
              <label className="field-label" htmlFor="location">
                Location
              </label>

              <select
                id="location"
                name="location"
                required
                className="field-input"
                value={form.location}
                onChange={handleChange}
                disabled={locationsLoading}
              >
                <option value="">
                  {locationsLoading
                    ? "Loading available locations..."
                    : "Select a supported location"}
                </option>

                {locations.map((location) => (
                  <option key={location} value={location}>
                    {location}
                  </option>
                ))}
              </select>

              {!locationsLoading && locations.length > 0 && (
                <p className="mt-2 text-xs text-ink-950/50">
                  {locations.length.toLocaleString("en-IN")} locations available for prediction
                </p>
              )}
            </div>

            <div>
              <label className="field-label" htmlFor="area">
                Property Area (Sq. Ft.)
              </label>
              <input
                id="area"
                name="area"
                type="number"
                min="1"
                required
                className="field-input"
                placeholder="1200"
                value={form.area}
                onChange={handleChange}
              />
            </div>

            <div>
              <label className="field-label" htmlFor="age">
                Property Age (years)
              </label>
              <input
                id="age"
                name="age"
                type="number"
                min="0"
                className="field-input"
                placeholder="5"
                value={form.age}
                onChange={handleChange}
              />
            </div>

            <div>
              <label className="field-label" htmlFor="bedrooms">
                Bedrooms (BHK)
              </label>
              <input
                id="bedrooms"
                name="bedrooms"
                type="number"
                min="0"
                required
                className="field-input"
                placeholder="3"
                value={form.bedrooms}
                onChange={handleChange}
              />
            </div>

            <div>
              <label className="field-label" htmlFor="bathrooms">
                Bathrooms
              </label>
              <input
                id="bathrooms"
                name="bathrooms"
                type="number"
                min="0"
                required
                className="field-input"
                placeholder="2"
                value={form.bathrooms}
                onChange={handleChange}
              />
            </div>

            <div>
              <label className="field-label" htmlFor="propertyType">
                Property Type
              </label>
              <select
                id="propertyType"
                name="propertyType"
                className="field-input"
                value={form.propertyType}
                onChange={handleChange}
              >
                <option>Apartment</option>
                <option>Independent House</option>
                <option>Villa</option>
                <option>Plot</option>
              </select>
            </div>

            <div>
              <label className="field-label" htmlFor="furnishing">
                Furnishing
              </label>
              <select
                id="furnishing"
                name="furnishing"
                className="field-input"
                value={form.furnishing}
                onChange={handleChange}
              >
                <option>Unfurnished</option>
                <option>Semi-Furnished</option>
                <option>Fully Furnished</option>
              </select>
            </div>

            <div>
              <label className="field-label" htmlFor="parking">
                Parking Spaces
              </label>
              <input
                id="parking"
                name="parking"
                type="number"
                min="0"
                className="field-input"
                placeholder="1"
                value={form.parking}
                onChange={handleChange}
              />
            </div>
          </div>

          {error && <Alert>{error}</Alert>}

          <button
            type="submit"
            disabled={loading || locationsLoading || !form.location}
            className="btn-primary w-full sm:w-auto"
          >
            {loading ? (
              <>
                <Spinner className="border-ink-950/30 border-t-ink-950" /> Predicting...
              </>
            ) : (
              <>Predict Price ?</>
            )}
          </button>
        </form>

        <div className="card flex flex-col items-center justify-center border-t-4 border-gold-500 bg-ink-950 py-14 text-center">
          {!result && !loading && (
            <>
              <p className="text-xs font-bold uppercase tracking-widest text-gold-500">
                Estimate
              </p>
              <p className="mt-4 max-w-xs text-sm text-white/60">
                Your predicted property value will appear here once you submit the form.
              </p>
            </>
          )}

          {loading && (
            <Spinner className="h-8 w-8 border-white/20 border-t-gold-500" />
          )}

          {result && !loading && (
  <div style={{ textAlign: "center" }}>
    <div
      style={{
        fontSize: "13px",
        fontWeight: "700",
        letterSpacing: "0.15em",
        textTransform: "uppercase",
        color: "#d99a00",
      }}
    >
      Predicted Value
    </div>

    <div
      style={{
        marginTop: "18px",
        fontSize: "42px",
        fontWeight: "800",
        color: "#111111",
        lineHeight: "1.2",
      }}
    >
      {formatINR(result.price)}
    </div>
  </div>
)}
        </div>
      </div>
    </div>
  );
}












