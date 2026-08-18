import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { profileApi } from "../api/client";
import { Alert, PageHeader, Spinner } from "../components/ui";

export default function Profile() {
  const { user, updateUser } = useAuth();
  const [form, setForm] = useState({ name: "", email: "", phone: "" });
  const [pwForm, setPwForm] = useState({ current_password: "", new_password: "" });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pwSaving, setPwSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [pwMessage, setPwMessage] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await profileApi.get();
        if (!cancelled) {
          setForm({ name: data.name || "", email: data.email || "", phone: data.phone || "" });
        }
      } catch {
        if (!cancelled) {
          setForm({ name: user?.name || "", email: user?.email || "", phone: user?.phone || "" });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChange = (e) => setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
  const handlePwChange = (e) => setPwForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage(null);
    try {
      const updated = await profileApi.update(form);
      updateUser(updated || form);
      setMessage({ kind: "success", text: "Profile updated." });
    } catch {
      setMessage({ kind: "error", text: "Couldn't save your profile. Please try again." });
    } finally {
      setSaving(false);
    }
  };

  const handlePasswordChange = async (e) => {
    e.preventDefault();
    setPwSaving(true);
    setPwMessage(null);
    try {
      await profileApi.changePassword(pwForm);
      setPwMessage({ kind: "success", text: "Password updated." });
      setPwForm({ current_password: "", new_password: "" });
    } catch {
      setPwMessage({ kind: "error", text: "Couldn't update your password. Check your current password." });
    } finally {
      setPwSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="grid min-h-[60vh] place-items-center">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  return (
    <div className="container-x py-14">
      <PageHeader eyebrow="Account" title="Profile" sub="Manage your personal details and password." />

      <div className="grid gap-8 lg:grid-cols-2">
        <form onSubmit={handleSave} className="card space-y-5">
          <h3 className="text-lg font-bold text-ink-950">Personal details</h3>
          {message && <Alert kind={message.kind}>{message.text}</Alert>}
          <div>
            <label className="field-label" htmlFor="name">
              Full name
            </label>
            <input
              id="name"
              name="name"
              className="field-input"
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
              className="field-input"
              value={form.email}
              onChange={handleChange}
            />
          </div>
          <div>
            <label className="field-label" htmlFor="phone">
              Phone
            </label>
            <input
              id="phone"
              name="phone"
              className="field-input"
              placeholder="+91 98765 43210"
              value={form.phone}
              onChange={handleChange}
            />
          </div>
          <button type="submit" disabled={saving} className="btn-primary">
            {saving ? "Saving…" : "Save changes"}
          </button>
        </form>

        <form onSubmit={handlePasswordChange} className="card space-y-5">
          <h3 className="text-lg font-bold text-ink-950">Change password</h3>
          {pwMessage && <Alert kind={pwMessage.kind}>{pwMessage.text}</Alert>}
          <div>
            <label className="field-label" htmlFor="current_password">
              Current password
            </label>
            <input
              id="current_password"
              name="current_password"
              type="password"
              required
              className="field-input"
              value={pwForm.current_password}
              onChange={handlePwChange}
            />
          </div>
          <div>
            <label className="field-label" htmlFor="new_password">
              New password
            </label>
            <input
              id="new_password"
              name="new_password"
              type="password"
              required
              minLength={6}
              className="field-input"
              value={pwForm.new_password}
              onChange={handlePwChange}
            />
          </div>
          <button type="submit" disabled={pwSaving} className="btn-outline-dark">
            {pwSaving ? "Updating…" : "Update password"}
          </button>
        </form>
      </div>
    </div>
  );
}
