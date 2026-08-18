export function StatCard({ label, value, sub }) {
  return (
    <div className="card">
      <p className="text-xs font-bold uppercase tracking-widest text-ink-600">{label}</p>
      <p className="mt-2 text-3xl font-display text-ink-950">{value}</p>
      {sub && <p className="mt-1 text-sm text-ink-600">{sub}</p>}
    </div>
  );
}

export function EmptyState({ title, body, action }) {
  return (
    <div className="card flex flex-col items-center gap-3 border-dashed py-16 text-center">
      <div className="grid h-14 w-14 place-items-center rounded-full bg-gold-50 text-gold-600">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M4 21V9l8-6 8 6v12" strokeLinejoin="round" />
          <path d="M9 21v-8h6v8" strokeLinejoin="round" />
        </svg>
      </div>
      <h3 className="text-lg font-bold text-ink-950">{title}</h3>
      <p className="max-w-sm text-sm text-ink-600">{body}</p>
      {action}
    </div>
  );
}

export function Alert({ kind = "error", children }) {
  const styles =
    kind === "error"
      ? "border-red-200 bg-red-50 text-red-700"
      : "border-green-200 bg-green-50 text-green-700";
  return <div className={`rounded-lg border px-4 py-3 text-sm font-medium ${styles}`}>{children}</div>;
}

export function Spinner({ className = "" }) {
  return (
    <div
      className={`h-5 w-5 animate-spin rounded-full border-2 border-ink-950/20 border-t-gold-500 ${className}`}
    />
  );
}

export function PageHeader({ eyebrow, title, sub }) {
  return (
    <div className="mb-10">
      {eyebrow && <span className="eyebrow border-gold-600 text-gold-600">{eyebrow}</span>}
      <h1 className="mt-4 font-display text-3xl text-ink-950 sm:text-4xl">{title}</h1>
      {sub && <p className="mt-3 max-w-2xl text-ink-600">{sub}</p>}
    </div>
  );
}
