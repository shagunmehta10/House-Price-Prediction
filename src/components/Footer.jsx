export default function Footer() {
  return (
    <footer className="border-t border-ink-950/10 bg-ink-950">
      <div className="container-x flex flex-col items-center justify-between gap-4 py-10 text-sm text-white/60 sm:flex-row">
        <p className="font-display text-lg text-white">
          House<span className="text-gold-500">AI</span>
        </p>
        <p>&copy; {new Date().getFullYear()} HouseAI. Machine-learning powered property estimation across India.</p>
      </div>
    </footer>
  );
}
