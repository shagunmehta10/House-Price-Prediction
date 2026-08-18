/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0a0a",
          900: "#121212",
          800: "#1a1a1a",
          700: "#242424",
          600: "#333333",
        },
        gold: {
          DEFAULT: "#F5B700",
          50: "#FFF9E5",
          100: "#FFF0BF",
          300: "#FFDB66",
          400: "#FFC91F",
          500: "#F5B700",
          600: "#CC9800",
          700: "#A37900",
        },
      },
      fontFamily: {
        display: ["Archivo Black", "Arial Black", "sans-serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        gold: "0 0 0 1px rgba(245,183,0,0.4), 0 20px 40px -20px rgba(245,183,0,0.35)",
      },
    },
  },
  plugins: [],
};
