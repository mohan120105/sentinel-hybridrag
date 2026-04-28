/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            color: '#d1d5db',
            a: { color: '#60a5fa' },
            strong: { color: '#f3f4f6' },
            code: { color: '#f9fafb', backgroundColor: '#374151', padding: '0.15em 0.3em', borderRadius: '0.25rem' },
            'code::before': { content: '""' },
            'code::after': { content: '""' },
            thead: { color: '#f3f4f6', borderBottomColor: '#4b5563' },
            'tbody tr': { borderBottomColor: '#374151' },
          },
        },
      },
    },
  },
  plugins: [],
}
