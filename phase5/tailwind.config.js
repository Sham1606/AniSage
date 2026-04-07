/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            colors: {
                void: '#080c18',
                deep: '#0d1120',
                surface: '#0f1628',
                panel: '#131b2e',
                card: '#161f35',
                border: '#1e2640',
                muted: '#2a3558',
                dim: '#4a5680',
                azure: '#3b82f6',
                glow: '#60a5fa',
                sky: '#93c5fd',
                gold: '#f59e0b',
                amber: '#fbbf24',
                sakura: '#f472b6',
                sage: '#34d399',
            },
            fontFamily: {
                display: ['"Syne"', 'sans-serif'],
                body: ['"DM Sans"', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
            },
            animation: {
                'fade-in': 'fadeIn 0.3s ease forwards',
                'slide-up': 'slideUp 0.3s ease forwards',
            },
            keyframes: {
                fadeIn: { from: { opacity: '0' }, to: { opacity: '1' } },
                slideUp: { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
            },
        },
    },
    plugins: [],
}