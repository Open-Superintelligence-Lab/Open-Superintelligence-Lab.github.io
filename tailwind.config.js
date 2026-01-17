/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
  	extend: {
  		fontFamily: {
  			sans: [
  				'var(--font-geist-sans)',
  				'system-ui',
  				'sans-serif'
  			],
  			mono: [
  				'var(--font-geist-mono)',
  				'monospace'
  			]
  		},
  		colors: {
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
			chart: {
				'1': 'hsl(var(--chart-1))',
				'2': 'hsl(var(--chart-2))',
				'3': 'hsl(var(--chart-3))',
				'4': 'hsl(var(--chart-4))',
				'5': 'hsl(var(--chart-5))'
			},
			gradient: {
				start: 'hsl(var(--gradient-start))',
				mid: 'hsl(var(--gradient-mid))',
				end: 'hsl(var(--gradient-end))',
				accent: {
					'1': 'hsl(var(--gradient-accent-1))',
					'2': 'hsl(var(--gradient-accent-2))',
					'3': 'hsl(var(--gradient-accent-3))'
				}
			},
			math: {
				primary: 'hsl(var(--math-primary))',
				secondary: 'hsl(var(--math-secondary))',
				accent: 'hsl(var(--math-accent))',
				text: 'hsl(var(--math-text))',
				relation: 'hsl(var(--math-relation))',
				bracket: 'hsl(var(--math-bracket))'
			},
			syntax: {
				bg: 'hsl(var(--syntax-bg))',
				text: 'hsl(var(--syntax-text))',
				keyword: 'hsl(var(--syntax-keyword))',
				string: 'hsl(var(--syntax-string))',
				comment: 'hsl(var(--syntax-comment))',
				number: 'hsl(var(--syntax-number))',
				function: 'hsl(var(--syntax-function))',
				variable: 'hsl(var(--syntax-variable))'
			},
			scrollbar: {
				track: 'hsl(var(--scrollbar-track))',
				thumb: 'hsl(var(--scrollbar-thumb))',
				'humb-hover': 'hsl(var(--scrollbar-thumb-hover))'
			}
		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
}
