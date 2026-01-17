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
			},
			// Override default Tailwind colors to use palette variables
			slate: {
				950: 'hsl(var(--background))',
				900: 'hsl(var(--background))',
				800: 'hsl(var(--card))',
				700: 'hsl(var(--secondary))',
				600: 'hsl(var(--border))',
				500: 'hsl(var(--muted))',
				400: 'hsl(var(--muted-foreground))',
				300: 'hsl(var(--muted-foreground))',
			},
			gray: {
				100: 'hsl(var(--foreground))',
				200: 'hsl(var(--foreground))',
				300: 'hsl(var(--muted-foreground))',
				400: 'hsl(var(--muted-foreground))',
				700: 'hsl(var(--border))',
				800: 'hsl(var(--card))',
				900: 'hsl(var(--background))',
			},
			blue: {
				600: 'hsl(var(--gradient-accent-1))',
				500: 'hsl(var(--gradient-accent-1))',
				400: 'hsl(var(--gradient-accent-1))',
			},
			purple: {
				600: 'hsl(var(--gradient-accent-2))',
				500: 'hsl(var(--gradient-accent-2))',
				400: 'hsl(var(--gradient-accent-2))',
			},
			cyan: {
				600: 'hsl(var(--gradient-accent-3))',
				500: 'hsl(var(--gradient-accent-3))',
				400: 'hsl(var(--gradient-accent-3))',
			},
			indigo: {
				900: 'hsl(var(--indigo-900))',
				600: 'hsl(var(--indigo-600))',
				500: 'hsl(var(--indigo-500))',
				400: 'hsl(var(--indigo-400))',
				300: 'hsl(var(--indigo-300))',
			},
			green: {
				900: 'hsl(var(--green-900))',
				600: 'hsl(var(--green-600))',
				500: 'hsl(var(--green-500))',
				400: 'hsl(var(--green-400))',
				300: 'hsl(var(--green-300))',
			},
			emerald: {
				900: 'hsl(var(--emerald-900))',
				600: 'hsl(var(--emerald-600))',
				500: 'hsl(var(--emerald-500))',
				400: 'hsl(var(--emerald-400))',
				300: 'hsl(var(--emerald-300))',
			},
			amber: {
				600: 'hsl(var(--amber-600))',
				500: 'hsl(var(--amber-500))',
				400: 'hsl(var(--amber-400))',
				300: 'hsl(var(--amber-300))',
			},
			orange: {
				600: 'hsl(var(--orange-600))',
				500: 'hsl(var(--orange-500))',
				400: 'hsl(var(--orange-400))',
			},
			violet: {
				900: 'hsl(var(--violet-900))',
				600: 'hsl(var(--violet-600))',
				500: 'hsl(var(--violet-500))',
				400: 'hsl(var(--violet-400))',
			},
			pink: {
				600: 'hsl(var(--pink-600))',
				500: 'hsl(var(--pink-500))',
				400: 'hsl(var(--pink-400))',
			},
			white: 'hsl(var(--foreground))'
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
