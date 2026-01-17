export interface Theme {
    name: string
    label: string
    cssVars: {
        light: Record<string, string>
        dark: Record<string, string>
    }
}

export const themes: Theme[] = [
    {
        name: "zinc",
        label: "Zinc",
        cssVars: {
            light: {
                "--background": "0 0% 100%",
                "--foreground": "240 10% 3.9%",
                "--card": "0 0% 100%",
                "--card-foreground": "240 10% 3.9%",
                "--popover": "0 0% 100%",
                "--popover-foreground": "240 10% 3.9%",
                "--primary": "240 5.9% 10%",
                "--primary-foreground": "0 0% 98%",
                "--secondary": "240 4.8% 95.9%",
                "--secondary-foreground": "240 5.9% 10%",
                "--muted": "240 4.8% 95.9%",
                "--muted-foreground": "240 3.8% 46.1%",
                "--accent": "240 4.8% 95.9%",
                "--accent-foreground": "240 5.9% 10%",
                "--destructive": "0 84.2% 60.2%",
                "--destructive-foreground": "0 0% 98%",
                "--border": "240 5.9% 90%",
                "--input": "240 5.9% 90%",
                "--ring": "240 10% 3.9%",
                "--radius": "0.5rem",
            },
            dark: {
                "--background": "240 10% 3.9%",
                "--foreground": "0 0% 98%",
                "--card": "240 10% 3.9%",
                "--card-foreground": "0 0% 98%",
                "--popover": "240 10% 3.9%",
                "--popover-foreground": "0 0% 98%",
                "--primary": "0 0% 98%",
                "--primary-foreground": "240 5.9% 10%",
                "--secondary": "240 3.7% 15.9%",
                "--secondary-foreground": "0 0% 98%",
                "--muted": "240 3.7% 15.9%",
                "--muted-foreground": "240 5% 64.9%",
                "--accent": "240 3.7% 15.9%",
                "--accent-foreground": "0 0% 98%",
                "--destructive": "0 62.8% 30.6%",
                "--destructive-foreground": "0 0% 98%",
                "--border": "240 3.7% 15.9%",
                "--input": "240 3.7% 15.9%",
                "--ring": "240 4.9% 83.9%",
            },
        },
    },
    {
        name: "ocean",
        label: "Ocean",
        cssVars: {
            light: {
                "--background": "0 0% 100%",
                "--foreground": "222.2 84% 4.9%",
                "--card": "0 0% 100%",
                "--card-foreground": "222.2 84% 4.9%",
                "--popover": "0 0% 100%",
                "--popover-foreground": "222.2 84% 4.9%",
                "--primary": "221.2 83.2% 53.3%",
                "--primary-foreground": "210 40% 98%",
                "--secondary": "210 40% 96.1%",
                "--secondary-foreground": "222.2 47.4% 11.2%",
                "--muted": "210 40% 96.1%",
                "--muted-foreground": "215.4 16.3% 46.9%",
                "--accent": "210 40% 96.1%",
                "--accent-foreground": "222.2 47.4% 11.2%",
                "--destructive": "0 84.2% 60.2%",
                "--destructive-foreground": "210 40% 98%",
                "--border": "214.3 31.8% 91.4%",
                "--input": "214.3 31.8% 91.4%",
                "--ring": "221.2 83.2% 53.3%",
                "--radius": "0.5rem",
            },
            dark: {
                "--background": "222.2 84% 4.9%",
                "--foreground": "210 40% 98%",
                "--card": "222.2 84% 4.9%",
                "--card-foreground": "210 40% 98%",
                "--popover": "222.2 84% 4.9%",
                "--popover-foreground": "210 40% 98%",
                "--primary": "217.2 91.2% 59.8%",
                "--primary-foreground": "222.2 47.4% 11.2%",
                "--secondary": "217.2 32.6% 17.5%",
                "--secondary-foreground": "210 40% 98%",
                "--muted": "217.2 32.6% 17.5%",
                "--muted-foreground": "215 20.2% 65.1%",
                "--accent": "217.2 32.6% 17.5%",
                "--accent-foreground": "210 40% 98%",
                "--destructive": "0 62.8% 30.6%",
                "--destructive-foreground": "210 40% 98%",
                "--border": "217.2 32.6% 17.5%",
                "--input": "217.2 32.6% 17.5%",
                "--ring": "224.3 76.3% 48%",
            },
        },
    },
    {
        name: "midnight",
        label: "Midnight",
        cssVars: {
            light: {
                "--background": "0 0% 100%",
                "--foreground": "224 71.4% 4.1%",
                "--card": "0 0% 100%",
                "--card-foreground": "224 71.4% 4.1%",
                "--popover": "0 0% 100%",
                "--popover-foreground": "224 71.4% 4.1%",
                "--primary": "262.1 83.3% 57.8%",
                "--primary-foreground": "210 20% 98%",
                "--secondary": "220 14.3% 95.9%",
                "--secondary-foreground": "220.9 39.3% 11%",
                "--muted": "220 14.3% 95.9%",
                "--muted-foreground": "220 8.9% 46.1%",
                "--accent": "220 14.3% 95.9%",
                "--accent-foreground": "220.9 39.3% 11%",
                "--destructive": "0 84.2% 60.2%",
                "--destructive-foreground": "210 20% 98%",
                "--border": "220 13% 91%",
                "--input": "220 13% 91%",
                "--ring": "262.1 83.3% 57.8%",
                "--radius": "0.5rem",
            },
            dark: {
                "--background": "224 71.4% 4.1%",
                "--foreground": "210 20% 98%",
                "--card": "224 71.4% 4.1%",
                "--card-foreground": "210 20% 98%",
                "--popover": "224 71.4% 4.1%",
                "--popover-foreground": "210 20% 98%",
                "--primary": "263.4 70% 50.4%",
                "--primary-foreground": "210 20% 98%",
                "--secondary": "215 27.9% 16.9%",
                "--secondary-foreground": "210 20% 98%",
                "--muted": "215 27.9% 16.9%",
                "--muted-foreground": "217.9 10.6% 64.9%",
                "--accent": "215 27.9% 16.9%",
                "--accent-foreground": "210 20% 98%",
                "--destructive": "0 62.8% 30.6%",
                "--destructive-foreground": "210 20% 98%",
                "--border": "215 27.9% 16.9%",
                "--input": "215 27.9% 16.9%",
                "--ring": "263.4 70% 50.4%",
            },
        },
    },
]

export function applyTheme(themeName: string, isDark: boolean = true) {
    const theme = themes.find((t) => t.name === themeName) || themes[0]
    const vars = isDark ? theme.cssVars.dark : theme.cssVars.light

    const root = document.documentElement

    Object.entries(vars).forEach(([key, value]) => {
        root.style.setProperty(key, value)
    })
}
