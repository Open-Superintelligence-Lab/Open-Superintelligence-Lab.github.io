# Open Superintelligence Lab

A modern, responsive website for the Open Superintelligence Lab, built with Next.js and deployed to GitHub Pages.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Open-Superintelligence-Lab/Open-Superintelligence-Lab.github.io.git
cd Open-Superintelligence-Lab.github.io
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📦 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build the application for production
- `npm run start` - Start production server
- `npm run export` - Build and export static files
- `npm run deploy` - Deploy to GitHub Pages

## 🏗️ Project Structure

```
├── app/                    # Next.js App Router
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/            # React components
│   ├── Hero.tsx           # Hero section
│   ├── About.tsx          # About section
│   ├── Research.tsx       # Research areas
│   ├── Contact.tsx        # Contact section
│   └── Footer.tsx         # Footer
├── .github/workflows/     # GitHub Actions
└── public/               # Static assets
```

## 🎨 Features

- **Modern Design**: Clean, professional layout with gradient backgrounds
- **Responsive**: Mobile-first design that works on all devices
- **Fast**: Optimized for performance with Next.js
- **SEO Ready**: Proper meta tags and semantic HTML
- **Accessible**: WCAG compliant components

## 🚀 Deployment

This site is automatically deployed to GitHub Pages using GitHub Actions. Every push to the `main` branch triggers a deployment.

### Manual Deployment

If you need to deploy manually:

```bash
npm run export
```

This will create an `out` folder with static files that can be served by GitHub Pages.

## 🔧 Configuration

The site is configured for GitHub Pages deployment in `next.config.js`:

- Static export enabled
- Asset prefix set for GitHub Pages
- Base path configured for repository name

## 📝 Customization

### Adding New Sections

1. Create a new component in `components/`
2. Import and add it to `app/page.tsx`
3. Style with Tailwind CSS classes

### Changing Colors

The site uses a blue-purple gradient theme. To change colors, update the gradient classes in:
- `components/Hero.tsx`
- `components/Footer.tsx`

### Updating Content

Edit the component files in `components/` to update text, links, and other content.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit: `git commit -m 'Add feature'`
5. Push: `git push origin feature-name`
6. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- **Live Site**: [https://open-superintelligence-lab.github.io/Open-Superintelligence-Lab.github.io](https://open-superintelligence-lab.github.io/Open-Superintelligence-Lab.github.io)
- **GitHub**: [https://github.com/Open-Superintelligence-Lab/Open-Superintelligence-Lab.github.io](https://github.com/Open-Superintelligence-Lab/Open-Superintelligence-Lab.github.io)

---

Built with ❤️ by the Open Superintelligence Lab team.
