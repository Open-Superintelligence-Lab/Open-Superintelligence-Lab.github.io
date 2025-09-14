import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Open Superintelligence Lab',
  description: 'Advancing AI research and development towards beneficial superintelligence',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
