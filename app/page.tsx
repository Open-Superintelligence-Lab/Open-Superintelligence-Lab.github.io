import Hero from '@/components/Hero'
import About from '@/components/About'
import Research from '@/components/Research'
import Contact from '@/components/Contact'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen">
      <Hero />
      <About />
      <Research />
      <Contact />
      <Footer />
    </main>
  )
}
