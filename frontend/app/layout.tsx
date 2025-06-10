import './globals.css'

export const metadata = {
  title: 'SafeVision - Violence Detection',
  description: 'AI-powered CCTV violence monitoring system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="">
        <main className="">
          {children}
        </main>
      </body>
    </html>
  )
}